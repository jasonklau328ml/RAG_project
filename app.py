import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chainlit as cl
import chromadb
from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
NEWS_DIR = PROJECT_ROOT / "data" / "hk_free_press_news"
CHROMA_DIR = PROJECT_ROOT / "chromadb_store"
SESSION_DIR = PROJECT_ROOT / "session"
COLLECTION_NAME = "news_chat"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "gemma3:1b"
DEFAULT_TOP_K = 5
MEMORY_TOKEN_LIMIT = 3000

CHAT_SYSTEM_PROMPT = """
You are a helpful RAG chatbot for an HK Free Press news knowledge base.
Use the retrieved context as your primary evidence.
If the retrieved context does not contain enough information, say that the knowledge base does not have enough evidence.
Use the current chat history to understand follow-up questions, but do not invent facts that are not supported by the retrieved context.
""".strip()

_rag_app: "RagNewsChatbot | None" = None


def reciprocal_rank_fusion(
    semantic_results: list[NodeWithScore],
    keyword_results: list[NodeWithScore],
    top_k: int,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    """Fuse semantic and keyword rankings without manually calibrating their scores."""
    fused_scores: dict[str, float] = defaultdict(float)
    node_lookup: dict[str, NodeWithScore] = {}

    for results in (semantic_results, keyword_results):
        for rank, result in enumerate(results, start=1):
            node_id = result.node.node_id
            fused_scores[node_id] += 1.0 / (rrf_k + rank)
            node_lookup.setdefault(node_id, result)

    ranked_nodes = sorted(
        node_lookup.values(),
        key=lambda result: fused_scores[result.node.node_id],
        reverse=True,
    )

    return [
        NodeWithScore(node=result.node, score=fused_scores[result.node.node_id])
        for result in ranked_nodes[:top_k]
    ]


class HybridRetriever(BaseRetriever):
    """LlamaIndex retriever that combines Chroma semantic search and BM25 keyword search."""

    def __init__(
        self,
        semantic_retriever,
        keyword_retriever: BM25Retriever,
        final_top_k: int = DEFAULT_TOP_K,
        rrf_k: int = 60,
    ):
        self._semantic_retriever = semantic_retriever
        self._keyword_retriever = keyword_retriever
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        semantic_results = self._semantic_retriever.retrieve(query_bundle)
        keyword_results = self._keyword_retriever.retrieve(query_bundle)

        return reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=self._final_top_k,
            rrf_k=self._rrf_k,
        )


class ChromaKnowledgeBase:
    """Adapter around the persisted ChromaDB collection used by the RAG app."""

    def __init__(self, chroma_dir: Path, collection_name: str):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._index = None
        self._nodes: list[TextNode] | None = None
        self._bm25_cache: dict[int, BM25Retriever] = {}

    @property
    def collection(self):
        # Keep this lazy so Chainlit can start before the vector store is touched.
        if self._collection is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )
        return self._index

    @property
    def nodes(self) -> list[TextNode]:
        if self._nodes is None:
            records = self.collection.get(include=["documents", "metadatas"])
            documents = records.get("documents", []) or []
            metadatas = records.get("metadatas", []) or []
            ids = records.get("ids", []) or []

            self._nodes = [
                TextNode(id_=node_id, text=document_text, metadata=metadata or {})
                for node_id, document_text, metadata in zip(ids, documents, metadatas)
                if document_text
            ]

            if not self._nodes:
                raise ValueError("No stored text nodes were found in the Chroma collection.")

        return self._nodes

    def bm25_retriever(self, top_k: int) -> BM25Retriever:
        if top_k not in self._bm25_cache:
            self._bm25_cache[top_k] = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=top_k,
            )
        return self._bm25_cache[top_k]

    def hybrid_retriever(
        self,
        final_top_k: int = DEFAULT_TOP_K,
        candidate_top_k: int | None = None,
        rrf_k: int = 60,
    ) -> HybridRetriever:
        candidate_count = candidate_top_k if candidate_top_k is not None else max(final_top_k * 2, 10)
        semantic_retriever = self.index.as_retriever(similarity_top_k=candidate_count)
        keyword_retriever = self.bm25_retriever(top_k=candidate_count)

        return HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )

    def count(self) -> int:
        return self.collection.count()


class JsonChatSessionStore:
    """File-backed chat storage, using one JSON file per chat session."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def safe_name(self, chat_id: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id.strip()).strip("_")
        return safe_name or "default_chat"

    def path_for(self, chat_id: str) -> Path:
        return self.session_dir / f"{self.safe_name(chat_id)}.json"

    def display_name_for(self, session_path: Path) -> str:
        return session_path.stem.replace("_", " ")

    def message_to_dict(self, message: ChatMessage) -> dict[str, Any]:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        return {
            "role": role,
            "content": message.content or "",
            "additional_kwargs": message.additional_kwargs or {},
        }

    def message_from_dict(self, data: dict[str, Any]) -> ChatMessage:
        role_text = data.get("role", "user")
        try:
            role = MessageRole(role_text)
        except ValueError:
            role = role_text

        return ChatMessage(
            role=role,
            content=data.get("content", ""),
            additional_kwargs=data.get("additional_kwargs", {}) or {},
        )

    def memory_from_payload(self, payload: dict[str, Any], token_limit: int) -> ChatMemoryBuffer:
        memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
        for message_data in payload.get("messages", []):
            memory.put(self.message_from_dict(message_data))
        return memory

    def payload_from_memory(self, chat_id: str, memory: ChatMemoryBuffer) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "chat_id": chat_id,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBED_MODEL_NAME,
            "llm_model": OLLAMA_MODEL,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [self.message_to_dict(message) for message in memory.get_all()],
        }

    def save(self, chat_id: str, memory: ChatMemoryBuffer) -> Path:
        session_path = self.path_for(chat_id)
        temp_path = session_path.with_suffix(".tmp")
        payload = self.payload_from_memory(chat_id, memory)

        # Write through a temp file so a crash cannot leave a half-written JSON session.
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(session_path)
        return session_path

    def load(self, chat_id: str) -> dict[str, Any]:
        session_path = self.path_for(chat_id)
        if not session_path.exists():
            raise FileNotFoundError(f"Saved chat session not found: {session_path}")
        return json.loads(session_path.read_text(encoding="utf-8"))

    def exists(self, chat_id: str) -> bool:
        return self.path_for(chat_id).exists()

    def list(self) -> list[Path]:
        return sorted(self.session_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


class RagNewsChatbot:
    """Facade for single-turn RAG, persistent chats, and Chainlit session switching."""

    def __init__(
        self,
        knowledge_base: ChromaKnowledgeBase,
        session_store: JsonChatSessionStore,
        final_top_k: int = DEFAULT_TOP_K,
        memory_token_limit: int = MEMORY_TOKEN_LIMIT,
        llm=None,
    ):
        self.knowledge_base = knowledge_base
        self.session_store = session_store
        self.memory_token_limit = memory_token_limit
        self.llm = llm if llm is not None else Settings.llm

        self.hybrid_retriever = knowledge_base.hybrid_retriever(final_top_k=final_top_k)
        self.chat_sessions: dict[str, dict[str, Any]] = {}

    def open_chat(self, chat_id: str, load_existing: bool = True, overwrite: bool = False) -> str:
        if load_existing and self.session_store.exists(chat_id) and not overwrite:
            payload = self.session_store.load(chat_id)
            memory = self.session_store.memory_from_payload(payload, token_limit=self.memory_token_limit)
        else:
            memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)

        self.chat_sessions[chat_id] = {
            "memory": memory,
            "chat_engine": None,
            "session_path": self.session_store.path_for(chat_id),
        }

        self.session_store.save(chat_id, memory)
        return chat_id

    def rename_chat(self, old_chat_id: str, new_chat_id: str) -> str:
        if old_chat_id == new_chat_id:
            return old_chat_id
        if old_chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {old_chat_id}")

        self.chat_sessions[new_chat_id] = self.chat_sessions.pop(old_chat_id)
        new_path = self.session_store.save(new_chat_id, self.chat_sessions[new_chat_id]["memory"])
        self.chat_sessions[new_chat_id]["session_path"] = new_path

        old_path = self.session_store.path_for(old_chat_id)
        if old_path.exists() and old_path != new_path:
            old_path.unlink()

        return new_chat_id

    def chat(self, chat_id: str, message: str):
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}. Call open_chat first.")

        session = self.chat_sessions[chat_id]
        if session["chat_engine"] is None:
            # Building the chat engine touches the Ollama server, so do it only when an answer is needed.
            session["chat_engine"] = ContextChatEngine.from_defaults(
                retriever=self.hybrid_retriever,
                memory=session["memory"],
                llm=self.llm,
                system_prompt=CHAT_SYSTEM_PROMPT,
            )

        response = session["chat_engine"].chat(message)
        self.session_store.save(chat_id, self.chat_sessions[chat_id]["memory"])
        return response

    def memory_messages(self, chat_id: str) -> list[ChatMessage]:
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}")
        return self.chat_sessions[chat_id]["memory"].get_all()


def configure_llama_index() -> None:
    """Set global LlamaIndex defaults once for the Chainlit process."""
    if not NEWS_DIR.exists():
        raise FileNotFoundError(f"News folder not found: {NEWS_DIR}")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)


def get_rag_app() -> RagNewsChatbot:
    """Create the expensive RAG objects once, then reuse them for every Chainlit session."""
    global _rag_app
    if _rag_app is None:
        configure_llama_index()
        knowledge_base = ChromaKnowledgeBase(CHROMA_DIR, COLLECTION_NAME)
        session_store = JsonChatSessionStore(SESSION_DIR)
        _rag_app = RagNewsChatbot(
            knowledge_base=knowledge_base,
            session_store=session_store,
            final_top_k=DEFAULT_TOP_K,
        )
    return _rag_app


def new_chat_id() -> str:
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def chainlit_thread_id() -> str | None:
    """Return Chainlit's persistent UI thread id when it is available."""
    try:
        session = cl.context.session
    except Exception:
        return None

    for attribute in ("thread_id", "id"):
        value = getattr(session, attribute, None)
        if value:
            return str(value)
    return None


def title_from_first_message(message: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", message)[:8]
    if not words:
        return new_chat_id()
    return "_".join(words)


def unique_chat_id(base_chat_id: str, session_store: JsonChatSessionStore) -> str:
    candidate = session_store.safe_name(base_chat_id)
    if not session_store.exists(candidate):
        return candidate

    suffix = datetime.now().strftime("%H%M%S")
    return session_store.safe_name(f"{candidate}_{suffix}")


def current_chat_id() -> str:
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        raise RuntimeError("No active chat session. Start or load a chat first.")
    return chat_id


def session_actions(session_store: JsonChatSessionStore, limit: int = 8) -> list[cl.Action]:
    actions: list[cl.Action] = [cl.Action(name="new_chat", label="New chat", payload={})]

    for session_path in session_store.list()[:limit]:
        actions.append(
            cl.Action(
                name="load_chat",
                label=session_store.display_name_for(session_path),
                payload={"chat_id": session_path.stem},
            )
        )

    return actions


def format_sources(response, max_sources: int = 3) -> str:
    source_nodes = getattr(response, "source_nodes", []) or []
    if not source_nodes:
        return ""

    lines = ["", "**Sources**"]
    for rank, source_node in enumerate(source_nodes[:max_sources], start=1):
        metadata = source_node.node.metadata or {}
        article_date = metadata.get("article_date", "unknown date")
        article_title = metadata.get("article_title", metadata.get("file_name", "unknown article"))
        preview = source_node.node.get_content().strip().replace("\n", " ")[:500]
        lines.append(f"{rank}. **{article_date} | {article_title}**")
        lines.append(f"   {preview}")

    return "\n".join(lines)


def format_recent_history(messages: list[ChatMessage], limit: int = 8) -> str:
    if not messages:
        return "This saved chat is empty. Send a message to begin."

    lines = ["Loaded the saved chat. Recent messages:"]
    for message in messages[-limit:]:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        content = (message.content or "").strip()
        if content:
            lines.append(f"\n**{role.title()}**: {content}")
    return "".join(lines)


async def ensure_active_chat(chat_id: str | None = None, overwrite: bool = False) -> str:
    rag_app = await asyncio.to_thread(get_rag_app)
    active_chat_id = chat_id or chainlit_thread_id() or new_chat_id()
    await asyncio.to_thread(rag_app.open_chat, active_chat_id, True, overwrite)

    cl.user_session.set("chat_id", active_chat_id)
    return active_chat_id


async def send_session_picker(content: str) -> None:
    rag_app = await asyncio.to_thread(get_rag_app)
    await cl.Message(
        content=content,
        actions=session_actions(rag_app.session_store),
    ).send()


@cl.on_chat_start
async def start_session():
    chat_id = await ensure_active_chat()
    await send_session_picker(
        "Welcome. A new RAG chat session is ready. Use the buttons below to start fresh or load a saved chat. "
        f"Current session: **{chat_id}**"
    )


@cl.on_chat_resume
async def resume_session(thread: dict[str, Any]):
    chat_id = str(thread.get("id") or chainlit_thread_id() or new_chat_id())
    await ensure_active_chat(chat_id=chat_id)


@cl.action_callback("new_chat")
async def new_chat_action(action: cl.Action):
    chat_id = await ensure_active_chat(overwrite=True)
    await cl.Message(content=f"Started a new chat: **{chat_id}**").send()
    await action.remove()


@cl.action_callback("load_chat")
async def load_chat_action(action: cl.Action):
    chat_id = action.payload.get("chat_id")
    if not chat_id:
        await cl.Message(content="I could not find that saved chat id.").send()
        return

    rag_app = await asyncio.to_thread(get_rag_app)
    await ensure_active_chat(chat_id=chat_id)
    messages = await asyncio.to_thread(rag_app.memory_messages, chat_id)

    await cl.Message(content=format_recent_history(messages)).send()
    await action.remove()


@cl.on_message
async def handle_message(message: cl.Message):
    rag_app = await asyncio.to_thread(get_rag_app)
    text = message.content.strip()

    # These slash commands are a small safety net for users who prefer typing over clicking buttons.
    if text == "/chats":
        await send_session_picker("Saved chats:")
        return

    if text == "/new":
        chat_id = await ensure_active_chat(overwrite=True)
        await cl.Message(content=f"Started a new chat: **{chat_id}**").send()
        return

    if text.startswith("/load "):
        chat_id = text.removeprefix("/load ").strip()
        await ensure_active_chat(chat_id=chat_id)
        messages = await asyncio.to_thread(rag_app.memory_messages, chat_id)
        await cl.Message(content=format_recent_history(messages)).send()
        return

    chat_id = current_chat_id()
    answer = cl.Message(content="")
    await answer.send()

    try:
        response = await asyncio.to_thread(rag_app.chat, chat_id, text)
    except Exception as exc:
        answer.content = f"Sorry, the RAG app failed while answering: `{exc}`"
        await answer.update()
        return

    answer.content = f"{response.response}{format_sources(response)}"
    await answer.update()