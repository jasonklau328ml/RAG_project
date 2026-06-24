from pathlib import Path

from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import DEFAULT_CHAT_SYSTEM_PROMPT, DEFAULT_MEMORY_TOKEN_LIMIT, DEFAULT_TOP_K
from .knowledge_base import ChromaKnowledgeBase
from .session_store import JsonChatSessionStore


class RagNewsChatbot:
    """Facade for single-turn RAG, persistent chats, and UI session switching."""

    def __init__(
        self,
        knowledge_base: ChromaKnowledgeBase,
        session_store: JsonChatSessionStore,
        final_top_k: int = DEFAULT_TOP_K,
        memory_token_limit: int = DEFAULT_MEMORY_TOKEN_LIMIT,
        llm=None,
        chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
    ):
        self.knowledge_base = knowledge_base
        self.session_store = session_store
        self.memory_token_limit = memory_token_limit
        self.llm = llm if llm is not None else Settings.llm
        self.chat_system_prompt = chat_system_prompt

        self.hybrid_retriever = knowledge_base.hybrid_retriever(final_top_k=final_top_k)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.hybrid_retriever,
            llm=self.llm,
        )
        self.chat_sessions: dict[str, dict[str, object]] = {}

    def ask(self, question: str):
        return self.query_engine.query(question)

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

    def rename_chat(self, old_chat_id: str, new_chat_id: str, overwrite: bool = False) -> str:
        if old_chat_id == new_chat_id:
            return old_chat_id

        new_path = self.session_store.rename_chat(old_chat_id, new_chat_id, overwrite=overwrite)

        if old_chat_id in self.chat_sessions:
            self.chat_sessions[new_chat_id] = self.chat_sessions.pop(old_chat_id)
            self.chat_sessions[new_chat_id]["session_path"] = new_path

        return new_chat_id

    def delete_chat(self, chat_id: str, *, close_open_session: bool = True, missing_ok: bool = False) -> bool:
        deleted = self.session_store.delete_chat(chat_id, missing_ok=missing_ok)
        if close_open_session:
            self.chat_sessions.pop(chat_id, None)
        return deleted

    def list_chat_ids(self) -> list[str]:
        return self.session_store.list_chat_ids()

    def count_chat_ids(self) -> int:
        return self.session_store.count_chat_ids()

    def chat(self, chat_id: str, message: str):
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}. Call open_chat first.")

        session = self.chat_sessions[chat_id]
        if session["chat_engine"] is None:
            session["chat_engine"] = ContextChatEngine.from_defaults(
                retriever=self.hybrid_retriever,
                memory=session["memory"],
                llm=self.llm,
                system_prompt=self.chat_system_prompt,
            )

        response = session["chat_engine"].chat(message)
        self.session_store.save(chat_id, self.chat_sessions[chat_id]["memory"])
        return response

    def memory_messages(self, chat_id: str) -> list[ChatMessage]:
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}")
        return self.chat_sessions[chat_id]["memory"].get_all()

    def show_history(self, chat_id: str) -> None:
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}")
        for message in self.chat_sessions[chat_id]["memory"].get_all():
            role = message.role.value if hasattr(message.role, "value") else str(message.role)
            print(f"{role}: {message.content}")
            if role == "assistant":
                print("-" * 40)

    def list_saved_chats(self) -> list[Path]:
        return self.session_store.list()

    def print_sources(self, response, max_sources: int = 3) -> None:
        print_sources(response, max_sources=max_sources)


def print_sources(response, max_sources: int = 3) -> None:
    source_nodes = getattr(response, "source_nodes", []) or []
    for rank, source_node in enumerate(source_nodes[:max_sources], start=1):
        metadata = source_node.node.metadata
        article_date = metadata.get("article_date", "unknown date")
        article_title = metadata.get("article_title", metadata.get("file_name", "unknown article"))
        print(f"\nSource {rank}: {article_date} | {article_title}")
        print(source_node.node.get_content()[:800])
