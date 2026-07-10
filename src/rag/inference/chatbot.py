from pathlib import Path

from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine

from .inference_config import DEFAULT_CHAT_SYSTEM_PROMPT, DEFAULT_MEMORY_TOKEN_LIMIT, DEFAULT_TOP_K
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
        """Create the high-level chatbot object used by the notebook and Chainlit app.

        Purpose:
        - Hide the lower-level LlamaIndex pieces behind a simpler interface for asking questions,
          opening chats, and saving conversation memory.
        - Build the hybrid retriever and a single-turn query engine up front.

        Output:
        - Stores the chatbot state on ``self``.
        - Does not return a separate value because this is the class constructor.
        """
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
        """Answer one standalone question without using multi-turn chat memory.

        Purpose:
        - Use the retriever plus LLM once for a single question.
        - This is the simplest RAG path: retrieve context, then generate an answer.

        Output:
        - Returns the LlamaIndex response object produced by ``query_engine.query(...)``.
        """
        return self.query_engine.query(question)

    def open_chat(self, chat_id: str, load_existing: bool = True, overwrite: bool = False) -> str:
        """Create or load one persistent multi-turn chat session.

        Purpose:
        - Prepare the memory buffer that stores the conversation history for one ``chat_id``.
        - Optionally reload an existing saved JSON chat so the conversation can continue.

        Output:
        - Returns the chat id that was opened.
        - As a side effect, creates an in-memory entry inside ``self.chat_sessions`` and ensures
          a JSON session file exists on disk.
        """
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
        """Rename a saved chat session everywhere this chatbot tracks it.

        Purpose:
        - Rename the JSON session file in persistent storage.
        - Keep the in-memory session dictionary synchronized with the new id.

        Output:
        - Returns the final chat id after rename.
        """
        if old_chat_id == new_chat_id:
            return old_chat_id

        new_path = self.session_store.rename_chat(old_chat_id, new_chat_id, overwrite=overwrite)

        if old_chat_id in self.chat_sessions:
            self.chat_sessions[new_chat_id] = self.chat_sessions.pop(old_chat_id)
            self.chat_sessions[new_chat_id]["session_path"] = new_path

        return new_chat_id

    def delete_chat(self, chat_id: str, *, close_open_session: bool = True, missing_ok: bool = False) -> bool:
        """Delete a saved chat session from disk and optionally from memory.

        Purpose:
        - Remove conversations the user no longer wants to keep.
        - Optionally forget the same session from the currently running Python process.

        Output:
        - Returns ``True`` when a file was deleted.
        - Returns ``False`` only when ``missing_ok=True`` and the file did not exist.
        """
        deleted = self.session_store.delete_chat(chat_id, missing_ok=missing_ok)
        if close_open_session:
            self.chat_sessions.pop(chat_id, None)
        return deleted

    def list_chat_ids(self) -> list[str]:
        """List all saved chat ids known to the session store.

        Output:
        - Returns a list of chat id strings.
        """
        return self.session_store.list_chat_ids()

    def count_chat_ids(self) -> int:
        """Count how many saved chat sessions exist.

        Output:
        - Returns the number of saved chat ids as an integer.
        """
        return self.session_store.count_chat_ids()

    def chat(self, chat_id: str, message: str):
        """Answer one message inside a persistent multi-turn chat session.

        Purpose:
        - Reuse prior conversation history so follow-up questions make sense.
        - Lazily create a ``CondensePlusContextChatEngine`` the first time the chat is used.
        - Save updated memory back to disk after each assistant reply.

        Output:
        - Returns the LlamaIndex chat response object from ``chat_engine.chat(...)``.
        - Raises ``KeyError`` if the caller forgot to open the chat first.
        """
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}. Call open_chat first.")

        session = self.chat_sessions[chat_id]
        if session["chat_engine"] is None:
            # Condense follow-up questions with saved memory before retrieval, then answer with context.
            session["chat_engine"] = CondensePlusContextChatEngine.from_defaults(
                retriever=self.hybrid_retriever,
                memory=session["memory"],
                llm=self.llm,
                system_prompt=self.chat_system_prompt,
            )

        response = session["chat_engine"].chat(message)
        self.session_store.save(chat_id, self.chat_sessions[chat_id]["memory"])
        return response

    def memory_messages(self, chat_id: str) -> list[ChatMessage]:
        """Read the current in-memory conversation history for one chat.

        Purpose:
        - Expose the full message history so the UI can replay a saved chat.

        Output:
        - Returns a list of ``ChatMessage`` objects.
        - Raises ``KeyError`` if the chat is not currently opened.
        """
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}")
        return self.chat_sessions[chat_id]["memory"].get_all()

    def show_history(self, chat_id: str) -> None:
        """Print one chat's history to the terminal or notebook output.

        Purpose:
        - Give a quick human-readable debugging view of the stored conversation.

        Output:
        - Prints messages to stdout and returns ``None``.
        """
        if chat_id not in self.chat_sessions:
            raise KeyError(f"Chat session not opened: {chat_id}")
        for message in self.chat_sessions[chat_id]["memory"].get_all():
            role = message.role.value if hasattr(message.role, "value") else str(message.role)
            print(f"{role}: {message.content}")
            if role == "assistant":
                print("-" * 40)

    def list_saved_chats(self) -> list[Path]:
        """List the actual JSON session files on disk.

        Output:
        - Returns a list of ``Path`` objects.
        """
        return self.session_store.list()

    def print_sources(self, response, max_sources: int = 3) -> None:
        """Print the top retrieved source snippets that supported an answer.

        Purpose:
        - Help the user inspect which documents the RAG answer relied on.

        Output:
        - Prints source metadata and text snippets to stdout.
        """
        print_sources(response, max_sources=max_sources)


def print_sources(response, max_sources: int = 3) -> None:
    """Print a compact preview of the retrieved source nodes in a response.

    Purpose:
    - Turn the raw LlamaIndex ``source_nodes`` structure into readable notebook or terminal text.

    Output:
    - Prints up to ``max_sources`` source previews and returns ``None``.
    """
    source_nodes = getattr(response, "source_nodes", []) or []
    for rank, source_node in enumerate(source_nodes[:max_sources], start=1):
        metadata = source_node.node.metadata
        article_date = metadata.get("article_date", "unknown date")
        article_title = metadata.get("article_title", metadata.get("file_name", "unknown article"))
        print(f"\nSource {rank}: {article_date} | {article_title}")
        print(source_node.node.get_content()[:800])


def print_response(response) -> None:
    """Print only the assistant text from a LlamaIndex response object.

    Output:
    - Prints the response text and returns ``None``.
    """
    llm_response = getattr(response, "response", []) or []
    print(llm_response)