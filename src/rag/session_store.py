import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from .config import DEFAULT_COLLECTION_NAME, DEFAULT_EMBED_MODEL_NAME, DEFAULT_OLLAMA_MODEL


class JsonChatSessionStore:
    """File-backed chat storage, using one JSON file per chat session."""

    def __init__(
        self,
        session_dir: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
        llm_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self.session_dir = session_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.llm_model = llm_model
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
            "collection_name": self.collection_name,
            "embedding_model": self.embed_model_name,
            "llm_model": self.llm_model,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [self.message_to_dict(message) for message in memory.get_all()],
        }

    def save(self, chat_id: str, memory: ChatMemoryBuffer) -> Path:
        session_path = self.path_for(chat_id)
        temp_path = session_path.with_suffix(".tmp")
        payload = self.payload_from_memory(chat_id, memory)

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
