from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from .chatbot import RagNewsChatbot
from .config import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_MEMORY_TOKEN_LIMIT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
)
from .knowledge_base import ChromaKnowledgeBase
from .session_store import JsonChatSessionStore


def configure_llama_index(
    news_dir: Path,
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> None:
    if not news_dir.exists():
        raise FileNotFoundError(f"News folder not found: {news_dir}")

    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.llm = Ollama(model=ollama_model, request_timeout=120.0)


def create_rag_app(
    *,
    chroma_dir: Path,
    session_dir: Path,
    news_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    final_top_k: int = DEFAULT_TOP_K,
    memory_token_limit: int = DEFAULT_MEMORY_TOKEN_LIMIT,
    chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> RagNewsChatbot:
    configure_llama_index(
        news_dir=news_dir,
        embed_model_name=embed_model_name,
        ollama_model=ollama_model,
    )

    knowledge_base = ChromaKnowledgeBase(chroma_dir, collection_name)
    session_store = JsonChatSessionStore(
        session_dir,
        collection_name=collection_name,
        embed_model_name=embed_model_name,
        llm_model=ollama_model,
    )
    return RagNewsChatbot(
        knowledge_base=knowledge_base,
        session_store=session_store,
        final_top_k=final_top_k,
        memory_token_limit=memory_token_limit,
        chat_system_prompt=chat_system_prompt,
    )
