import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from .chatbot import RagNewsChatbot
from .config import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_HUGGINGFACE_MODEL_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MEMORY_TOKEN_LIMIT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
    HUGGINGFACE_CHAT_MODELS,
    LLM_PROVIDER_HUGGINGFACE,
    LLM_PROVIDER_OLLAMA,
)
from .embeddings import create_embedding_model
from .huggingface_llm import HuggingFaceChatLLM
from .knowledge_base import ChromaKnowledgeBase
from .session_store import JsonChatSessionStore


LlmProvider = Literal["ollama", "huggingface"]


def _normalize_news_dirs(news_dir: Path | Iterable[Path]) -> list[Path]:
    if isinstance(news_dir, Path):
        return [news_dir]

    normalized = [Path(path) for path in news_dir]
    if not normalized:
        raise ValueError("At least one news source directory must be provided.")
    return normalized


def _load_huggingface_api_key() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY"):
        api_key = os.getenv(env_name, "").strip()
        if api_key:
            return api_key

    try:
        from src import utils_private
    except ImportError as error:
        raise RuntimeError("Hugging Face API key not found. Add hf_api_key to src/utils_private.py.") from error

    api_key = getattr(utils_private, "hf_api_key", "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError("Hugging Face API key is empty. Set hf_api_key in src/utils_private.py.")
    return api_key.strip()


def resolve_huggingface_model(model_key_or_id: str) -> str:
    return HUGGINGFACE_CHAT_MODELS.get(model_key_or_id, model_key_or_id)


def create_llm(
    *,
    llm_provider: LlmProvider = DEFAULT_LLM_PROVIDER,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    huggingface_model: str = DEFAULT_HUGGINGFACE_MODEL_KEY,
    huggingface_provider: str = "auto",
    huggingface_api_key: str | None = None,
):
    if llm_provider == LLM_PROVIDER_OLLAMA:
        return Ollama(model=ollama_model, request_timeout=120.0), ollama_model

    if llm_provider == LLM_PROVIDER_HUGGINGFACE:
        model_id = resolve_huggingface_model(huggingface_model)
        return (
            HuggingFaceChatLLM(
                model=model_id,
                api_key=huggingface_api_key or _load_huggingface_api_key(),
                provider=huggingface_provider,
            ),
            model_id,
        )

    supported = ", ".join([LLM_PROVIDER_OLLAMA, LLM_PROVIDER_HUGGINGFACE])
    raise ValueError(f"Unsupported llm_provider {llm_provider!r}. Choose one of: {supported}")


def configure_llama_index(
    news_dir: Path | Iterable[Path],
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    llm_provider: LlmProvider = DEFAULT_LLM_PROVIDER,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    huggingface_model: str = DEFAULT_HUGGINGFACE_MODEL_KEY,
    huggingface_provider: str = "auto",
    huggingface_api_key: str | None = None,
) -> str:
    news_dirs = _normalize_news_dirs(news_dir)
    missing_news_dirs = [path for path in news_dirs if not path.exists()]
    if missing_news_dirs:
        missing_list = ", ".join(str(path) for path in missing_news_dirs)
        raise FileNotFoundError(f"News folder(s) not found: {missing_list}")

    Settings.embed_model = create_embedding_model(embed_model_name)
    Settings.llm, resolved_llm_model = create_llm(
        llm_provider=llm_provider,
        ollama_model=ollama_model,
        huggingface_model=huggingface_model,
        huggingface_provider=huggingface_provider,
        huggingface_api_key=huggingface_api_key,
    )
    return resolved_llm_model


def create_rag_app(
    *,
    chroma_dir: Path,
    session_dir: Path,
    news_dir: Path | Iterable[Path],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    llm_provider: LlmProvider = DEFAULT_LLM_PROVIDER,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    huggingface_model: str = DEFAULT_HUGGINGFACE_MODEL_KEY,
    huggingface_provider: str = "auto",
    huggingface_api_key: str | None = None,
    final_top_k: int = DEFAULT_TOP_K,
    memory_token_limit: int = DEFAULT_MEMORY_TOKEN_LIMIT,
    chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> RagNewsChatbot:
    resolved_llm_model = configure_llama_index(
        news_dir=news_dir,
        embed_model_name=embed_model_name,
        llm_provider=llm_provider,
        ollama_model=ollama_model,
        huggingface_model=huggingface_model,
        huggingface_provider=huggingface_provider,
        huggingface_api_key=huggingface_api_key,
    )

    knowledge_base = ChromaKnowledgeBase(chroma_dir, collection_name)
    session_store = JsonChatSessionStore(
        session_dir,
        collection_name=collection_name,
        embed_model_name=embed_model_name,
        llm_provider=llm_provider,
        llm_model=resolved_llm_model,
    )
    return RagNewsChatbot(
        knowledge_base=knowledge_base,
        session_store=session_store,
        final_top_k=final_top_k,
        memory_token_limit=memory_token_limit,
        chat_system_prompt=chat_system_prompt,
    )
