from dataclasses import dataclass
from pathlib import Path


DEFAULT_COLLECTION_NAME = "news_chat"
DEFAULT_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = "gemma3:1b"
DEFAULT_LLM_PROVIDER = "huggingface" # ollama or huggingface
LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_HUGGINGFACE = "huggingface"
DEFAULT_HUGGINGFACE_MODEL_KEY = "deepseek_v3"
HUGGINGFACE_CHAT_MODELS = {
    "deepseek_v3": "deepseek-ai/DeepSeek-V3-0324",
    "minimax_m3": "MiniMaxAI/MiniMax-M1-80k",
    "qwen_3_5": "Qwen/Qwen3-235B-A22B",
}
DEFAULT_TOP_K = 5
DEFAULT_MEMORY_TOKEN_LIMIT = 3000

DEFAULT_CHAT_SYSTEM_PROMPT = """
You are a helpful RAG chatbot for an HK Free Press news knowledge base.
Use the retrieved context as your primary evidence.
If the retrieved context does not contain enough information, say that the knowledge base does not have enough evidence.
Use the current chat history to understand follow-up questions, but do not invent facts that are not supported by the retrieved context.
""".strip()


@dataclass(frozen=True)
class RagPaths:
    project_root: Path
    news_dir: Path
    chroma_dir: Path
    session_dir: Path


def default_paths(project_root: Path) -> RagPaths:
    return RagPaths(
        project_root=project_root,
        news_dir=project_root / "data" / "hk_free_press_news",
        chroma_dir=project_root / "chromadb_store",
        session_dir=project_root / "session",
    )
