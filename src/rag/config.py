from dataclasses import dataclass
from pathlib import Path


# DEFAULT_COLLECTION_NAME = "news_chat_multilingual_e5_base"
DEFAULT_COLLECTION_NAME = "news_chat_multilingual_e5_base"
DEFAULT_EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
# DEFAULT_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
E5_QUERY_INSTRUCTION = "query: "
E5_TEXT_INSTRUCTION = "passage: "
# DEFAULT_NEWS_SOURCE_DIR_NAMES = (
#     "hk_free_press_news",
#     "hk01_news",
#     "the_standard_news",
# )
DEFAULT_NEWS_SOURCE_DIR_NAMES = (
    "hk_free_press_news",
)
DEFAULT_OLLAMA_MODEL = "gemma3:1b"
DEFAULT_LLM_PROVIDER = "ollama" # ollama or huggingface
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
    news_dirs: tuple[Path, ...]
    chroma_dir: Path
    session_dir: Path


def default_news_dirs(project_root: Path) -> tuple[Path, ...]:
    return tuple(project_root / "data" / dir_name for dir_name in DEFAULT_NEWS_SOURCE_DIR_NAMES)


def default_paths(project_root: Path) -> RagPaths:
    news_dirs = default_news_dirs(project_root)
    return RagPaths(
        project_root=project_root,
        # Keep news_dir for backward compatibility in notebooks/scripts that still use one folder.
        news_dir=news_dirs[0],
        news_dirs=news_dirs,
        chroma_dir=project_root / "chromadb_store",
        session_dir=project_root / "session",
    )
