from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# DEFAULT_INGESTION_COLLECTION_NAME = "news_chat_multilingual_e5_base"
DEFAULT_INGESTION_COLLECTION_NAME = "run_testing"
# DEFAULT_INGESTION_EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
DEFAULT_INGESTION_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_INGESTION_CHUNK_SIZE = 800
DEFAULT_INGESTION_CHUNK_OVERLAP = 120
# DEFAULT_INGESTION_NEWS_SOURCE_DIR_NAMES = (
#     "hk_free_press_news",
#     "hk01_news",
#     "the_standard_news",
# )
DEFAULT_INGESTION_NEWS_SOURCE_DIR_NAMES = (
    "hk_free_press_news",
    # "hk01_news",
    # "the_standard_news",
)

@dataclass(frozen=True)
class IngestionPaths:
    """Filesystem paths used by the insertion notebook and ingestion scripts."""

    project_root: Path
    news_dirs: tuple[Path, ...]
    chroma_dir: Path


@dataclass(frozen=True)
class IngestionRunConfig:
    """Stable insertion settings shared by notebooks and scripts.

    This intentionally duplicates some retrieval config values so insertion.ipynb can be
    understood and changed without mentally jumping into retrieval-specific defaults.
    """

    project_root: Path
    paths: IngestionPaths
    collection_name: str = DEFAULT_INGESTION_COLLECTION_NAME
    chunk_size: int = DEFAULT_INGESTION_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_INGESTION_CHUNK_OVERLAP
    embed_model_name: str = DEFAULT_INGESTION_EMBED_MODEL_NAME


@dataclass(frozen=True)
class IngestionObservabilityConfig:
    """Stable ingestion observability settings shared by notebooks and scripts.

    Keep reusable defaults here so the notebook stays focused on running the workflow.
    Values that change for a one-off experiment can still be overridden inside the notebook.
    """

    phoenix_enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "rag-news-ingestion"
    phoenix_dataset_name: str = "rag-news-ingestion-chunk-metrics"
    phoenix_project_description: str = (
        "RAG ingestion pipeline for HK news corpus. "
        "Tracks chunking, embedding, vector insertion, and dataset export metrics."
    )
    chunking_method: str = "SentenceSplitter"


DEFAULT_INGESTION_OBSERVABILITY_CONFIG = IngestionObservabilityConfig()


def default_ingestion_news_dirs(project_root: Path) -> tuple[Path, ...]:
    """Return the text news folders used by the insertion pipeline."""
    return tuple(project_root / "data" / dir_name for dir_name in DEFAULT_INGESTION_NEWS_SOURCE_DIR_NAMES)


def default_ingestion_paths(project_root: Path) -> IngestionPaths:
    """Build insertion paths from the current project root."""
    return IngestionPaths(
        project_root=project_root,
        news_dirs=default_ingestion_news_dirs(project_root),
        chroma_dir=project_root / "chromadb_store",
    )


def default_ingestion_run_config(project_root: Path) -> IngestionRunConfig:
    """Build the complete insertion config object consumed by insertion.ipynb."""
    return IngestionRunConfig(
        project_root=project_root,
        paths=default_ingestion_paths(project_root),
    )


def build_ingestion_trace_attributes(
    *,
    config: IngestionObservabilityConfig = DEFAULT_INGESTION_OBSERVABILITY_CONFIG,
    run_config: IngestionRunConfig,
) -> dict[str, str | int]:
    """Build Phoenix span attributes that describe the ingestion run configuration."""
    return {
        "ingestion.project_name": config.phoenix_project_name,
        "ingestion.dataset_name": config.phoenix_dataset_name,
        "ingestion.chunking_method": config.chunking_method,
        "ingestion.chunk_size": run_config.chunk_size,
        "ingestion.chunk_overlap": run_config.chunk_overlap,
        "ingestion.embedding_model": run_config.embed_model_name,
        "ingestion.collection_name": run_config.collection_name,
    }