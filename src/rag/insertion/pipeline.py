from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm.auto import tqdm


def upsert_phoenix_project_description(base_url: str, project_name: str, description: str) -> None:
    """Create or update a Phoenix project description for better UI context."""
    try:
        from phoenix.client import Client
    except (ImportError, ModuleNotFoundError):
        print("Phoenix client package not available; skip project description sync.")
        return

    client = Client(base_url=base_url)
    try:
        client.projects.update(project_name=project_name, description=description)
        print(f"Phoenix project description updated: {project_name}")
    except Exception:
        client.projects.create(name=project_name, description=description)
        print(f"Phoenix project created with description: {project_name}")


def list_text_files(news_dirs: Iterable[Path]) -> list[Path]:
    """Return all text news files under all configured source folders."""
    text_files: list[Path] = []
    for news_dir in news_dirs:
        text_files.extend(sorted(news_dir.rglob("*.txt")))
    return sorted(text_files)


def normalize_news_text(text: str) -> str:
    """Clean text-file artifacts while preserving article content."""
    cleaned_text = unicodedata.normalize("NFKC", text or "")
    cleaned_text = cleaned_text.replace("\x00", " ").replace("\ufeff", "").replace("\u00ad", "")
    cleaned_text = re.sub(r"\r\n?", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def parse_news_file_metadata(file_path: Path) -> dict[str, str]:
    """Extract article metadata from a news text file path."""
    file_stem = file_path.stem
    date_part, separator, title_part = file_stem.partition("_")

    article_date = date_part if separator else "unknown_date"
    article_title = title_part if separator else file_stem
    source_folder = file_path.parent.name

    return {
        "source_type": "news_txt",
        "source_folder": source_folder,
        "article_date": article_date,
        "article_title": article_title,
        "file_name": file_path.name,
        "file_path": str(file_path),
    }


def read_text_file(file_path: Path) -> str:
    """Read one news text file with small encoding fallback."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return file_path.read_text(encoding="utf-8", errors="replace")


def load_news_documents(news_dirs: Iterable[Path], show_progress: bool = True) -> list[Document]:
    """Load text news files into LlamaIndex Documents with metadata."""
    documents: list[Document] = []
    skipped_files: list[str] = []
    text_paths = list_text_files(news_dirs)

    file_iterator = text_paths
    if show_progress:
        file_iterator = tqdm(text_paths, desc="Loading and cleaning articles", unit="file")

    for text_path in file_iterator:
        raw_text = read_text_file(text_path)
        cleaned_text = normalize_news_text(raw_text)

        if not cleaned_text:
            skipped_files.append(text_path.name)
            continue

        documents.append(
            Document(
                text=cleaned_text,
                metadata=parse_news_file_metadata(text_path),
            )
        )

    if skipped_files:
        print("Warning: some text files were empty after cleaning and were skipped.")
        for file_name in skipped_files[:10]:
            print(f"- {file_name}")
        if len(skipped_files) > 10:
            print(f"... and {len(skipped_files) - 10} more skipped file(s)")

    return documents


def preview_news_documents(documents: list[Document], preview_count: int = 3) -> None:
    """Print a few loaded articles for quick quality checks."""
    for index, document in enumerate(documents[:preview_count], start=1):
        metadata = document.metadata
        print(f"\n--- Preview {index} ---")
        print(f"Source folder: {metadata.get('source_folder', 'unknown')}")
        print(f"Date: {metadata['article_date']}")
        print(f"Title: {metadata['article_title']}")
        print(document.text[:1200])


def build_text_nodes(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    show_progress: bool = True,
) -> list[TextNode]:
    """Split loaded documents into retrieval chunks and annotate chunk metadata."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    node_iterator = enumerate(nodes, start=1)
    if show_progress:
        node_iterator = enumerate(
            tqdm(nodes, desc="Annotating chunk metadata", unit="chunk"),
            start=1,
        )

    for chunk_number, node in node_iterator:
        node.metadata["chunk_number"] = chunk_number
        node.metadata["chunk_size"] = chunk_size
        node.metadata["chunk_overlap"] = chunk_overlap

    return nodes


def get_chroma_collection(
    chroma_dir: Path,
    collection_name: str,
    reset_collection: bool = False,
    metadata: dict[str, Any] | None = None,
):
    """Create or load a persistent ChromaDB collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset_collection:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except (NotFoundError, ValueError):
            print(f"Collection did not exist yet: {collection_name}")

    return client.get_or_create_collection(collection_name, metadata=metadata)


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only Chroma-compatible scalar metadata values."""
    scalar_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            scalar_metadata[key] = value
        elif value is not None:
            scalar_metadata[key] = str(value)
    return scalar_metadata


class IngestionVectorIndexer:
    """Build and update Chroma vector indexes for the ingestion pipeline."""

    def __init__(
        self,
        *,
        observer: Any,
        news_dirs: Iterable[Path],
        embed_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.observer = observer
        self.news_dirs = tuple(news_dirs)
        self.embed_model_name = embed_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_or_update_vector_index(
        self,
        text_nodes: list[TextNode],
        *,
        chroma_dir: Path,
        collection_name: str,
        reset_collection: bool = False,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        source_folders = sorted(str(news_dir) for news_dir in self.news_dirs)
        collection_metadata = {
            "embedding_model": self.embed_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "source_folder_count": len(source_folders),
            "source_folders": " | ".join(source_folders),
        }

        collection = get_chroma_collection(
            chroma_dir,
            collection_name,
            reset_collection,
            metadata=collection_metadata,
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)

        if not text_nodes:
            return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)

        batch_starts = list(range(0, len(text_nodes), batch_size))
        batch_iterator = batch_starts
        if show_progress:
            batch_iterator = tqdm(batch_starts, desc="Embedding and writing Chroma batches", unit="batch")

        for batch_index, start in enumerate(batch_iterator, start=1):
            batch_nodes = text_nodes[start : start + batch_size]
            texts = [node.get_content() for node in batch_nodes]
            ids = [node.node_id for node in batch_nodes]
            metadatas = [sanitize_chroma_metadata(node.metadata or {}) for node in batch_nodes]

            embedding_start = time.perf_counter()
            embeddings = Settings.embed_model.get_text_embedding_batch(texts, show_progress=False)
            embedding_latency = time.perf_counter() - embedding_start

            insertion_start = time.perf_counter()
            collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            insertion_latency = time.perf_counter() - insertion_start

            if self.observer is not None:
                self.observer.record_vector_batch(
                    batch_index=batch_index,
                    batch_size=len(batch_nodes),
                    remaining_items=max(len(text_nodes) - start - len(batch_nodes), 0),
                    embedding_latency_seconds=embedding_latency,
                    insertion_latency_seconds=insertion_latency,
                )

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
        if self.observer is not None:
            with self.observer.span(
                "ingestion.index_build_complete",
                {
                    "index.state": "complete",
                    "chunks.count": len(text_nodes),
                    "batches.count": len(batch_starts),
                    "collection.vector_count": collection.count(),
                },
            ):
                pass
        return index
