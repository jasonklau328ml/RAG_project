from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError


@dataclass(frozen=True)
class ChromaCollectionSummary:
    """Small, notebook-friendly summary of one ChromaDB collection."""

    name: str
    count: int
    metadata: dict[str, Any]


class ChromaVectorStoreAdmin:
    """Daily operations helper for a persistent ChromaDB vector database.

    This class is intentionally separate from the RAG retriever. Retrieval code should focus
    on answering questions, while this class handles operational tasks such as listing,
    inspecting, sampling, and deleting collections from notebooks or maintenance scripts.
    """

    def __init__(self, chroma_dir: Path):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

    def _collection_name(self, collection: Any) -> str:
        return collection if isinstance(collection, str) else collection.name

    def list_collection_names(self) -> list[str]:
        """Return all collection names in the local persistent ChromaDB directory."""
        return sorted(self._collection_name(collection) for collection in self.client.list_collections())

    def count_collections(self) -> int:
        """Return how many collections currently exist."""
        return len(self.list_collection_names())

    def get_collection(self, collection_name: str):
        """Load one collection by name and raise a clear error if it does not exist."""
        try:
            return self.client.get_collection(collection_name)
        except (NotFoundError, ValueError) as error:
            raise ValueError(f"Chroma collection not found: {collection_name}") from error

    def get_collection_metadata(self, collection_name: str) -> dict[str, Any]:
        """Return metadata stored on a collection, or an empty dict when none exists."""
        collection = self.get_collection(collection_name)
        return dict(collection.metadata or {})

    def count_records(self, collection_name: str) -> int:
        """Return the number of vectors/documents stored in one collection."""
        return self.get_collection(collection_name).count()

    def describe_collection(self, collection_name: str) -> ChromaCollectionSummary:
        """Return name, record count, and metadata for one collection."""
        collection = self.get_collection(collection_name)
        return ChromaCollectionSummary(
            name=collection_name,
            count=collection.count(),
            metadata=dict(collection.metadata or {}),
        )

    def describe_collections(self) -> list[ChromaCollectionSummary]:
        """Return summaries for every collection in the vector database."""
        return [self.describe_collection(collection_name) for collection_name in self.list_collection_names()]

    def sample_records(
        self,
        collection_name: str,
        limit: int = 5,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Return a small sample of IDs, text, and metadata from a collection.

        Embeddings are omitted by default because they are large and not useful for routine
        inspection. Set include_embeddings=True only when debugging vector dimensions.
        """
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        collection = self.get_collection(collection_name)
        return collection.get(limit=limit, include=include)

    def delete_collection(self, collection_name: str, missing_ok: bool = False) -> bool:
        """Delete one collection by name.

        Returns True when a collection was deleted. Returns False only when missing_ok=True
        and the target collection does not exist.
        """
        try:
            self.client.delete_collection(collection_name)
            return True
        except (NotFoundError, ValueError):
            if missing_ok:
                return False
            raise