from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import DEFAULT_TOP_K
from .retrievers import HybridRetriever


class ChromaKnowledgeBase:
    """Adapter around the persisted ChromaDB collection used by the RAG app."""

    def __init__(self, chroma_dir: Path, collection_name: str):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._index = None
        self._nodes: list[TextNode] | None = None
        self._bm25_cache: dict[int, BM25Retriever] = {}

    @property
    def collection(self):
        if self._collection is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )
        return self._index

    @property
    def nodes(self) -> list[TextNode]:
        if self._nodes is None:
            records = self.collection.get(include=["documents", "metadatas"])
            documents = records.get("documents", []) or []
            metadatas = records.get("metadatas", []) or []
            ids = records.get("ids", []) or []

            self._nodes = [
                TextNode(id_=node_id, text=document_text, metadata=metadata or {})
                for node_id, document_text, metadata in zip(ids, documents, metadatas)
                if document_text
            ]

            if not self._nodes:
                raise ValueError("No stored text nodes were found in the Chroma collection.")

        return self._nodes

    def bm25_retriever(self, top_k: int) -> BM25Retriever:
        if top_k not in self._bm25_cache:
            self._bm25_cache[top_k] = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=top_k,
            )
        return self._bm25_cache[top_k]

    def hybrid_retriever(
        self,
        final_top_k: int = DEFAULT_TOP_K,
        candidate_top_k: int | None = None,
        rrf_k: int = 60,
    ) -> HybridRetriever:
        candidate_count = candidate_top_k if candidate_top_k is not None else max(final_top_k * 2, 10)
        semantic_retriever = self.index.as_retriever(similarity_top_k=candidate_count)
        keyword_retriever = self.bm25_retriever(top_k=candidate_count)

        return HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )

    def count(self) -> int:
        return self.collection.count()
