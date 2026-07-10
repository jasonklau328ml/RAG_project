from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from .inference_config import DEFAULT_TOP_K
from .retrievers import HybridRetriever


class ChromaKnowledgeBase:
    """Adapter around the persisted ChromaDB collection used by the RAG app."""

    def __init__(self, chroma_dir: Path, collection_name: str):
        """Remember where the persisted vector database lives and delay heavy loading.

        Purpose:
        - Store connection details for the Chroma collection.
        - Defer actual database reads until a property such as ``collection`` or ``index`` is used.

        Output:
        - Initializes the object state on ``self``.
        """
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._index = None
        self._nodes: list[TextNode] | None = None
        self._bm25_cache: dict[int, BM25Retriever] = {}

    @property
    def collection(self):
        """Open the Chroma collection that stores embedded text chunks.

        Purpose:
        - Connect lazily to the on-disk vector database.
        - Reuse the same collection object after the first access.

        Output:
        - Returns the Chroma collection object.
        """
        if self._collection is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    @property
    def index(self) -> VectorStoreIndex:
        """Build the LlamaIndex vector index wrapper over the Chroma collection.

        Purpose:
        - Give LlamaIndex a standard interface for semantic retrieval over stored embeddings.

        Output:
        - Returns a ``VectorStoreIndex`` instance.
        """
        if self._index is None:
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )
        return self._index

    @property
    def nodes(self) -> list[TextNode]:
        """Load stored documents from Chroma and rebuild them as LlamaIndex text nodes.

        Purpose:
        - BM25 keyword retrieval needs raw text nodes, not only embeddings.
        - Convert Chroma records into the ``TextNode`` objects expected by LlamaIndex retrievers.

        Output:
        - Returns a list of ``TextNode`` objects.
        - Raises ``ValueError`` if the collection does not contain any stored text.
        """
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
        """Create or reuse a BM25 keyword retriever for a given result count.

        Purpose:
        - Support exact or near-exact word matching alongside embedding search.
        - Cache retrievers so repeated calls with the same ``top_k`` are cheap.

        Output:
        - Returns a ``BM25Retriever`` instance.
        """
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
        """Combine semantic search and keyword search into one hybrid retriever.

        Purpose:
        - Semantic search is good for meaning similarity.
        - BM25 is good for exact important words and names.
        - Combining both usually gives more robust RAG retrieval than using only one method.

        Output:
        - Returns a ``HybridRetriever`` instance.
        """
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
        """Count how many vector records are stored in the Chroma collection.

        Output:
        - Returns the collection size as an integer.
        """
        return self.collection.count()
