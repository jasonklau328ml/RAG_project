from collections import defaultdict

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from .config import DEFAULT_TOP_K


def reciprocal_rank_fusion(
    semantic_results: list[NodeWithScore],
    keyword_results: list[NodeWithScore],
    top_k: int,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    """Fuse semantic and keyword rankings without manually calibrating their scores."""
    fused_scores: dict[str, float] = defaultdict(float)
    node_lookup: dict[str, NodeWithScore] = {}

    for results in (semantic_results, keyword_results):
        for rank, result in enumerate(results, start=1):
            node_id = result.node.node_id
            fused_scores[node_id] += 1.0 / (rrf_k + rank)
            node_lookup.setdefault(node_id, result)

    ranked_nodes = sorted(
        node_lookup.values(),
        key=lambda result: fused_scores[result.node.node_id],
        reverse=True,
    )

    return [
        NodeWithScore(node=result.node, score=fused_scores[result.node.node_id])
        for result in ranked_nodes[:top_k]
    ]


class HybridRetriever(BaseRetriever):
    """LlamaIndex retriever that combines Chroma semantic search and BM25 keyword search."""

    def __init__(
        self,
        semantic_retriever,
        keyword_retriever: BM25Retriever,
        final_top_k: int = DEFAULT_TOP_K,
        rrf_k: int = 60,
    ):
        self._semantic_retriever = semantic_retriever
        self._keyword_retriever = keyword_retriever
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        semantic_results = self._semantic_retriever.retrieve(query_bundle)
        keyword_results = self._keyword_retriever.retrieve(query_bundle)

        return reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=self._final_top_k,
            rrf_k=self._rrf_k,
        )
