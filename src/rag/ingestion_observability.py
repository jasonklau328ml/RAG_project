from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, TypeVar

import psutil
from llama_index.core.schema import Document, TextNode
from opentelemetry import trace


T = TypeVar("T")

DEFAULT_REQUIRED_METADATA_KEYS = (
    "source_folder",
    "article_date",
    "article_title",
    "file_name",
    "file_path",
)


def estimate_token_count(text: str) -> int:
    """Estimate token count for ingestion quality metrics.

    This deliberately avoids binding ingestion monitoring to a specific LLM tokenizer. The
    estimate is stable enough for trend monitoring, token-to-character ratio checks, and
    finding unusually noisy scraped articles.
    """
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def current_ram_usage_mb() -> float:
    """Return current process RAM usage in MiB for span attributes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class IngestionDocumentMetric:
    """Dataset-friendly per-document metrics captured during ingestion."""

    file_name: str
    source_folder: str
    article_date: str
    article_title: str
    char_count: int
    token_count: int
    token_to_char_ratio: float
    chunk_count: int = 0
    metadata_valid: bool = True
    missing_metadata_keys: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_folder": self.source_folder,
            "article_date": self.article_date,
            "article_title": self.article_title,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "token_to_char_ratio": self.token_to_char_ratio,
            "chunk_count": self.chunk_count,
            "metadata_valid": self.metadata_valid,
            "missing_metadata_keys": ", ".join(self.missing_metadata_keys),
        }


class IngestionPhoenixObserver:
    """Phoenix/OpenTelemetry observer for the RAG ingestion pipeline.

    Span metrics are best for runtime behavior: latency, RAM usage, queue depth, throughput,
    and batch state. Dataset records are best for document-level quality signals: token counts,
    chunk counts, metadata validation, and token-to-character ratios.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_model_name: str,
        phoenix_base_url: str = "http://localhost:6006",
        dataset_name: str = "rag-ingestion-chunk-metrics",
        enabled: bool = True,
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.phoenix_base_url = phoenix_base_url.rstrip("/")
        self.dataset_name = dataset_name
        self.enabled = enabled
        self.tracer = trace.get_tracer("rag.ingestion")
        self.document_metrics: dict[str, IngestionDocumentMetric] = {}
        self.chunk_metrics: list[dict[str, Any]] = []
        self.batch_metrics: list[dict[str, Any]] = []
        self.last_dataset_export_error: str | None = None
        self.last_dataset_export_action: str | None = None

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        """Create a custom ingestion span with common collection/model attributes."""
        if not self.enabled:
            yield None
            return

        span_attributes = {
            "rag.collection_name": self.collection_name,
            "rag.embedding_model": self.embedding_model_name,
            **(attributes or {}),
        }
        with self.tracer.start_as_current_span(name) as active_span:
            for key, value in span_attributes.items():
                active_span.set_attribute(key, value)
            active_span.set_attribute("process.ram_mb.start", current_ram_usage_mb())
            try:
                yield active_span
            finally:
                active_span.set_attribute("process.ram_mb.end", current_ram_usage_mb())

    def measure(self, name: str, func: Callable[[], T], attributes: dict[str, Any] | None = None) -> tuple[T, float]:
        """Run a callable inside a span and return its result plus elapsed seconds."""
        start_time = time.perf_counter()
        with self.span(name, attributes) as active_span:
            result = func()
            elapsed_seconds = time.perf_counter() - start_time
            if active_span is not None:
                active_span.set_attribute("latency.seconds", elapsed_seconds)
            return result, elapsed_seconds

    def build_document_metrics(
        self,
        documents: Sequence[Document],
        required_metadata_keys: Sequence[str] = DEFAULT_REQUIRED_METADATA_KEYS,
    ) -> list[IngestionDocumentMetric]:
        """Validate metadata and compute per-document token/character quality metrics."""
        metrics: list[IngestionDocumentMetric] = []
        for document in documents:
            metadata = document.metadata or {}
            missing_keys = [key for key in required_metadata_keys if not metadata.get(key)]
            char_count = len(document.text or "")
            token_count = estimate_token_count(document.text or "")
            token_to_char_ratio = token_count / char_count if char_count else 0.0
            metric = IngestionDocumentMetric(
                file_name=str(metadata.get("file_name", "unknown")),
                source_folder=str(metadata.get("source_folder", "unknown")),
                article_date=str(metadata.get("article_date", "unknown")),
                article_title=str(metadata.get("article_title", "unknown")),
                char_count=char_count,
                token_count=token_count,
                token_to_char_ratio=token_to_char_ratio,
                metadata_valid=not missing_keys,
                missing_metadata_keys=missing_keys,
            )
            self.document_metrics[metric.file_name] = metric
            metrics.append(metric)

        invalid_count = sum(not metric.metadata_valid for metric in metrics)
        total_tokens = sum(metric.token_count for metric in metrics)
        with self.span(
            "ingestion.metadata_validation",
            {
                "documents.count": len(metrics),
                "documents.invalid_metadata_count": invalid_count,
                "tokens.total_ingested_estimate": total_tokens,
            },
        ):
            pass

        return metrics

    def attach_chunk_counts(self, nodes: Sequence[TextNode]) -> None:
        """Attach chunk counts per source document to already-built document metrics."""
        chunk_counts: dict[str, int] = {}
        for node in nodes:
            file_name = str((node.metadata or {}).get("file_name", "unknown"))
            chunk_counts[file_name] = chunk_counts.get(file_name, 0) + 1

        for file_name, chunk_count in chunk_counts.items():
            if file_name in self.document_metrics:
                self.document_metrics[file_name].chunk_count = chunk_count

    def record_chunking(self, documents: Sequence[Document], nodes: Sequence[TextNode], latency_seconds: float) -> None:
        """Record aggregate chunking latency, total tokens, and chunk count metrics as a span."""
        self.attach_chunk_counts(nodes)
        self.chunk_metrics = []
        for node in nodes:
            metadata = node.metadata or {}
            chunk_text = node.get_content()
            char_count = len(chunk_text)
            token_count = estimate_token_count(chunk_text)
            token_to_char_ratio = token_count / char_count if char_count else 0.0
            file_name = str(metadata.get("file_name", "unknown"))
            chunk_count_in_document = self.document_metrics.get(file_name).chunk_count if file_name in self.document_metrics else 0
            required_missing = [key for key in DEFAULT_REQUIRED_METADATA_KEYS if not metadata.get(key)]

            self.chunk_metrics.append(
                {
                    "node_id": node.node_id,
                    "file_name": file_name,
                    "source_folder": str(metadata.get("source_folder", "unknown")),
                    "article_title": str(metadata.get("article_title", "unknown")),
                    "article_date": str(metadata.get("article_date", "unknown")),
                    "chunk_number": int(metadata.get("chunk_number", 0) or 0),
                    "chunk_size": int(metadata.get("chunk_size", 0) or 0),
                    "chunk_overlap": int(metadata.get("chunk_overlap", 0) or 0),
                    "chunk_text": chunk_text,
                    "chunk_char_count": char_count,
                    "chunk_token_count": token_count,
                    "token_to_char_ratio": token_to_char_ratio,
                    "metadata_valid": not required_missing,
                    "missing_metadata_keys": ", ".join(required_missing),
                    "chunk_count_in_document": chunk_count_in_document,
                }
            )

        total_tokens = sum(metric.token_count for metric in self.document_metrics.values())
        with self.span(
            "ingestion.chunking_summary",
            {
                "latency.seconds": latency_seconds,
                "documents.count": len(documents),
                "chunks.count": len(nodes),
                "chunks.per_document.average": len(nodes) / len(documents) if documents else 0.0,
                "tokens.total_ingested_estimate": total_tokens,
                "dataset.rows_prepared": len(self.chunk_metrics),
            },
        ):
            pass

    def record_vector_batch(
        self,
        *,
        batch_index: int,
        batch_size: int,
        remaining_items: int,
        embedding_latency_seconds: float,
        insertion_latency_seconds: float,
    ) -> None:
        """Record batch-level embedding/insertion latency, VPS, queue depth, and RAM."""
        total_latency = embedding_latency_seconds + insertion_latency_seconds
        vectors_per_second = batch_size / total_latency if total_latency else 0.0
        metric = {
            "batch_index": batch_index,
            "batch_size": batch_size,
            "queue_depth_remaining": remaining_items,
            "embedding_latency_seconds": embedding_latency_seconds,
            "vector_db_insertion_latency_seconds": insertion_latency_seconds,
            "vectors_per_second": vectors_per_second,
            "ram_mb": current_ram_usage_mb(),
        }
        self.batch_metrics.append(metric)

        with self.span(
            "ingestion.vector_batch",
            {
                "batch.index": batch_index,
                "batch.size": batch_size,
                "index.queue_depth_remaining": remaining_items,
                "embedding.latency.seconds": embedding_latency_seconds,
                "vectordb.insertion_latency.seconds": insertion_latency_seconds,
                "vectordb.vectors_per_second": vectors_per_second,
                "index.state": "batch_complete",
            },
        ):
            pass

    def export_document_metrics_dataset(self) -> Any | None:
        """Upload chunk-level ingestion metrics to Phoenix as a dataset when available."""
        self.last_dataset_export_error = None
        self.last_dataset_export_action = None

        if not self.enabled or not self.chunk_metrics:
            self.last_dataset_export_action = "skipped"
            self.last_dataset_export_error = "observer disabled or no chunk metrics to export"
            return None

        try:
            from phoenix.client import Client
        except (ImportError, ModuleNotFoundError) as error:
            self.last_dataset_export_action = "skipped"
            self.last_dataset_export_error = f"phoenix client import failed: {error}"
            return None

        examples = [
            {
                "input": {
                    "file_name": record["file_name"],
                    "source_folder": record["source_folder"],
                    "article_title": record["article_title"],
                    "chunk_number": record["chunk_number"],
                },
                "output": {
                    "chunk_text": record["chunk_text"],
                },
                "metadata": {
                    "article_date": record["article_date"],
                    "node_id": record["node_id"],
                    "chunk_size": record["chunk_size"],
                    "chunk_overlap": record["chunk_overlap"],
                    "chunk_char_count": record["chunk_char_count"],
                    "chunk_token_count": record["chunk_token_count"],
                    "token_to_char_ratio": record["token_to_char_ratio"],
                    "metadata_valid": record["metadata_valid"],
                    "missing_metadata_keys": record["missing_metadata_keys"],
                    "chunk_count_in_document": record["chunk_count_in_document"],
                },
            }
            for record in self.chunk_metrics
        ]
        client = Client(base_url=self.phoenix_base_url)
        try:
            result = client.datasets.create_dataset(
                name=self.dataset_name,
                examples=examples,
                input_keys=("file_name", "source_folder", "article_title", "chunk_number"),
                output_keys=("chunk_text",),
                metadata_keys=(
                    "article_date",
                    "node_id",
                    "chunk_size",
                    "chunk_overlap",
                    "chunk_char_count",
                    "chunk_token_count",
                    "token_to_char_ratio",
                    "metadata_valid",
                    "missing_metadata_keys",
                    "chunk_count_in_document",
                ),
                dataset_description="Chunk-level metrics captured during RAG vector ingestion.",
                timeout=10,
            )
            self.last_dataset_export_action = "created"
            return result
        except Exception as create_error:
            # On reruns the dataset may already exist, so fall back to appending examples.
            try:
                result = client.datasets.add_examples_to_dataset(
                    dataset=self.dataset_name,
                    examples=examples,
                    input_keys=("file_name", "source_folder", "article_title", "chunk_number"),
                    output_keys=("chunk_text",),
                    metadata_keys=(
                        "article_date",
                        "node_id",
                        "chunk_size",
                        "chunk_overlap",
                        "chunk_char_count",
                        "chunk_token_count",
                        "token_to_char_ratio",
                        "metadata_valid",
                        "missing_metadata_keys",
                        "chunk_count_in_document",
                    ),
                    timeout=10,
                )
                self.last_dataset_export_action = "appended"
                return result
            except Exception as append_error:
                # Dataset upload is useful but non-critical. Tracing should not fail ingestion.
                self.last_dataset_export_action = "failed"
                self.last_dataset_export_error = (
                    f"create failed: {create_error}; append failed: {append_error}"
                )
                return None

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary of metrics captured so far."""
        total_tokens = sum(metric.token_count for metric in self.document_metrics.values())
        invalid_metadata = sum(not metric.metadata_valid for metric in self.document_metrics.values())
        total_vectors = sum(metric["batch_size"] for metric in self.batch_metrics)
        total_batch_seconds = sum(
            metric["embedding_latency_seconds"] + metric["vector_db_insertion_latency_seconds"]
            for metric in self.batch_metrics
        )
        return {
            "documents": len(self.document_metrics),
            "chunks": len(self.chunk_metrics),
            "invalid_metadata_documents": invalid_metadata,
            "total_tokens_ingested_estimate": total_tokens,
            "vector_batches": len(self.batch_metrics),
            "vectors_written": total_vectors,
            "average_vectors_per_second": total_vectors / total_batch_seconds if total_batch_seconds else 0.0,
            "ram_mb_current": current_ram_usage_mb(),
        }