from __future__ import annotations

import os
from urllib.parse import urlparse
from urllib.request import urlopen
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator


PHOENIX_INSTALL_COMMAND = (
    "pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-llama-index"
)
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006"
DEFAULT_PHOENIX_PROJECT_NAME = "rag-news-chatbot"

_PHOENIX_SESSION: Any | None = None
_TRACER_PROVIDER: Any | None = None
_LLAMA_INDEX_INSTRUMENTED = False


@dataclass(frozen=True)
class PhoenixObservabilityStatus:
    """Small status object that notebooks and app.py can display without importing Phoenix directly."""

    enabled: bool
    project_name: str
    endpoint: str
    ui_url: str
    install_command: str = PHOENIX_INSTALL_COMMAND
    message: str = ""


def _dependency_error(error: ImportError | ModuleNotFoundError) -> RuntimeError:
    dependency_error = RuntimeError(
        "Phoenix observability dependencies are not installed. "
        f"Install them in the active Python environment with: {PHOENIX_INSTALL_COMMAND}"
    )
    dependency_error.__cause__ = error
    return dependency_error


def _call_register(*, project_name: str, endpoint: str, auto_instrument: bool, batch: bool):
    try:
        from phoenix.otel import register
    except (ImportError, ModuleNotFoundError) as error:
        raise _dependency_error(error)

    try:
        return register(
            project_name=project_name,
            endpoint=endpoint,
            auto_instrument=auto_instrument,
            batch=batch,
        )
    except TypeError:
        # Older phoenix.otel versions expose fewer keyword arguments. Keep a narrow fallback so
        # tracing still works if the environment is slightly behind the current documentation.
        return register(project_name=project_name, endpoint=endpoint)


def _normalize_collector_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme in {"http", "https"} and parsed.path in {"", "/"}:
        # OTLP/HTTP trace exporter expects the traces route; local Phoenix UI URL alone causes 405.
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def start_phoenix_server(host: str = "127.0.0.1", port: int = 6006) -> str:
    """Start the local Phoenix UI server and return the browser URL."""
    global _PHOENIX_SESSION

    try:
        import phoenix as px
    except (ImportError, ModuleNotFoundError) as error:
        raise _dependency_error(error)

    if _PHOENIX_SESSION is None:
        # New Phoenix versions prefer env vars over host/port function args.
        os.environ.setdefault("PHOENIX_HOST", host)
        os.environ.setdefault("PHOENIX_PORT", str(port))
        try:
            # use_temp_dir=False avoids fragile temp-file behavior on Windows.
            _PHOENIX_SESSION = px.launch_app(use_temp_dir=False)
        except TypeError:
            _PHOENIX_SESSION = px.launch_app(host=host, port=port, use_temp_dir=False)

    return getattr(_PHOENIX_SESSION, "url", None) or f"http://localhost:{port}/"


def _is_url_reachable(url: str, timeout_seconds: float = 1.5) -> bool:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return response.status < 500
    except Exception:
        return False


def _local_url_fallbacks(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.hostname
    if host == "localhost":
        return [url, url.replace("localhost", "127.0.0.1")]
    if host == "127.0.0.1":
        return [url, url.replace("127.0.0.1", "localhost")]
    return [url]


def setup_phoenix_observability(
    *,
    project_name: str = DEFAULT_PHOENIX_PROJECT_NAME,
    endpoint: str = DEFAULT_PHOENIX_ENDPOINT,
    enabled: bool = True,
    launch_server: bool = False,
    auto_instrument: bool = False,
    batch: bool = False,
    raise_on_missing: bool = True,
) -> PhoenixObservabilityStatus:
    """Configure Phoenix/OpenInference tracing for LlamaIndex.

    Call this before creating query engines or chat engines. Phoenix receives OpenTelemetry
    spans from LlamaIndex, so your retrieval calls, LLM calls, prompts, latency, and errors
    appear in the Phoenix UI while the RAG app runs normally.
    """
    global _TRACER_PROVIDER, _LLAMA_INDEX_INSTRUMENTED

    ui_url = endpoint.rstrip("/")
    collector_endpoint = _normalize_collector_endpoint(endpoint)
    if not enabled:
        return PhoenixObservabilityStatus(
            False,
            project_name,
            collector_endpoint,
            ui_url,
            message="Phoenix tracing is disabled.",
        )

    try:
        if launch_server:
            ui_url = start_phoenix_server()

        if _TRACER_PROVIDER is None:
            _TRACER_PROVIDER = _call_register(
                project_name=project_name,
                endpoint=collector_endpoint,
                auto_instrument=auto_instrument,
                batch=batch,
            )

        if not _LLAMA_INDEX_INSTRUMENTED:
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

            LlamaIndexInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
            _LLAMA_INDEX_INSTRUMENTED = True

    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        if raise_on_missing:
            raise
        return PhoenixObservabilityStatus(
            False,
            project_name,
            collector_endpoint,
            ui_url,
            message=str(error),
        )

    status = PhoenixObservabilityStatus(
        True,
        project_name,
        collector_endpoint,
        ui_url,
        message="Phoenix tracing is enabled. Run RAG queries, then open the Phoenix UI to inspect traces.",
    )

    if launch_server:
        for candidate_url in _local_url_fallbacks(status.ui_url):
            if _is_url_reachable(candidate_url):
                if candidate_url != status.ui_url:
                    status = PhoenixObservabilityStatus(
                        status.enabled,
                        status.project_name,
                        status.endpoint,
                        candidate_url,
                        status.install_command,
                        status.message,
                    )
                break
        else:
            status = PhoenixObservabilityStatus(
                status.enabled,
                status.project_name,
                status.endpoint,
                status.ui_url,
                status.install_command,
                "Phoenix tracing is enabled, but the local UI endpoint is not reachable yet. "
                "Wait 5-10 seconds and retry with localhost/127.0.0.1.",
            )

    return status


@contextmanager
def trace_chat_session(
    session_id: str,
    *,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Attach Phoenix session/user metadata to spans created inside this block."""
    try:
        from phoenix.otel import using_metadata, using_session, using_user
    except (ImportError, ModuleNotFoundError):
        yield
        return

    with ExitStack() as stack:
        # Session id lets Phoenix group multiple RAG turns from the same saved chat.
        stack.enter_context(using_session(session_id))
        stack.enter_context(using_user(user_id) if user_id else nullcontext())
        stack.enter_context(using_metadata(metadata) if metadata else nullcontext())
        yield