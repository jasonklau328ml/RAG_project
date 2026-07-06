from __future__ import annotations

from typing import Any

from ..core.observability import (
    DEFAULT_PHOENIX_ENDPOINT,
    DEFAULT_PHOENIX_PROJECT_NAME,
    PHOENIX_INSTALL_COMMAND,
    PhoenixObservabilityStatus,
    setup_phoenix_observability as _setup_core_phoenix_observability,
    trace_chat_session as _trace_core_chat_session,
    trace_rag_chat_turn as _trace_core_rag_chat_turn,
)


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
    """Configure Phoenix for retrieval-facing programs (notebook + Chainlit app).

    This wrapper keeps retrieval code decoupled from core internals while reusing the
    shared Phoenix bootstrap implementation.
    """
    return _setup_core_phoenix_observability(
        project_name=project_name,
        endpoint=endpoint,
        enabled=enabled,
        launch_server=launch_server,
        auto_instrument=auto_instrument,
        batch=batch,
        raise_on_missing=raise_on_missing,
    )


def trace_chat_session(
    session_id: str,
    *,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Attach chat-level session metadata for retrieval traces.

    Keeping this wrapper in retrieval makes notebook/app imports explicit and stable
    while delegating the actual OpenInference context propagation to core.
    """
    return _trace_core_chat_session(
        session_id,
        user_id=user_id,
        metadata=metadata,
    )


def trace_rag_chat_turn(
    *,
    chat_id: str,
    interface: str,
    collection_name: str,
    embed_model_name: str,
    llm_model: str,
    message: str,
):
    """Create one parent span for a retrieval/app chat turn.

    The parent span groups child LlamaIndex spans under one turn, making Phoenix
    timelines easier to inspect for each user message.
    """
    return _trace_core_rag_chat_turn(
        chat_id=chat_id,
        interface=interface,
        collection_name=collection_name,
        embed_model_name=embed_model_name,
        llm_model=llm_model,
        message=message,
    )


__all__ = [
    "DEFAULT_PHOENIX_ENDPOINT",
    "DEFAULT_PHOENIX_PROJECT_NAME",
    "PHOENIX_INSTALL_COMMAND",
    "PhoenixObservabilityStatus",
    "setup_phoenix_observability",
    "trace_chat_session",
    "trace_rag_chat_turn",
]
