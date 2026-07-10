# Phoenix Observability Dataflow (Notebook + Chainlit)

This document focuses on observability dataflow only: how trace data is created, enriched, exported, and visualized in Phoenix.

## 1) Phoenix Bootstrapping Dataflow

```mermaid
flowchart TD
    A["Entry point starts\nnotebook/inference.ipynb or app.py"] --> B["setup_phoenix_observability"]
    B --> C["_normalize_collector_endpoint"]
    C --> D["_call_register phoenix.otel.register"]
    D --> E["Create/OpenTelemetry tracer provider"]
    E --> F["LlamaIndexInstrumentor.instrument"]
    F --> G["LlamaIndex spans become exportable"]
    D --> H["Collector endpoint: /v1/traces"]
    H --> I["Phoenix collector receives spans"]
    I --> J["Phoenix UI query + timeline"]
```

Observability data produced at bootstrap:

1. Phoenix project name
2. Collector endpoint (normalized to OTLP traces route)
3. Tracer provider instance
4. Instrumentation state (`LlamaIndex` instrumented once)

## 2) Per-Message Trace Context Dataflow

```mermaid
flowchart LR
    U["User message"] --> A["trace_chat_session with session_id chat_id"]
    A --> B["using_session with chat_id"]
    A --> C["using_metadata interface notebook or chainlit"]
    A --> D["using_user optional"]
    B --> E["Context attached to child spans"]
    C --> E
    D --> E
```

What this context does:

1. Groups many spans under one saved chat id
2. Tags traces by interface (`notebook` vs `chainlit`)
3. Ensures retrieval and LLM spans inherit same session metadata

## 3) Chainlit Turn-Level Parent Span Dataflow

Chainlit adds one manual parent span per user turn (`trace_rag_chat_turn`), then runs `rag_app.chat(...)` inside it.

```mermaid
flowchart TD
    A["run_traced_chat"] --> B["trace_chat_session"]
    B --> C["trace_rag_chat_turn start span: rag.chat_turn"]
    C --> D["Set input attributes"]
    D --> D1["session.id = chat_id"]
    D --> D2["input.value = user message JSON"]
    D --> D3["input.mime_type = application/json"]
    D --> D4["rag config attrs\ncollection, embedding model, llm model, interface"]
    C --> E["rag_app.chat executes"]
    E --> F["LlamaIndex child spans\nretrieval + prompt + llm"]
    E --> G["response returned"]
    G --> H["Set output attributes in app.py"]
    H --> H1["output.value = assistant response JSON"]
    H --> H2["output.mime_type = application/json"]
    H --> H3["rag.assistant_response.length"]
    H --> I["Set status OK or ERROR"]
    I --> J["Span exported to Phoenix collector"]
```

Important difference:

1. Notebook: only chat-session context wrapper
2. Chainlit app: chat-session context + explicit `rag.chat_turn` parent span

## 4) End-to-End Phoenix Trace Export Path

```mermaid
sequenceDiagram
    participant App as Notebook or Chainlit
    participant Ctx as trace_chat_session / trace_rag_chat_turn
    participant LI as LlamaIndex Instrumentor
    participant OTel as OpenTelemetry SDK
    participant Col as Phoenix Collector
    participant UI as Phoenix UI

    App->>Ctx: start context/span for chat turn
    App->>LI: execute retrieval + llm calls
    LI->>OTel: emit child spans and events
    Ctx->>OTel: emit parent/context span attributes
    OTel->>Col: OTLP HTTP export to /v1/traces
    Col-->>UI: indexed traces available
    UI-->>App: visual inspection by project/session/span
```

## 5) Attribute-Level Data Contract (what Phoenix receives)

Core attributes set by your code:

1. `session.id` = saved chat id
2. `openinference.span.kind` = `CHAIN` (for parent turn span)
3. `input.value` = JSON with user message
4. `input.mime_type` = `application/json`
5. `output.value` = JSON with assistant response (Chainlit path)
6. `output.mime_type` = `application/json` (Chainlit path)
7. `rag.interface` = `notebook` or `chainlit`
8. `rag.chat_id`, `rag.collection_name`, `rag.embedding_model`, `rag.llm_model`
9. `rag.user_message.length`, `rag.assistant_response.length`
10. Status + exception info on error

## 6) Observability-Focused Function Map

```mermaid
graph TD
    A["src/rag/core/observability.py"] --> A1["setup_phoenix_observability"]
    A --> A2["trace_chat_session"]
    A --> A3["trace_rag_chat_turn"]

    B["src/rag/inference/inference_observability.py"] --> B1["thin wrapper for inference layer"]

    C["notebook/inference.ipynb"] --> C1["setup_phoenix_observability before create_rag_app"]
    C --> C2["notebook_chat uses trace_chat_session"]

    D["app.py"] --> D1["global phoenix setup at import time"]
    D --> D2["run_traced_chat uses trace_chat_session"]
    D --> D3["run_traced_chat uses trace_rag_chat_turn"]
    D --> D4["run_traced_chat writes output attributes"]
```

## 7) Quick Read: How Phoenix Data Passes Through Your Program

1. Startup config creates tracer provider and instruments LlamaIndex.
2. Each user message enters a session context (`chat_id`-scoped).
3. Chainlit also creates one explicit parent turn span and writes input/output JSON attributes.
4. Retrieval and LLM operations emit child spans automatically via instrumentation.
5. OpenTelemetry exporter sends spans to Phoenix collector endpoint.
6. Phoenix UI organizes traces by project and session id so you can inspect one conversation turn-by-turn.
