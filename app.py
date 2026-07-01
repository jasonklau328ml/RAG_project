import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
    
import chainlit as cl
from llama_index.core.llms import ChatMessage

from src.rag.core import (
    DEFAULT_PHOENIX_ENDPOINT,
    DEFAULT_PHOENIX_PROJECT_NAME,
    setup_phoenix_observability,
    trace_rag_chat_turn,
    trace_chat_session,
)
from src.rag.retrieval import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_MEMORY_TOKEN_LIMIT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K as DEFAULT_TOP_K_VALUE,
    RagNewsChatbot,
    create_rag_app,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chromadb_store"
SESSION_DIR = PROJECT_ROOT / "session"

COLLECTION_NAME = DEFAULT_COLLECTION_NAME
EMBED_MODEL_NAME = DEFAULT_EMBED_MODEL_NAME
OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_TOP_K = DEFAULT_TOP_K_VALUE
MEMORY_TOKEN_LIMIT = DEFAULT_MEMORY_TOKEN_LIMIT
CHAT_SYSTEM_PROMPT = DEFAULT_CHAT_SYSTEM_PROMPT
ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", "1").lower() in {"1", "true", "yes"}
LAUNCH_PHOENIX_SERVER = os.getenv("LAUNCH_PHOENIX_SERVER", "0").lower() in {"1", "true", "yes"}
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", DEFAULT_PHOENIX_PROJECT_NAME)
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_PHOENIX_ENDPOINT)

# Configure Phoenix once when Chainlit imports this file. If the Phoenix packages are not
# installed yet, the app still runs and /observability shows the install command.
PHOENIX_OBSERVABILITY = setup_phoenix_observability(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    enabled=ENABLE_PHOENIX,
    launch_server=LAUNCH_PHOENIX_SERVER,
    raise_on_missing=False,
)

_rag_app: RagNewsChatbot | None = None


def get_rag_app() -> RagNewsChatbot:
    """Create the expensive RAG objects once, then reuse them for every Chainlit session."""
    global _rag_app
    if _rag_app is None:
        _rag_app = create_rag_app(
            chroma_dir=CHROMA_DIR,
            session_dir=SESSION_DIR,
            collection_name=COLLECTION_NAME,
            embed_model_name=EMBED_MODEL_NAME,
            ollama_model=OLLAMA_MODEL,
            final_top_k=DEFAULT_TOP_K,
            memory_token_limit=MEMORY_TOKEN_LIMIT,
            chat_system_prompt=CHAT_SYSTEM_PROMPT,
        )
    return _rag_app


def new_chat_id() -> str:
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def chainlit_thread_id() -> str | None:
    """Return Chainlit's persistent UI thread id when it is available."""
    try:
        session = cl.context.session
    except Exception:
        return None

    for attribute in ("thread_id", "id"):
        value = getattr(session, attribute, None)
        if value:
            return str(value)
    return None


def initial_chat_id() -> str:
    """Create a readable RAG chat id instead of reusing Chainlit's internal thread UUID."""
    return new_chat_id()


def current_chat_id() -> str:
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        raise RuntimeError("No active chat session. Start or load a chat first.")
    return chat_id


def active_chat_id() -> str | None:
    """Return the selected RAG chat id, if the user has chosen or started one."""
    return cl.user_session.get("chat_id")


def session_actions(session_store, limit: int = 8) -> list[cl.Action]:
    actions: list[cl.Action] = [cl.Action(name="new_chat", label="New chat", payload={})]

    for session_path in session_store.list()[:limit]:
        actions.append(
            cl.Action(
                name="load_chat",
                label=session_store.display_name_for(session_path),
                payload={"chat_id": session_path.stem},
            )
        )

    return actions


async def send_chat_history(chat_id: str, messages: list[ChatMessage]) -> None:
    """Replay every saved message as normal Chainlit chat bubbles."""
    if not messages:
        await cl.Message(content=f"Loaded **{chat_id}**. This saved chat is empty.").send()
        return

    await cl.Message(content=f"Loaded **{chat_id}**. Replaying {len(messages)} saved message(s).").send()
    for saved_message in messages:
        role = saved_message.role.value if hasattr(saved_message.role, "value") else str(saved_message.role)
        content = (saved_message.content or "").strip()
        if not content:
            continue

        # Use Chainlit's message type instead of one combined Markdown summary so the UI
        # shows the full notebook-created conversation in the same format as live chat.
        message_type = "user_message" if role == "user" else "assistant_message"
        author = "User" if role == "user" else "Assistant"
        await cl.Message(content=content, author=author, type=message_type).send()


def format_observability_status() -> str:
    status = PHOENIX_OBSERVABILITY
    lines = [
        f"Phoenix enabled: **{status.enabled}**",
        f"Project: `{status.project_name}`",
        f"Collector endpoint: `{status.endpoint}`",
        f"UI: {status.ui_url}",
    ]
    if status.message:
        lines.append(status.message)
    if not status.enabled:
        lines.append("Set `ENABLE_PHOENIX=1` before starting Chainlit to enable tracing.")
        lines.append(f"Install command: `{status.install_command}`")
    return "\n".join(lines)


def run_traced_chat(rag_app: RagNewsChatbot, chat_id: str, text: str):
    # The context manager adds the saved chat id to every LlamaIndex span created during
    # this turn, which makes Phoenix group multi-turn traces by conversation/session.
    with trace_chat_session(chat_id, metadata={"interface": "chainlit"}):
        # This parent span guarantees Phoenix shows one trace for each Chainlit message,
        # even if a library-level child instrumentor changes behavior across versions.
        with trace_rag_chat_turn(
            chat_id=chat_id,
            interface="chainlit",
            collection_name=COLLECTION_NAME,
            embed_model_name=EMBED_MODEL_NAME,
            llm_model=OLLAMA_MODEL,
            message=text,
        ) as turn_span:
            response = rag_app.chat(chat_id, text)
            if turn_span is not None:
                # Phoenix renders session turns from OpenInference input/output attributes.
                turn_span.set_attribute("output.mime_type", "application/json")
                turn_span.set_attribute(
                    "output.value",
                    json.dumps({"response": response.response or ""}, ensure_ascii=False),
                )
                turn_span.set_attribute("rag.assistant_response.length", len(response.response or ""))
            return response


async def ensure_active_chat(chat_id: str | None = None, overwrite: bool = False) -> str:
    rag_app = await asyncio.to_thread(get_rag_app)
    # Keep local RAG chat sessions under user-facing names. Chainlit thread ids are UI-internal
    # UUIDs and should not create extra JSON chat files or Phoenix session names.
    active_chat_id = chat_id or initial_chat_id()
    await asyncio.to_thread(rag_app.open_chat, active_chat_id, True, overwrite)

    cl.user_session.set("chat_id", active_chat_id)
    return active_chat_id


async def send_session_picker(content: str) -> None:
    rag_app = await asyncio.to_thread(get_rag_app)
    await cl.Message(
        content=content,
        actions=session_actions(rag_app.session_store),
    ).send()


@cl.on_chat_start
async def start_session():
    await send_session_picker(
        "Welcome. Use the buttons below to start fresh or load a saved RAG chat. "
        "No local chat file is created until you choose one or send a message."
    )


@cl.on_chat_resume
async def resume_session(thread: dict[str, Any]):
    # Chainlit resume ids are not the same as our saved RAG chat ids, so ask the user to
    # choose an existing saved chat instead of creating a UUID-named local session.
    await send_session_picker("Choose a saved RAG chat to continue:")


@cl.action_callback("new_chat")
async def new_chat_action(action: cl.Action):
    chat_id = await ensure_active_chat(overwrite=True)
    await cl.Message(content=f"Started a new chat: **{chat_id}**").send()
    await action.remove()


@cl.action_callback("load_chat")
async def load_chat_action(action: cl.Action):
    chat_id = action.payload.get("chat_id")
    if not chat_id:
        await cl.Message(content="I could not find that saved chat id.").send()
        return

    rag_app = await asyncio.to_thread(get_rag_app)
    await ensure_active_chat(chat_id=chat_id)
    messages = await asyncio.to_thread(rag_app.memory_messages, chat_id)

    await send_chat_history(chat_id, messages)
    await action.remove()


@cl.on_message
async def handle_message(message: cl.Message):
    rag_app = await asyncio.to_thread(get_rag_app)
    text = message.content.strip()

    if text == "/chats":
        await send_session_picker("Saved chats:")
        return

    if text == "/new":
        chat_id = await ensure_active_chat(overwrite=True)
        await cl.Message(content=f"Started a new chat: **{chat_id}**").send()
        return

    if text == "/observability":
        await cl.Message(content=format_observability_status()).send()
        return

    if text.startswith("/load "):
        chat_id = text.removeprefix("/load ").strip()
        await ensure_active_chat(chat_id=chat_id)
        messages = await asyncio.to_thread(rag_app.memory_messages, chat_id)
        await send_chat_history(chat_id, messages)
        return

    chat_id = active_chat_id()
    if not chat_id:
        chat_id = await ensure_active_chat()
    answer = cl.Message(content="")
    await answer.send()

    try:
        response = await asyncio.to_thread(run_traced_chat, rag_app, chat_id, text)
    except Exception as exc:
        answer.content = f"Sorry, the RAG app failed while answering: `{exc}`"
        await answer.update()
        return

    answer.content = response.response
    await answer.update()
