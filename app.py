import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import chainlit as cl
from llama_index.core.llms import ChatMessage

from src.rag.inference import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_MEMORY_TOKEN_LIMIT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K as DEFAULT_TOP_K_VALUE,
    RagNewsChatbot,
    create_rag_app,
)
from src.rag.inference.inference_observability import (
    DEFAULT_PHOENIX_ENDPOINT,
    DEFAULT_PHOENIX_PROJECT_NAME,
    setup_phoenix_observability,
    trace_rag_chat_turn,
    trace_chat_session,
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
    """Build the shared RAG application object the first time it is needed.

    Purpose:
    - Create the retrieval engine, chat engine support objects, and session storage only once.
    - Reuse the same object for every browser chat so the app does not reconnect to ChromaDB
      and rebuild retrievers on every message.

    Output:
    - Returns one ready-to-use ``RagNewsChatbot`` instance.
    - The same instance is returned on later calls during the current Python process.
    """
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
    """Create a readable id for a brand-new saved chat session.

    Purpose:
    - Give each new conversation a human-readable name instead of an opaque UUID.
    - Keep the same id format for both the saved JSON file name and Phoenix session label.

    Output:
    - Returns a string such as ``chat_20260710_154500``.
    """
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def chainlit_thread_id() -> str | None:
    """Read Chainlit's own internal thread id, if Chainlit has created one.

    Purpose:
    - Expose the UI-level conversation id that Chainlit manages internally.
    - This is mostly useful for debugging because this app intentionally uses its own
      saved ``chat_id`` values instead of reusing Chainlit's internal ids.

    Output:
    - Returns Chainlit's thread id as a string when available.
    - Returns ``None`` if the current request is outside an active Chainlit session.
    """
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
    """Pick the first saved-chat id for a newly started conversation.

    Purpose:
    - Keep the app's saved-chat naming policy in one place.
    - Avoid leaking Chainlit's internal UUIDs into local JSON files and Phoenix traces.

    Output:
    - Returns the next readable chat id produced by ``new_chat_id()``.
    """
    return new_chat_id()


def current_chat_id() -> str:
    """Return the currently selected saved chat id and fail loudly if none is active.

    Purpose:
    - Let other functions ask for the active chat when they require one to exist.
    - Protect the app from sending messages before the user has opened or created a chat.

    Output:
    - Returns the active chat id string.
    - Raises ``RuntimeError`` if no chat has been selected in the current Chainlit session.
    """
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        raise RuntimeError("No active chat session. Start or load a chat first.")
    return chat_id


def active_chat_id() -> str | None:
    """Return the selected saved chat id without raising an error.

    Purpose:
    - Provide a safe way to check whether the user has already chosen or started a chat.
    - This is the soft version of ``current_chat_id()``.

    Output:
    - Returns the chat id string when one is active.
    - Returns ``None`` when the user has not opened a chat yet.
    """
    return cl.user_session.get("chat_id")


def session_actions(session_store, limit: int = 8) -> list[cl.Action]:
    """Build the clickable Chainlit buttons used to create or load chats.

    Purpose:
    - Convert saved chat files into UI actions that Chainlit can render as buttons.
    - Always include one ``New chat`` action plus a short list of recently modified chats.

    Output:
    - Returns a list of ``cl.Action`` objects.
    - Each action contains the label shown in the UI and the payload needed by the callback.
    """
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
    """Replay previously saved messages into the Chainlit chat window.

    Purpose:
    - Turn stored LlamaIndex ``ChatMessage`` objects back into normal Chainlit bubbles.
    - Make a loaded conversation look the same as a live conversation in the browser.

    Output:
    - Sends zero or more Chainlit messages to the browser.
    - Returns ``None`` because the visible result is the UI update, not a Python value.
    """
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
    """Create a human-readable Phoenix status report for the ``/observability`` command.

    Purpose:
    - Summarize whether tracing is enabled and where spans are being sent.
    - Give the user a quick troubleshooting message inside Chainlit without opening code.

    Output:
    - Returns one formatted multi-line string for display in the chat UI.
    """
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
    """Run one RAG chat turn while attaching Phoenix tracing metadata.

    Purpose:
    - Wrap the real ``rag_app.chat(...)`` call in Phoenix/OpenTelemetry context.
    - Make Phoenix show a parent span for the user turn and child spans for retrieval
      and LLM work done by LlamaIndex.
    - Attach extra output fields so Phoenix shows the final assistant text clearly.

    Output:
    - Returns the LlamaIndex chat response object produced by ``rag_app.chat(...)``.
    - If Phoenix is installed, it also emits spans as a side effect.
    """
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
    """Make sure the current browser session has an opened RAG chat session.

    Purpose:
    - Create or load a saved chat in the backend before the user sends messages.
    - Store the chosen chat id in Chainlit's per-user session state for later requests.

    Output:
    - Returns the active chat id string that the current browser tab should use.
    - As a side effect, it opens the chat in ``RagNewsChatbot`` and stores the id in
      ``cl.user_session``.
    """
    rag_app = await asyncio.to_thread(get_rag_app)
    # Keep local RAG chat sessions under user-facing names. Chainlit thread ids are UI-internal
    # UUIDs and should not create extra JSON chat files or Phoenix session names.
    active_chat_id = chat_id or initial_chat_id()
    await asyncio.to_thread(rag_app.open_chat, active_chat_id, True, overwrite)

    cl.user_session.set("chat_id", active_chat_id)
    return active_chat_id


async def send_session_picker(content: str) -> None:
    """Send a message plus action buttons that let the user choose a chat session.

    Purpose:
    - Reuse the same UI pattern for welcome, resume, and saved-chat listing commands.
    - Keep the logic for building the buttons separate from the event handlers.

    Output:
    - Sends one Chainlit message containing action buttons.
    - Returns ``None`` because the meaningful result is the UI update.
    """
    rag_app = await asyncio.to_thread(get_rag_app)
    await cl.Message(
        content=content,
        actions=session_actions(rag_app.session_store),
    ).send()


@cl.on_chat_start
async def start_session():
    """Handle the moment a user opens a fresh Chainlit chat page.

    Purpose:
    - Show the first menu instead of silently creating files or sessions too early.
    - Teach the user that they can either start a new chat or load an existing one.

    Output:
    - Sends the welcome message and session-picker buttons to the UI.
    """
    await send_session_picker(
        "Welcome. Use the buttons below to start fresh or load a saved RAG chat. "
        "No local chat file is created until you choose one or send a message."
    )


@cl.on_chat_resume
async def resume_session(thread: dict[str, Any]):
    """Handle Chainlit's browser-level resume event.

    Purpose:
    - When Chainlit reconnects a browser thread, remind the user to pick one of this app's
      saved RAG chats rather than trusting Chainlit's internal thread id.
    - The ``thread`` argument is provided by Chainlit, but this app does not use it for
      backend persistence.

    Output:
    - Sends the saved-chat picker to the UI.
    """
    # Chainlit resume ids are not the same as our saved RAG chat ids, so ask the user to
    # choose an existing saved chat instead of creating a UUID-named local session.
    await send_session_picker("Choose a saved RAG chat to continue:")


@cl.action_callback("new_chat")
async def new_chat_action(action: cl.Action):
    """Handle clicks on the ``New chat`` button.

    Purpose:
    - Create a brand-new saved chat session and make it the active session for the browser.
    - Confirm the chosen chat id back to the user.

    Output:
    - Opens a new chat session, sends one confirmation message, and removes the clicked
      action button from the UI.
    """
    chat_id = await ensure_active_chat(overwrite=True)
    await cl.Message(content=f"Started a new chat: **{chat_id}**").send()
    await action.remove()


@cl.action_callback("load_chat")
async def load_chat_action(action: cl.Action):
    """Handle clicks on a saved-chat button.

    Purpose:
    - Activate an existing saved chat file and replay its messages into the UI.
    - Bridge the gap between persistent JSON storage and the live Chainlit browser session.

    Output:
    - Loads the requested chat, re-sends its history to the UI, and removes the action button.
    - Returns early with an error message if the payload does not contain a chat id.
    """
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
    """Main Chainlit message handler for commands and normal user questions.

    Purpose:
    - Route slash commands such as ``/chats`` and ``/observability``.
    - Ensure a saved chat session exists before normal AI questions are processed.
    - Call the traced RAG pipeline and stream the final answer back into the UI.

    Output:
    - For commands, sends the corresponding UI response and returns ``None``.
    - For normal questions, updates the placeholder assistant message with the final answer.
    - On failure, updates the placeholder message with an error string instead of crashing.
    """
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
