import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
    
import chainlit as cl
from llama_index.core.llms import ChatMessage

from src.rag import (
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
NEWS_DIR = PROJECT_ROOT / "data" / "hk_free_press_news"
CHROMA_DIR = PROJECT_ROOT / "chromadb_store"
SESSION_DIR = PROJECT_ROOT / "session"

COLLECTION_NAME = DEFAULT_COLLECTION_NAME
EMBED_MODEL_NAME = DEFAULT_EMBED_MODEL_NAME
OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_TOP_K = DEFAULT_TOP_K_VALUE
MEMORY_TOKEN_LIMIT = DEFAULT_MEMORY_TOKEN_LIMIT
CHAT_SYSTEM_PROMPT = DEFAULT_CHAT_SYSTEM_PROMPT

_rag_app: RagNewsChatbot | None = None


def get_rag_app() -> RagNewsChatbot:
    """Create the expensive RAG objects once, then reuse them for every Chainlit session."""
    global _rag_app
    if _rag_app is None:
        _rag_app = create_rag_app(
            chroma_dir=CHROMA_DIR,
            session_dir=SESSION_DIR,
            news_dir=NEWS_DIR,
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


def current_chat_id() -> str:
    chat_id = cl.user_session.get("chat_id")
    if not chat_id:
        raise RuntimeError("No active chat session. Start or load a chat first.")
    return chat_id


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


def format_recent_history(messages: list[ChatMessage], limit: int = 8) -> str:
    if not messages:
        return "This saved chat is empty. Send a message to begin."

    lines = ["Loaded the saved chat. Recent messages:"]
    for message in messages[-limit:]:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        content = (message.content or "").strip()
        if content:
            lines.append(f"\n**{role.title()}**: {content}")
    return "".join(lines)


async def ensure_active_chat(chat_id: str | None = None, overwrite: bool = False) -> str:
    rag_app = await asyncio.to_thread(get_rag_app)
    active_chat_id = chat_id or chainlit_thread_id() or new_chat_id()
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
    chat_id = await ensure_active_chat()
    await send_session_picker(
        "Welcome. A new RAG chat session is ready. Use the buttons below to start fresh or load a saved chat. "
        f"Current session: **{chat_id}**"
    )


@cl.on_chat_resume
async def resume_session(thread: dict[str, Any]):
    chat_id = str(thread.get("id") or chainlit_thread_id() or new_chat_id())
    await ensure_active_chat(chat_id=chat_id)


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

    await cl.Message(content=format_recent_history(messages)).send()
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

    if text.startswith("/load "):
        chat_id = text.removeprefix("/load ").strip()
        await ensure_active_chat(chat_id=chat_id)
        messages = await asyncio.to_thread(rag_app.memory_messages, chat_id)
        await cl.Message(content=format_recent_history(messages)).send()
        return

    chat_id = current_chat_id()
    answer = cl.Message(content="")
    await answer.send()

    try:
        response = await asyncio.to_thread(rag_app.chat, chat_id, text)
    except Exception as exc:
        answer.content = f"Sorry, the RAG app failed while answering: `{exc}`"
        await answer.update()
        return

    answer.content = response.response
    await answer.update()
