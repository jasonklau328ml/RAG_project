from collections.abc import Generator, Sequence
from typing import Any

from huggingface_hub import InferenceClient
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from pydantic import Field, PrivateAttr


class HuggingFaceChatLLM(CustomLLM):
    """LlamaIndex LLM adapter for Hugging Face chat-completions models."""

    model: str = Field(description="Hugging Face model id used for chat completions.")
    api_key: str = Field(repr=False, description="Hugging Face API token.")
    provider: str = Field(default="auto", description="Hugging Face inference provider routing option.")
    temperature: float = Field(default=0.2, ge=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    context_window: int = Field(default=128_000, gt=0)

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = InferenceClient(api_key=self.api_key, provider=self.provider)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )

    def _message_to_dict(self, message: ChatMessage) -> dict[str, str]:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role not in {"system", "user", "assistant"}:
            role = "user"
        return {"role": role, "content": message.content or ""}

    def _extract_message_content(self, completion: Any) -> str:
        choice = completion.choices[0]
        message = choice.message
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return getattr(message, "content", "") or ""

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Keep LlamaIndex chat memory intact by sending the full engine-supplied message list.
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[self._message_to_dict(message) for message in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self._extract_message_content(completion)),
            raw=completion,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return CompletionResponse(text=self._extract_message_content(completion), raw=completion)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        # The app currently uses non-streaming calls; yield one response to satisfy CustomLLM.
        def response_generator() -> Generator[CompletionResponse, None, None]:
            yield self.complete(prompt, formatted=formatted, **kwargs)

        return response_generator()