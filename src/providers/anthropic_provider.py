from typing import Iterator, Optional
import anthropic

import config
from src.providers.base import CompletionResponse, Message, ModelProvider


class AnthropicProvider(ModelProvider):
    """
    Concrete implementation of ModelProvider for Anthropic's API.
    """

    def __init__(self, model: str = config.MODEL):
        self._model = model
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResponse:
        # Convert our generic Message dataclass into the
        # Anthropic SDK's expected dict format
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Build kwargs — only pass system if provided
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            messages=anthropic_messages,
        )
        if system:
            kwargs["system"] = system

        # The actual API call — all SDK-specific code is here
        response = self._client.messages.create(**kwargs)

        # Translate the Anthropic response into our generic format
        return CompletionResponse(
            reply_text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )

    def stream_complete(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            messages=anthropic_messages,
        )
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        # Anthropic SDK has a built-in token counter
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens