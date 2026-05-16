from typing import Iterator, Optional
import anthropic

import config
from src.providers.base import CompletionResponse, Message, ModelProvider, Tool, ToolCall


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
        tools: Optional[list[Tool]] = None,
    ) -> CompletionResponse:

        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens or config.MAX_TOKENS,
            temperature=temperature or config.TEMPERATURE,
            messages=anthropic_messages,
        )

        if system:
            kwargs["system"] = system

        # Translate our Tool dataclasses into Anthropic's expected format
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.type,
                                "description": p.description,
                            }
                            for p in t.parameters
                        },
                        "required": [
                            p.name for p in t.parameters if p.required
                        ],
                    },
                }
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        # Extract any tool calls from the response
        tool_calls = []
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return CompletionResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
            tool_calls=tool_calls if tool_calls else None,
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