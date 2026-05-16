from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any


@dataclass
class Message:
    """A single turn in a conversation."""
    role: str      # "user" or "assistant"
    content: str

@dataclass
class ToolParameter:
    """Describes one parameter a tool accepts."""
    name: str
    type: str               # "string", "number", "boolean"
    description: str
    required: bool = True


@dataclass
class Tool:
    """
    A tool definition you hand to the model.
    The model reads name + description to decide WHEN to use it.
    It reads parameters to know WHAT arguments to pass.
    """
    name: str
    description: str        # ← this is the most important field
    parameters: list[ToolParameter]


@dataclass
class ToolCall:
    """What the model sends back when it wants to use a tool."""
    id: str                 # Anthropic assigns this — you echo it back
    name: str               # which tool it chose
    arguments: dict[str, Any]  # the args it wants to pass


@dataclass
class CompletionResponse:
    """Replace the existing CompletionResponse with this expanded version."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str        # "end_turn" or "tool_use"
    tool_calls: list[ToolCall] = None   # populated when stop_reason="tool_use"

    def wants_tool(self) -> bool:
        """Convenience method — did the model ask for a tool?"""
        return self.stop_reason == "tool_use" and bool(self.tool_calls)


class ModelProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResponse:
        """
        Send messages and return the model's reply.
        This is the core method.
        """
        ...

    @abstractmethod
    def stream_complete(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Send messages and yield text chunks as they arrive from the model.
        Each yielded value is a raw string delta (may be a single token or
        a few characters depending on the provider).
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        ...