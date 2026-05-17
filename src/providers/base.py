from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union


MessageContent = Union[str, list[dict[str, Any]]]


@dataclass
class Message:
    """A single turn in a conversation."""
    role: str                  # "user" or "assistant"
    content: MessageContent    # str OR list of content blocks

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
    stop_reason: str                                  # "end_turn" or "tool_use"
    tool_calls: Optional[list[ToolCall]] = None      # populated when stop_reason="tool_use"

    def wants_tool(self) -> bool:
        """Convenience method — did the model ask for a tool?"""
        return self.stop_reason == "tool_use" and bool(self.tool_calls)

@dataclass
class AgentResult:
    """
    Everything produced by one complete agent run.
    Returned by the harness — richer than a plain string.
    """
    answer: str
    success: bool
    turns: int
    tool_calls_made: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    history: list["Message"]


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
        tools: Optional[list[Tool]] = None,
    ) -> CompletionResponse:
        """
        Send messages and return the model's reply.
        If `tools` is provided, the model may respond with tool calls
        (CompletionResponse.tool_calls will be populated).
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