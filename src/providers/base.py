from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    """A single turn in a conversation."""
    role: str      # "user" or "assistant"
    content: str


@dataclass
class CompletionResponse:
    """Everything the provider returns from one API call."""
    reply_text: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str


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
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        ...