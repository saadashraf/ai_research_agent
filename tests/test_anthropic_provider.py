
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import CompletionResponse, Message


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    with patch("src.providers.anthropic_provider.anthropic.Anthropic") as MockClient:
        p = AnthropicProvider(model="claude-test-model")
        p._client = MockClient.return_value
        yield p


def _sdk_response(text="Hello!", input_tokens=10, output_tokens=5,
                  model="claude-test-model", stop_reason="end_turn"):
    """Minimal fake SDK response object matching the fields we read."""
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.model = model
    resp.stop_reason = stop_reason
    return resp


@contextmanager
def _mock_stream(chunks):
    """Context manager that mimics the SDK's messages.stream() behaviour."""
    stream = MagicMock()
    stream.text_stream = iter(chunks)
    yield stream


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    def test_returns_completion_response(self, provider):
        provider._client.messages.create.return_value = _sdk_response("Hi!")
        result = provider.complete([Message(role="user", content="Hello")])
        assert isinstance(result, CompletionResponse)

    def test_reply_text_mapped_correctly(self, provider):
        provider._client.messages.create.return_value = _sdk_response(text="Pong")
        result = provider.complete([Message(role="user", content="Ping")])
        assert result.reply_text == "Pong"

    def test_token_counts_mapped(self, provider):
        provider._client.messages.create.return_value = _sdk_response(
            input_tokens=7, output_tokens=3
        )
        result = provider.complete([Message(role="user", content="x")])
        assert result.input_tokens == 7
        assert result.output_tokens == 3

    def test_model_and_stop_reason_mapped(self, provider):
        provider._client.messages.create.return_value = _sdk_response(
            model="claude-special", stop_reason="max_tokens"
        )
        result = provider.complete([Message(role="user", content="x")])
        assert result.model == "claude-special"
        assert result.stop_reason == "max_tokens"

    def test_messages_converted_to_dicts(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        msgs = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
        provider.complete(msgs)
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_system_prompt_included_when_provided(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")], system="Be brief.")
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be brief."

    def test_system_prompt_omitted_when_none(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")])
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_custom_temperature_forwarded(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")], temperature=0.1)
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    def test_temperature_zero_not_overridden_by_config(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")], temperature=0.0)
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

    def test_custom_max_tokens_forwarded(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")], max_tokens=128)
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128

    def test_model_name_forwarded(self, provider):
        provider._client.messages.create.return_value = _sdk_response()
        provider.complete([Message(role="user", content="x")])
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-test-model"


# ---------------------------------------------------------------------------
# stream_complete()
# ---------------------------------------------------------------------------

class TestStreamComplete:
    def test_yields_chunks_in_order(self, provider):
        chunks = ["Hello", ", ", "world", "!"]
        provider._client.messages.stream.return_value = _mock_stream(chunks)
        result = list(provider.stream_complete([Message(role="user", content="Hi")]))
        assert result == chunks

    def test_yields_nothing_for_empty_stream(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        result = list(provider.stream_complete([Message(role="user", content="Hi")]))
        assert result == []

    def test_returns_iterator(self, provider):
        provider._client.messages.stream.return_value = _mock_stream(["x"])
        result = provider.stream_complete([Message(role="user", content="Hi")])
        assert hasattr(result, "__iter__") and hasattr(result, "__next__")

    def test_messages_converted_to_dicts(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        msgs = [Message(role="user", content="Test")]
        list(provider.stream_complete(msgs))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "Test"}]

    def test_system_prompt_included_when_provided(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete(
            [Message(role="user", content="x")], system="You are a bot."
        ))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["system"] == "You are a bot."

    def test_system_prompt_omitted_when_none(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete([Message(role="user", content="x")]))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert "system" not in call_kwargs

    def test_custom_temperature_forwarded(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete(
            [Message(role="user", content="x")], temperature=0.2
        ))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2

    def test_temperature_zero_not_overridden_by_config(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete(
            [Message(role="user", content="x")], temperature=0.0
        ))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

    def test_custom_max_tokens_forwarded(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete(
            [Message(role="user", content="x")], max_tokens=64
        ))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["max_tokens"] == 64

    def test_model_name_forwarded(self, provider):
        provider._client.messages.stream.return_value = _mock_stream([])
        list(provider.stream_complete([Message(role="user", content="x")]))
        call_kwargs = provider._client.messages.stream.call_args.kwargs
        assert call_kwargs["model"] == "claude-test-model"

    def test_concatenated_chunks_form_full_reply(self, provider):
        chunks = ["The", " answer", " is", " 42", "."]
        provider._client.messages.stream.return_value = _mock_stream(chunks)
        result = "".join(
            provider.stream_complete([Message(role="user", content="Hi")])
        )
        assert result == "The answer is 42."


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_returns_integer(self, provider):
        provider._client.messages.count_tokens.return_value = MagicMock(input_tokens=5)
        result = provider.count_tokens("hello world")
        assert isinstance(result, int)

    def test_returns_correct_count(self, provider):
        provider._client.messages.count_tokens.return_value = MagicMock(input_tokens=42)
        result = provider.count_tokens("some text")
        assert result == 42

    def test_text_wrapped_as_user_message(self, provider):
        provider._client.messages.count_tokens.return_value = MagicMock(input_tokens=1)
        provider.count_tokens("hi")
        call_kwargs = provider._client.messages.count_tokens.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_model_name_forwarded(self, provider):
        provider._client.messages.count_tokens.return_value = MagicMock(input_tokens=1)
        provider.count_tokens("hi")
        call_kwargs = provider._client.messages.count_tokens.call_args.kwargs
        assert call_kwargs["model"] == "claude-test-model"


# ---------------------------------------------------------------------------
# model_name property
# ---------------------------------------------------------------------------

class TestModelName:
    def test_returns_model_passed_at_init(self):
        with patch("src.providers.anthropic_provider.anthropic.Anthropic"):
            provider = AnthropicProvider(model="claude-haiku-4-5")
        assert provider.model_name == "claude-haiku-4-5"
