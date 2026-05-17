"""
Tests for the agent harness (src/agent/harness.py) and the AgentResult
dataclass added in phase/3/agent-loop.

The provider is fully mocked — no API calls, no key required.
The harness now accepts an injected `provider=` arg, which is what the
tests use throughout. `get_provider()` is patched only in the few tests
that verify the fallback path.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent import run_agent
from src.providers.base import (
    AgentResult,
    CompletionResponse,
    Message,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reply(text="done", stop_reason="end_turn", tool_calls=None,
           input_tokens=3, output_tokens=4):
    return CompletionResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="claude-test",
        stop_reason=stop_reason,
        tool_calls=tool_calls,
    )


def _fake_provider(responses):
    """A MagicMock provider whose .complete() yields the given responses in order."""
    provider = MagicMock()
    provider.complete.side_effect = list(responses)
    return provider


def _calc_call(id_="toolu_1", a=2, b=3):
    return ToolCall(id=id_, name="calculator",
                    arguments={"operation": "add", "a": a, "b": b})


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_fields_round_trip(self):
        history = [Message(role="user", content="hi")]
        result = AgentResult(
            answer="hi",
            success=True,
            turns=2,
            tool_calls_made=1,
            input_tokens=10,
            output_tokens=20,
            stop_reason="end_turn",
            history=history,
        )
        assert result.answer == "hi"
        assert result.success is True
        assert result.turns == 2
        assert result.tool_calls_made == 1
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.stop_reason == "end_turn"
        assert result.history == history

    def test_history_is_list_of_messages(self):
        """run_agent populates history with Message instances."""
        result = run_agent("hi",
                           provider=_fake_provider([_reply(text="hello")]))
        assert isinstance(result.history, list)
        assert all(isinstance(m, Message) for m in result.history)


# ---------------------------------------------------------------------------
# Provider injection
# ---------------------------------------------------------------------------

class TestProviderInjection:
    def test_injected_provider_is_used(self):
        provider = _fake_provider([_reply(text="hi")])
        result = run_agent("q", provider=provider)
        assert result.answer == "hi"
        provider.complete.assert_called_once()

    def test_get_provider_called_when_none_injected(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_fake_provider([_reply(text="hi")])) as factory:
            run_agent("q")
        factory.assert_called_once()

    def test_get_provider_not_called_when_provider_injected(self):
        with patch("src.agent.harness.get_provider") as factory:
            run_agent("q", provider=_fake_provider([_reply(text="hi")]))
        factory.assert_not_called()


# ---------------------------------------------------------------------------
# run_agent — happy paths (no tools, finish on turn 1)
# ---------------------------------------------------------------------------

class TestRunAgentNoTools:
    def test_returns_agent_result(self):
        result = run_agent("Capital of France?",
                           provider=_fake_provider([_reply(text="Paris")]))
        assert isinstance(result, AgentResult)

    def test_answer_taken_from_response_text(self):
        result = run_agent("Capital of France?",
                           provider=_fake_provider([_reply(text="Paris")]))
        assert result.answer == "Paris"

    def test_marks_success(self):
        result = run_agent("q", provider=_fake_provider([_reply()]))
        assert result.success is True

    def test_single_turn(self):
        result = run_agent("q", provider=_fake_provider([_reply()]))
        assert result.turns == 1

    def test_no_tool_calls_made(self):
        result = run_agent("q", provider=_fake_provider([_reply()]))
        assert result.tool_calls_made == 0

    def test_stop_reason_propagated(self):
        result = run_agent("q", provider=_fake_provider(
            [_reply(stop_reason="end_turn")]))
        assert result.stop_reason == "end_turn"

    def test_history_contains_user_message(self):
        result = run_agent("hello there",
                           provider=_fake_provider([_reply()]))
        assert result.history[0] == Message(role="user", content="hello there")


# ---------------------------------------------------------------------------
# run_agent — tool-calling path
# ---------------------------------------------------------------------------

class TestRunAgentWithTools:
    """Model wants a tool on turn 1, then gives final answer on turn 2."""

    def _scenario(self):
        return [
            _reply(text="Let me compute",
                   stop_reason="tool_use",
                   tool_calls=[_calc_call()]),
            _reply(text="The answer is 5"),
        ]

    def test_two_turns(self):
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?", provider=_fake_provider(self._scenario()))
        assert result.turns == 2

    def test_tool_calls_counted(self):
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?", provider=_fake_provider(self._scenario()))
        assert result.tool_calls_made == 1

    def test_final_answer_from_last_turn(self):
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?", provider=_fake_provider(self._scenario()))
        assert result.answer == "The answer is 5"

    def test_tokens_accumulated_across_turns(self):
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?", provider=_fake_provider(self._scenario()))
        # 3 + 3 input, 4 + 4 output across the two turns
        assert result.input_tokens == 6
        assert result.output_tokens == 8

    def test_tools_kwarg_forwarded_to_provider(self):
        provider = _fake_provider(self._scenario())
        with patch("src.agent.harness.execute_tool", return_value="5"):
            run_agent("q", tools=["fake-tool-obj"], provider=provider)
        for call in provider.complete.call_args_list:
            assert call.kwargs["tools"] == ["fake-tool-obj"]

    def test_system_kwarg_forwarded_to_provider(self):
        provider = _fake_provider([_reply()])
        run_agent("q", system="You are a bot.", provider=provider)
        assert provider.complete.call_args.kwargs["system"] == "You are a bot."

    def test_execute_tool_receives_the_tool_call(self):
        with patch("src.agent.harness.execute_tool",
                   return_value="5") as mock_exec:
            run_agent("q", provider=_fake_provider(self._scenario()))
        passed = mock_exec.call_args.args[0]
        assert isinstance(passed, ToolCall)
        assert passed.name == "calculator"

    def test_history_includes_assistant_tool_use_and_tool_result(self):
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("q",
                               provider=_fake_provider(self._scenario()))
        # history: user query, assistant(tool_use), user(tool_result)
        assert result.history[0].role == "user"
        assert result.history[1].role == "assistant"
        types = [b["type"] for b in result.history[1].content]
        assert "tool_use" in types
        assert result.history[2].role == "user"
        assert result.history[2].content[0]["type"] == "tool_result"
        assert result.history[2].content[0]["tool_use_id"] == "toolu_1"
        assert result.history[2].content[0]["content"] == "5"

    def test_empty_reply_text_skips_text_block_in_history(self):
        responses = [
            _reply(text="", stop_reason="tool_use", tool_calls=[_calc_call()]),
            _reply(text="2"),
        ]
        with patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("1+1?", provider=_fake_provider(responses))
        assistant_blocks = result.history[1].content
        assert all(b["type"] != "text" for b in assistant_blocks)

    def test_tool_error_string_flows_back_as_tool_result(self):
        """`execute_tool` already returns error strings; harness just forwards them."""
        with patch("src.agent.harness.execute_tool",
                   return_value="Error: invalid arguments for 'calculator'"):
            result = run_agent("q",
                               provider=_fake_provider(self._scenario()))
        tool_result = result.history[2].content[0]
        assert tool_result["content"].startswith("Error:")
        assert result.success is True   # agent still finished cleanly

    def test_multiple_tool_calls_in_one_turn_all_counted(self):
        """If a turn returns N tool calls, total_tool_calls increases by N."""
        responses = [
            _reply(text="",
                   stop_reason="tool_use",
                   tool_calls=[_calc_call("a"), _calc_call("b"), _calc_call("c")]),
            _reply(text="done"),
        ]
        with patch("src.agent.harness.execute_tool", return_value="ok"):
            result = run_agent("q", provider=_fake_provider(responses))
        assert result.tool_calls_made == 3


# ---------------------------------------------------------------------------
# run_agent — max_turns validation
# ---------------------------------------------------------------------------

class TestMaxTurnsValidation:
    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    def test_rejects_non_positive_max_turns(self, bad_value):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            run_agent("q",
                      provider=_fake_provider([_reply()]),
                      max_turns=bad_value)

    def test_provider_not_called_when_max_turns_invalid(self):
        provider = _fake_provider([_reply()])
        with pytest.raises(ValueError):
            run_agent("q", provider=provider, max_turns=0)
        provider.complete.assert_not_called()

    def test_accepts_max_turns_one(self):
        """Boundary: max_turns=1 is valid and runs exactly one turn."""
        result = run_agent("q",
                           provider=_fake_provider([_reply(text="hi")]),
                           max_turns=1)
        assert result.turns == 1
        assert result.success is True


# ---------------------------------------------------------------------------
# run_agent — provider.complete() error handling
# ---------------------------------------------------------------------------

class TestProviderError:
    """When provider.complete() raises, the harness must return a failed
    AgentResult instead of propagating the exception."""

    def _failing_provider(self, exc):
        provider = MagicMock()
        provider.complete.side_effect = exc
        return provider

    def test_returns_agent_result_on_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = run_agent("q", provider=provider)
        assert isinstance(result, AgentResult)

    def test_marks_failure_on_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = run_agent("q", provider=provider)
        assert result.success is False

    def test_stop_reason_is_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = run_agent("q", provider=provider)
        assert result.stop_reason == "provider_error"

    def test_answer_contains_exception_message(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = run_agent("q", provider=provider)
        assert "invalid API key" in result.answer

    def test_turns_reflects_failed_attempt(self):
        """The turn that errored still counts as a turn attempted."""
        provider = self._failing_provider(RuntimeError("boom"))
        result = run_agent("q", provider=provider)
        assert result.turns == 1

    def test_history_preserved_on_provider_error(self):
        """User message is kept in history even after an early failure."""
        provider = self._failing_provider(RuntimeError("boom"))
        result = run_agent("hello", provider=provider)
        assert result.history[0] == Message(role="user", content="hello")

    def test_error_on_second_turn_keeps_prior_history(self):
        """If a tool turn succeeds and the next provider call fails,
        we still get the assistant + tool_result entries in history."""
        provider = MagicMock()
        provider.complete.side_effect = [
            _reply(text="thinking",
                   stop_reason="tool_use",
                   tool_calls=[_calc_call()]),
            RuntimeError("rate limited"),
        ]
        with patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("q", provider=provider)
        assert result.success is False
        assert result.stop_reason == "provider_error"
        assert result.turns == 2
        assert result.tool_calls_made == 1
        # user, assistant(tool_use), user(tool_result)
        assert len(result.history) == 3
        assert result.history[1].role == "assistant"
        assert result.history[2].role == "user"


# ---------------------------------------------------------------------------
# run_agent — max_turns safety net
# ---------------------------------------------------------------------------

class TestRunAgentMaxTurns:
    """When the model keeps calling tools forever, run_agent must stop cleanly."""

    def _looping_responses(self, n):
        return [_reply(text="t", stop_reason="tool_use",
                       tool_calls=[_calc_call("toolu_x")])
                for _ in range(n)]

    def test_stops_after_max_turns(self):
        with patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop",
                               provider=_fake_provider(self._looping_responses(10)),
                               max_turns=3)
        assert result.turns == 3

    def test_marks_failure_on_max_turns(self):
        with patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop",
                               provider=_fake_provider(self._looping_responses(10)),
                               max_turns=3)
        assert result.success is False

    def test_stop_reason_is_max_turns_exceeded(self):
        with patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop",
                               provider=_fake_provider(self._looping_responses(10)),
                               max_turns=3)
        assert result.stop_reason == "max_turns_exceeded"


# ---------------------------------------------------------------------------
# Direct tests for the extracted helpers
# ---------------------------------------------------------------------------

from src.agent.harness import _build_assistant_message, _run_tool_call


class TestBuildAssistantMessage:
    def test_text_block_precedes_tool_use_blocks(self):
        resp = _reply(text="Let me compute",
                      stop_reason="tool_use",
                      tool_calls=[_calc_call()])
        msg = _build_assistant_message(resp)
        assert msg.role == "assistant"
        assert msg.content[0]["type"] == "text"
        assert msg.content[0]["text"] == "Let me compute"
        assert msg.content[1]["type"] == "tool_use"

    def test_no_text_block_when_text_empty(self):
        resp = _reply(text="", stop_reason="tool_use",
                      tool_calls=[_calc_call()])
        msg = _build_assistant_message(resp)
        assert all(b["type"] != "text" for b in msg.content)

    def test_tool_use_fields_mapped(self):
        resp = _reply(text="",
                      stop_reason="tool_use",
                      tool_calls=[_calc_call(id_="toolu_abc")])
        msg = _build_assistant_message(resp)
        block = msg.content[0]
        assert block["id"] == "toolu_abc"
        assert block["name"] == "calculator"
        assert block["input"] == {"operation": "add", "a": 2, "b": 3}

    def test_handles_none_tool_calls(self):
        """If tool_calls is None (plain text response), no tool_use blocks added."""
        resp = _reply(text="just text", stop_reason="end_turn", tool_calls=None)
        msg = _build_assistant_message(resp)
        assert msg.content == [{"type": "text", "text": "just text"}]


class TestRunToolCall:
    def test_returns_tool_result_block(self):
        tc = _calc_call(id_="toolu_xyz")
        with patch("src.agent.harness.execute_tool", return_value="5"):
            block = _run_tool_call(tc)
        assert block == {
            "type": "tool_result",
            "tool_use_id": "toolu_xyz",
            "content": "5",
        }

    def test_passes_tool_call_to_execute_tool(self):
        tc = _calc_call()
        with patch("src.agent.harness.execute_tool",
                   return_value="x") as mock_exec:
            _run_tool_call(tc)
        mock_exec.assert_called_once_with(tc)
