"""
Tests for the agent harness (src/agent/harness.py) and the AgentResult
dataclass added in phase/3/agent-loop.

The provider is fully mocked — no API calls, no key required.
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


def _patched_provider(responses):
    """Return a mock provider whose .complete() yields the given responses in order."""
    provider = MagicMock()
    provider.complete.side_effect = list(responses)
    return provider


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_fields_round_trip(self):
        result = AgentResult(
            answer="hi",
            success=True,
            turns=2,
            tool_calls_made=1,
            input_tokens=10,
            output_tokens=20,
            stop_reason="end_turn",
            history=[],
        )
        assert result.answer == "hi"
        assert result.success is True
        assert result.turns == 2
        assert result.tool_calls_made == 1
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.stop_reason == "end_turn"
        assert result.history == []


# ---------------------------------------------------------------------------
# run_agent — happy paths
# ---------------------------------------------------------------------------

class TestRunAgentNoTools:
    """Model finishes on turn 1 — no tool calls."""

    def test_returns_agent_result(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply(text="Paris")])):
            result = run_agent("Capital of France?")
        assert isinstance(result, AgentResult)

    def test_answer_taken_from_response_text(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply(text="Paris")])):
            result = run_agent("Capital of France?")
        assert result.answer == "Paris"

    def test_marks_success(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply()])):
            result = run_agent("q")
        assert result.success is True

    def test_single_turn(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply()])):
            result = run_agent("q")
        assert result.turns == 1

    def test_no_tool_calls_made(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply()])):
            result = run_agent("q")
        assert result.tool_calls_made == 0

    def test_stop_reason_propagated(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply(stop_reason="end_turn")])):
            result = run_agent("q")
        assert result.stop_reason == "end_turn"

    def test_history_contains_user_message(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider([_reply()])):
            result = run_agent("hello there")
        assert result.history[0] == Message(role="user", content="hello there")


# ---------------------------------------------------------------------------
# run_agent — tool-calling path
# ---------------------------------------------------------------------------

class TestRunAgentWithTools:
    """Model wants a tool on turn 1, then gives final answer on turn 2."""

    def _scenario(self):
        tool_call = ToolCall(
            id="toolu_1",
            name="calculator",
            arguments={"operation": "add", "a": 2, "b": 3},
        )
        return [
            _reply(text="Let me compute",
                   stop_reason="tool_use",
                   tool_calls=[tool_call]),
            _reply(text="The answer is 5"),
        ]

    def test_two_turns(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?")
        assert result.turns == 2

    def test_tool_calls_counted(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?")
        assert result.tool_calls_made == 1

    def test_final_answer_from_last_turn(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?")
        assert result.answer == "The answer is 5"

    def test_tokens_accumulated_across_turns(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("2+3?")
        # 3 + 3 input, 4 + 4 output across the two turns
        assert result.input_tokens == 6
        assert result.output_tokens == 8

    def test_tools_kwarg_forwarded_to_provider(self):
        provider = _patched_provider(self._scenario())
        with patch("src.agent.harness.get_provider", return_value=provider), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            run_agent("q", tools=["fake-tool-obj"])
        # Both turns should have been called with the same tools list
        for call in provider.complete.call_args_list:
            assert call.kwargs["tools"] == ["fake-tool-obj"]

    def test_system_kwarg_forwarded_to_provider(self):
        provider = _patched_provider([_reply()])
        with patch("src.agent.harness.get_provider", return_value=provider):
            run_agent("q", system="You are a bot.")
        assert provider.complete.call_args.kwargs["system"] == "You are a bot."

    def test_execute_tool_receives_the_tool_call(self):
        provider = _patched_provider(self._scenario())
        with patch("src.agent.harness.get_provider", return_value=provider), \
             patch("src.agent.harness.execute_tool",
                   return_value="5") as mock_exec:
            run_agent("q")
        passed = mock_exec.call_args.args[0]
        assert isinstance(passed, ToolCall)
        assert passed.name == "calculator"

    def test_history_includes_assistant_tool_use_and_tool_result(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", return_value="5"):
            result = run_agent("q")
        # history: user query, assistant(tool_use), user(tool_result), assistant(final)?
        # Final assistant turn isn't appended — run_agent returns before that.
        assert result.history[0].role == "user"
        assert result.history[1].role == "assistant"
        # assistant content is a list with at least one tool_use block
        types = [b["type"] for b in result.history[1].content]
        assert "tool_use" in types
        # third entry is the tool_result message
        assert result.history[2].role == "user"
        assert result.history[2].content[0]["type"] == "tool_result"
        assert result.history[2].content[0]["tool_use_id"] == "toolu_1"
        assert result.history[2].content[0]["content"] == "5"

    def test_empty_reply_text_skips_text_block_in_history(self):
        """If response.text is empty, no text block is added to history."""
        tool_call = ToolCall(id="toolu_1", name="calculator",
                             arguments={"operation": "add", "a": 1, "b": 1})
        responses = [
            _reply(text="", stop_reason="tool_use", tool_calls=[tool_call]),
            _reply(text="2"),
        ]
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(responses)), \
             patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("1+1?")
        assistant_blocks = result.history[1].content
        assert all(b["type"] != "text" for b in assistant_blocks)

    def test_tool_exception_captured_as_result_string(self):
        """If execute_tool raises, the agent must keep going with an error string."""
        def boom(_tc):
            raise RuntimeError("kaboom")

        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._scenario())), \
             patch("src.agent.harness.execute_tool", side_effect=boom):
            result = run_agent("q")
        tool_result_block = result.history[2].content[0]
        assert "Tool error" in tool_result_block["content"]
        assert "kaboom" in tool_result_block["content"]
        assert result.success is True   # agent still finished cleanly


# ---------------------------------------------------------------------------
# run_agent — verbose path (just exercise the print branches)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# run_agent — max_turns safety net
# ---------------------------------------------------------------------------

class TestRunAgentMaxTurns:
    """When the model keeps calling tools forever, run_agent must stop cleanly."""

    def _looping_responses(self, n):
        tc = ToolCall(id="toolu_x", name="calculator",
                      arguments={"operation": "add", "a": 1, "b": 1})
        return [_reply(text="t", stop_reason="tool_use", tool_calls=[tc])
                for _ in range(n)]

    def test_stops_after_max_turns(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._looping_responses(10))), \
             patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop", max_turns=3)
        assert result.turns == 3

    def test_marks_failure_on_max_turns(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._looping_responses(10))), \
             patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop", max_turns=3)
        assert result.success is False

    def test_stop_reason_is_max_turns_exceeded(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_patched_provider(self._looping_responses(10))), \
             patch("src.agent.harness.execute_tool", return_value="2"):
            result = run_agent("loop", max_turns=3)
        assert result.stop_reason == "max_turns_exceeded"

