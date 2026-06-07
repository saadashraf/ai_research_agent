"""
Tests for the agent harness (src/agent/harness.py) and the AgentResult
dataclass.

The harness is async and provider-agnostic:
  * run_agent(...) is a coroutine; the provider is fully mocked (no API calls).
  * Tools are executed through an injected `executor` callable — the harness no
    longer knows about execute_tool. The executor receives a single ToolCall and
    returns a string; it may be sync OR async (the harness awaits if needed).
    This is the contract both the local execute_tool and MCPClient.call_tool
    satisfy, which is what lets the harness drive local or remote tools
    interchangeably.

Async tests run under pytest-asyncio (asyncio_mode = auto in pytest.ini), so
test coroutines are awaited directly. Purely synchronous tests (dataclasses,
helper builders) stay plain functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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


def _executor(return_value="5"):
    """A sync executor stand-in: takes a ToolCall, returns a string."""
    return MagicMock(return_value=return_value)


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

    async def test_history_is_list_of_messages(self):
        """run_agent populates history with Message instances."""
        result = await run_agent("hi",
                                 provider=_fake_provider([_reply(text="hello")]))
        assert isinstance(result.history, list)
        assert all(isinstance(m, Message) for m in result.history)


# ---------------------------------------------------------------------------
# Provider injection
# ---------------------------------------------------------------------------

class TestProviderInjection:
    async def test_injected_provider_is_used(self):
        provider = _fake_provider([_reply(text="hi")])
        result = await run_agent("q", provider=provider)
        assert result.answer == "hi"
        provider.complete.assert_called_once()

    async def test_get_provider_called_when_none_injected(self):
        with patch("src.agent.harness.get_provider",
                   return_value=_fake_provider([_reply(text="hi")])) as factory:
            await run_agent("q")
        factory.assert_called_once()

    async def test_get_provider_not_called_when_provider_injected(self):
        with patch("src.agent.harness.get_provider") as factory:
            await run_agent("q", provider=_fake_provider([_reply(text="hi")]))
        factory.assert_not_called()


# ---------------------------------------------------------------------------
# run_agent — happy paths (no tools, finish on turn 1)
# ---------------------------------------------------------------------------

class TestRunAgentNoTools:
    async def test_returns_agent_result(self):
        result = await run_agent("Capital of France?",
                                 provider=_fake_provider([_reply(text="Paris")]))
        assert isinstance(result, AgentResult)

    async def test_answer_taken_from_response_text(self):
        result = await run_agent("Capital of France?",
                                 provider=_fake_provider([_reply(text="Paris")]))
        assert result.answer == "Paris"

    async def test_marks_success(self):
        result = await run_agent("q", provider=_fake_provider([_reply()]))
        assert result.success is True

    async def test_single_turn(self):
        result = await run_agent("q", provider=_fake_provider([_reply()]))
        assert result.turns == 1

    async def test_no_tool_calls_made(self):
        result = await run_agent("q", provider=_fake_provider([_reply()]))
        assert result.tool_calls_made == 0

    async def test_stop_reason_propagated(self):
        result = await run_agent("q", provider=_fake_provider(
            [_reply(stop_reason="end_turn")]))
        assert result.stop_reason == "end_turn"

    async def test_history_contains_user_message(self):
        result = await run_agent("hello there",
                                 provider=_fake_provider([_reply()]))
        assert result.history[0] == Message(role="user", content="hello there")

    async def test_executor_not_required_when_no_tools(self):
        """No tool requested → executor is never needed, may be omitted."""
        result = await run_agent("q", provider=_fake_provider([_reply()]))
        assert result.success is True


# ---------------------------------------------------------------------------
# run_agent — tool-calling path (executor injection)
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

    async def test_two_turns(self):
        result = await run_agent("2+3?", provider=_fake_provider(self._scenario()),
                                 executor=_executor("5"))
        assert result.turns == 2

    async def test_tool_calls_counted(self):
        result = await run_agent("2+3?", provider=_fake_provider(self._scenario()),
                                 executor=_executor("5"))
        assert result.tool_calls_made == 1

    async def test_final_answer_from_last_turn(self):
        result = await run_agent("2+3?", provider=_fake_provider(self._scenario()),
                                 executor=_executor("5"))
        assert result.answer == "The answer is 5"

    async def test_tokens_accumulated_across_turns(self):
        result = await run_agent("2+3?", provider=_fake_provider(self._scenario()),
                                 executor=_executor("5"))
        # 3 + 3 input, 4 + 4 output across the two turns
        assert result.input_tokens == 6
        assert result.output_tokens == 8

    async def test_tools_kwarg_forwarded_to_provider(self):
        provider = _fake_provider(self._scenario())
        await run_agent("q", tools=["fake-tool-obj"], provider=provider,
                        executor=_executor("5"))
        for call in provider.complete.call_args_list:
            assert call.kwargs["tools"] == ["fake-tool-obj"]

    async def test_system_kwarg_forwarded_to_provider(self):
        provider = _fake_provider([_reply()])
        await run_agent("q", system="You are a bot.", provider=provider)
        assert provider.complete.call_args.kwargs["system"] == "You are a bot."

    async def test_executor_receives_the_tool_call(self):
        executor = _executor("5")
        await run_agent("q", provider=_fake_provider(self._scenario()),
                        executor=executor)
        passed = executor.call_args.args[0]
        assert isinstance(passed, ToolCall)
        assert passed.name == "calculator"

    async def test_history_includes_assistant_tool_use_and_tool_result(self):
        result = await run_agent("q", provider=_fake_provider(self._scenario()),
                                 executor=_executor("5"))
        # history: user query, assistant(tool_use), user(tool_result)
        assert result.history[0].role == "user"
        assert result.history[1].role == "assistant"
        types = [b["type"] for b in result.history[1].content]
        assert "tool_use" in types
        assert result.history[2].role == "user"
        assert result.history[2].content[0]["type"] == "tool_result"
        assert result.history[2].content[0]["tool_use_id"] == "toolu_1"
        assert result.history[2].content[0]["content"] == "5"

    async def test_empty_reply_text_skips_text_block_in_history(self):
        responses = [
            _reply(text="", stop_reason="tool_use", tool_calls=[_calc_call()]),
            _reply(text="2"),
        ]
        result = await run_agent("1+1?", provider=_fake_provider(responses),
                                 executor=_executor("2"))
        assistant_blocks = result.history[1].content
        assert all(b["type"] != "text" for b in assistant_blocks)

    async def test_tool_error_string_flows_back_as_tool_result(self):
        """The executor returns error strings; the harness just forwards them."""
        result = await run_agent(
            "q", provider=_fake_provider(self._scenario()),
            executor=_executor("Error: invalid arguments for 'calculator'"))
        tool_result = result.history[2].content[0]
        assert tool_result["content"].startswith("Error:")
        assert result.success is True   # agent still finished cleanly

    async def test_multiple_tool_calls_in_one_turn_all_counted(self):
        """If a turn returns N tool calls, total_tool_calls increases by N."""
        responses = [
            _reply(text="",
                   stop_reason="tool_use",
                   tool_calls=[_calc_call("a"), _calc_call("b"), _calc_call("c")]),
            _reply(text="done"),
        ]
        result = await run_agent("q", provider=_fake_provider(responses),
                                 executor=_executor("ok"))
        assert result.tool_calls_made == 3

    async def test_async_executor_is_awaited(self):
        """The harness supports async executors (e.g. MCPClient.call_tool)."""
        async_exec = AsyncMock(return_value="5")
        result = await run_agent("2+3?", provider=_fake_provider(self._scenario()),
                                 executor=async_exec)
        async_exec.assert_awaited_once()
        assert result.history[2].content[0]["content"] == "5"


# ---------------------------------------------------------------------------
# run_agent — executor required only when a tool is actually called
# ---------------------------------------------------------------------------

class TestExecutorRequired:
    async def test_missing_executor_raises_when_tool_requested(self):
        responses = [
            _reply(text="", stop_reason="tool_use", tool_calls=[_calc_call()]),
            _reply(text="done"),
        ]
        with pytest.raises(ValueError, match="no executor"):
            await run_agent("q", provider=_fake_provider(responses))


# ---------------------------------------------------------------------------
# run_agent — max_turns validation
# ---------------------------------------------------------------------------

class TestMaxTurnsValidation:
    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    async def test_rejects_non_positive_max_turns(self, bad_value):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            await run_agent("q",
                            provider=_fake_provider([_reply()]),
                            max_turns=bad_value)

    async def test_provider_not_called_when_max_turns_invalid(self):
        provider = _fake_provider([_reply()])
        with pytest.raises(ValueError):
            await run_agent("q", provider=provider, max_turns=0)
        provider.complete.assert_not_called()

    async def test_accepts_max_turns_one(self):
        """Boundary: max_turns=1 is valid and runs exactly one turn."""
        result = await run_agent("q",
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

    async def test_returns_agent_result_on_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = await run_agent("q", provider=provider)
        assert isinstance(result, AgentResult)

    async def test_marks_failure_on_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = await run_agent("q", provider=provider)
        assert result.success is False

    async def test_stop_reason_is_provider_error(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = await run_agent("q", provider=provider)
        assert result.stop_reason == "provider_error"

    async def test_answer_contains_exception_message(self):
        provider = self._failing_provider(RuntimeError("invalid API key"))
        result = await run_agent("q", provider=provider)
        assert "invalid API key" in result.answer

    async def test_turns_reflects_failed_attempt(self):
        """The turn that errored still counts as a turn attempted."""
        provider = self._failing_provider(RuntimeError("boom"))
        result = await run_agent("q", provider=provider)
        assert result.turns == 1

    async def test_history_preserved_on_provider_error(self):
        """User message is kept in history even after an early failure."""
        provider = self._failing_provider(RuntimeError("boom"))
        result = await run_agent("hello", provider=provider)
        assert result.history[0] == Message(role="user", content="hello")

    async def test_error_on_second_turn_keeps_prior_history(self):
        """If a tool turn succeeds and the next provider call fails,
        we still get the assistant + tool_result entries in history."""
        provider = MagicMock()
        provider.complete.side_effect = [
            _reply(text="thinking",
                   stop_reason="tool_use",
                   tool_calls=[_calc_call()]),
            RuntimeError("rate limited"),
        ]
        result = await run_agent("q", provider=provider, executor=_executor("5"))
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

    async def test_stops_after_max_turns(self):
        result = await run_agent("loop",
                                 provider=_fake_provider(self._looping_responses(10)),
                                 executor=_executor("2"), max_turns=3)
        assert result.turns == 3

    async def test_marks_failure_on_max_turns(self):
        result = await run_agent("loop",
                                 provider=_fake_provider(self._looping_responses(10)),
                                 executor=_executor("2"), max_turns=3)
        assert result.success is False

    async def test_stop_reason_is_max_turns_exceeded(self):
        result = await run_agent("loop",
                                 provider=_fake_provider(self._looping_responses(10)),
                                 executor=_executor("2"), max_turns=3)
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
    """_run_tool_call is async and runs a tool through the injected executor."""

    async def test_returns_tool_result_block(self):
        tc = _calc_call(id_="toolu_xyz")
        block = await _run_tool_call(tc, _executor("5"))
        assert block == {
            "type": "tool_result",
            "tool_use_id": "toolu_xyz",
            "content": "5",
        }

    async def test_passes_tool_call_to_executor(self):
        tc = _calc_call()
        executor = _executor("x")
        await _run_tool_call(tc, executor)
        executor.assert_called_once_with(tc)

    async def test_awaits_async_executor(self):
        tc = _calc_call(id_="toolu_async")
        async_exec = AsyncMock(return_value="async-result")
        block = await _run_tool_call(tc, async_exec)
        assert block["content"] == "async-result"
        async_exec.assert_awaited_once_with(tc)
