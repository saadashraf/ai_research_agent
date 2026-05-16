"""Tests for dataclasses and helpers in src/providers/base.py."""

from src.providers.base import CompletionResponse, ToolCall


def _resp(stop_reason, tool_calls=None):
    return CompletionResponse(
        text="",
        input_tokens=0,
        output_tokens=0,
        model="m",
        stop_reason=stop_reason,
        tool_calls=tool_calls,
    )


class TestWantsTool:
    def test_true_when_stop_reason_and_tool_calls_present(self):
        call = ToolCall(id="t1", name="calculator", arguments={})
        assert _resp("tool_use", [call]).wants_tool() is True

    def test_false_when_stop_reason_is_end_turn(self):
        call = ToolCall(id="t1", name="calculator", arguments={})
        assert _resp("end_turn", [call]).wants_tool() is False

    def test_false_when_tool_calls_is_none(self):
        assert _resp("tool_use", None).wants_tool() is False

    def test_false_when_tool_calls_is_empty_list(self):
        assert _resp("tool_use", []).wants_tool() is False
