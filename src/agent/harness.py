import logging
import inspect
from typing import Optional, Callable

from src.providers import get_provider
from src.providers.base import (
    AgentResult,
    CompletionResponse,
    Message,
    ModelProvider,
    Tool,
    ToolCall,
)

logger = logging.getLogger(__name__)


def _build_assistant_message(response: CompletionResponse) -> Message:
    """
    Convert a CompletionResponse into the assistant-turn Message
    that gets appended to history.

    Order matters: text block must precede tool_use blocks
    (Anthropic API requirement).
    """
    blocks: list[dict] = []

    if response.text:
        logger.debug("[Thinking] %s", response.text)
        blocks.append({"type": "text", "text": response.text})

    if response.tool_calls:
        for tc in response.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

    return Message(role="assistant", content=blocks)


async def _run_tool_call(tool_call: ToolCall, executor: Callable) -> dict:
    """
    Execute one tool call using the provided executor.
    Handles both sync and async executors transparently —
    sync execute_tool and async MCP client both work without
    the caller needing to care which one it is.
    """
    logger.debug("[Tool]  %s(%s)", tool_call.name, tool_call.arguments)

    result = executor(tool_call)

    if inspect.isawaitable(result):
        result = await result

    logger.debug("[Result] %s", result)

    return {
        "type": "tool_result",
        "tool_use_id": tool_call.id,
        "content": result,
    }


async def run_agent(
        user_query: str,
        system: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        executor: Optional[Callable] = None,
        max_turns: int = 5,
        provider: Optional[ModelProvider] = None,
        ) -> AgentResult:
    """
    Async agent loop. Runs until one of three things happens:
      1. Model returns stop_reason="end_turn"  → clean finish
      2. Turn count hits max_turns             → forced stop
      3. provider.complete() raises            → captured in result

    executor: callable that runs a tool, sync or async.
              Only required if the model actually calls a tool.
              Pass execute_tool for local tools or an async MCP
              executor for remote tools.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    if provider is None:
        provider = get_provider()

    history: list[Message] = [Message(role="user", content=user_query)]

    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    turns = 0

    logger.debug("[Agent] Starting run")
    logger.debug("[User]  %s", user_query)

    while turns < max_turns:
        turns += 1
        logger.debug("[Turn %d] Calling model...", turns)

        try:
            response = provider.complete(
                messages=history,
                system=system,
                tools=tools,
            )
        except Exception as exc:
            logger.exception("[Turn %d] provider.complete() failed", turns)
            return AgentResult(
                answer=f"Agent stopped: provider error: {exc}",
                success=False,
                turns=turns,
                tool_calls_made=total_tool_calls,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                stop_reason="provider_error",
                history=history,
            )

        total_input_tokens  += response.input_tokens
        total_output_tokens += response.output_tokens

        logger.info("[Turn %d] Model returned response: %s", turns, response.text)

        if not response.wants_tool():
            logger.debug("[Turn %d] Model finished. stop_reason=%s",
                         turns, response.stop_reason)
            logger.debug("[Answer] %s", response.text)

            return AgentResult(
                answer=response.text,
                success=True,
                turns=turns,
                tool_calls_made=total_tool_calls,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                stop_reason=response.stop_reason,
                history=history,
            )

        # Model wants tools — now executor matters
        if executor is None:
            raise ValueError(
                "Model requested a tool but no executor was provided — "
                "pass execute_tool for local tools or an MCP executor"
            )

        history.append(_build_assistant_message(response))

        tool_results = []
        for tool_call in response.tool_calls:
            result_block = await _run_tool_call(tool_call, executor)
            tool_results.append(result_block)
            total_tool_calls += 1

        history.append(Message(role="user", content=tool_results))

    logger.debug("[Agent] Hit max_turns (%d). Stopping.", max_turns)

    return AgentResult(
        answer="Agent stopped: exceeded maximum turn limit without a final answer.",
        success=False,
        turns=turns,
        tool_calls_made=total_tool_calls,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        stop_reason="max_turns_exceeded",
        history=history,
    )