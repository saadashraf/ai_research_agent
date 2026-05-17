import logging
from typing import Optional

from src.providers import get_provider
from src.providers.base import (
    AgentResult,
    CompletionResponse,
    Message,
    ModelProvider,
    Tool,
    ToolCall,
)
from src.tools import execute_tool


logger = logging.getLogger(__name__)


def _build_assistant_message(response: CompletionResponse) -> Message:
    """
    Convert a CompletionResponse with tool_use blocks into the
    assistant-turn Message that gets appended to history.

    Order matters: any text block must precede tool_use blocks
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


def _run_tool_call(tool_call: ToolCall) -> dict:
    """
    Execute one tool call and return the tool_result block ready to
    drop into a user-turn message.

    Tool exceptions are already converted to error strings by
    `execute_tool`, so no try/except is needed here.
    """
    logger.debug("[Tool]  %s(%s)", tool_call.name, tool_call.arguments)
    result = execute_tool(tool_call)
    logger.debug("[Result] %s", result)

    return {
        "type": "tool_result",
        "tool_use_id": tool_call.id,   # must match the tool_use id above
        "content": result,
    }


def run_agent(
        user_query: str,
        system: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        max_turns: int = 5,
        provider: Optional[ModelProvider] = None,
        ) -> AgentResult:
    """
    Runs the agent loop until one of three things happens:
      1. Model returns stop_reason="end_turn"  → clean finish
      2. Turn count hits max_turns             → forced stop, safety net
      3. An unexpected exception               → captured, returned in result

    A `provider` may be injected (handy for tests and for reusing a
    single client across multiple runs); otherwise one is built from config.

    Per-step progress is emitted at DEBUG level via the module logger.
    Enable with:  logging.basicConfig(level=logging.DEBUG)
    """
    if provider is None:
        provider = get_provider()

    # Seed history with the user's message.
    # Every subsequent append grows this list — the agent's memory.
    history: list[Message] = [Message(role="user", content=user_query)]

    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    turns = 0

    logger.debug("[Agent] Starting run")
    logger.debug("[User]  %s", user_query)

    # ── THE LOOP ──────────────────────────────────────────────────
    while turns < max_turns:
        turns += 1
        logger.debug("[Turn %d] Calling model...", turns)

        response = provider.complete(
            messages=history,
            system=system,
            tools=tools,
        )

        total_input_tokens  += response.input_tokens
        total_output_tokens += response.output_tokens

        # ── STOP: model is done ───────────────────────────────────
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

        # ── CONTINUE: model wants tools ───────────────────────────
        # The assistant message (with tool_use blocks) must go into
        # history BEFORE the tool results. If you flip the order,
        # the Anthropic API rejects the request.
        history.append(_build_assistant_message(response))

        # Run each tool the model asked for, collecting one result block per call.
        tool_results = []
        for tool_call in response.tool_calls:
            result_block = _run_tool_call(tool_call)
            tool_results.append(result_block)
            total_tool_calls += 1

        # All tool results go back as a single user message.
        history.append(Message(role="user", content=tool_results))
        # Loop continues → model sees the results next turn

    # ── SAFETY NET: too many turns ────────────────────────────────
    logger.debug("[Agent] Hit MAX_TURNS (%d). Stopping.", max_turns)

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
