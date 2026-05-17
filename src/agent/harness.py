import logging

from src.providers import get_provider
from src.providers.base import AgentResult, Message
from src.tools import execute_tool


logger = logging.getLogger(__name__)


def run_agent(
        user_query: str, system: str = None, tools: list = None, max_turns: int = 5
        ) -> AgentResult:
    """
    Runs the agent loop until one of three things happens:
      1. Model returns stop_reason="end_turn"  → clean finish
      2. Turn count hits max_turns             → forced stop, safety net
      3. An unexpected exception               → captured, returned in result

    Per-step progress is emitted at DEBUG level via the module logger.
    Enable with:  logging.basicConfig(level=logging.DEBUG)
    """
    provider = get_provider()

    # Seed history with the user's message.
    # Every subsequent append grows this list — this is the agent's memory.
    history: list[Message] = [Message(role="user", content=user_query)]

    # Counters — accumulated across all turns
    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    turns = 0

    logger.debug("=" * 60)
    logger.debug("[Agent] Starting run")
    logger.debug("[User]  %s", user_query)
    logger.debug("=" * 60)

    # ── THE LOOP ──────────────────────────────────────────────────
    while turns < max_turns:
        turns += 1

        logger.debug("[Turn %d] Calling model...", turns)

        # ── MODEL CALL ────────────────────────────────────────────
        # We pass the FULL history every time. The model has no memory
        # of its own — the history list is its memory.
        response = provider.complete(
            messages=history,
            system=system,
            tools=tools,
        )

        # Accumulate token usage across turns
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
        # The model's reply (containing tool_use blocks) must go into
        # history BEFORE the tool results. Anthropic requires this order:
        #   assistant: [tool_use block]
        #   user:      [tool_result block]
        # If you flip the order, the API rejects the request.

        assistant_content = []

        if response.text:  # only add if Claude actually said something
            logger.debug("[Thinking] %s", response.text)
            assistant_content.append({"type": "text", "text": response.text})

        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        history.append(Message(role="assistant", content=assistant_content))

        # Now run each tool and collect results
        tool_results = []
        for tool_call in response.tool_calls:
            total_tool_calls += 1

            logger.debug("[Tool]  %s(%s)", tool_call.name, tool_call.arguments)

            # Wrap in try/except so one broken tool doesn't abort the run.
            # We return the error string as the result — the model will
            # see it and (usually) explain what went wrong to the user.
            try:
                result = execute_tool(tool_call)
            except Exception as e:
                result = f"Tool error: {e}"

            logger.debug("[Result] %s", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,  # ← must match the tool_use id above
                "content": result,
            })

        # All tool results go back as a single user message.
        history.append(Message(role="user", content=tool_results))
        # Loop continues → model sees the results next turn

    # ── SAFETY NET: too many turns ────────────────────────────────
    # We exhausted max_turns without a clean end_turn.
    # This is abnormal — log it clearly.
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
