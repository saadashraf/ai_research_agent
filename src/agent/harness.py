from src.providers import get_provider
from src.providers.base import AgentResult, Message
from src.tools import ALL_TOOLS, execute_tool

# Safety ceiling — the loop WILL stop here even if the model keeps asking for tools.
# 10 is generous for a research agent; raise it if needed.
MAX_TURNS = 10

SYSTEM = """You are a helpful research assistant with access to tools.
Use the calculator tool for any arithmetic — never compute in your head.
Use read_file when asked about file contents.
Use web_search when asked about current information from the web.
After receiving a tool result, always summarise what you learned before
deciding whether to call another tool or give a final answer.
"""


def run_agent(user_query: str, verbose: bool = True) -> AgentResult:
    """
    Runs the agent loop until one of three things happens:
      1. Model returns stop_reason="end_turn"  → clean finish
      2. Turn count hits MAX_TURNS             → forced stop, safety net
      3. An unexpected exception               → captured, returned in result

    verbose=True prints each step
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

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Agent] Starting run")
        print(f"[User]  {user_query}")
        print(f"{'='*60}")

    # ── THE LOOP ──────────────────────────────────────────────────
    while turns < MAX_TURNS:
        turns += 1

        if verbose:
            print(f"\n[Turn {turns}] Calling model...")

        # ── MODEL CALL ────────────────────────────────────────────
        # We pass the FULL history every time. The model has no memory
        # of its own — the history list is its memory.
        response = provider.complete(
            messages=history,
            system=SYSTEM,
            tools=ALL_TOOLS,
        )

        # Accumulate token usage across turns
        total_input_tokens  += response.input_tokens
        total_output_tokens += response.output_tokens

        # ── STOP: model is done ───────────────────────────────────
        if not response.wants_tool():
            if verbose:
                print(f"[Turn {turns}] Model finished. stop_reason={response.stop_reason}")
                print(f"[Answer] {response.text}")

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
        history.append(Message(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
                for tc in response.tool_calls   # may be multiple tools in one turn
            ]
        ))

        # Now run each tool and collect results
        tool_results = []
        for tool_call in response.tool_calls:
            total_tool_calls += 1

            if verbose:
                print(f"[Tool]  {tool_call.name}({tool_call.arguments})")

            # Wrap in try/except so one broken tool doesn't abort the run.
            # We return the error string as the result — the model will
            # see it and (usually) explain what went wrong to the user.
            try:
                result = execute_tool(tool_call)
            except Exception as e:
                result = f"Tool error: {e}"

            if verbose:
                print(f"[Result] {result}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,  # ← must match the tool_use id above
                "content": result,
            })

        # All tool results go back as a single user message.
        history.append(Message(role="user", content=tool_results))
        # Loop continues → model sees the results next turn

    # ── SAFETY NET: too many turns ────────────────────────────────
    # We exhausted MAX_TURNS without a clean end_turn.
    # This is abnormal — log it clearly.
    if verbose:
        print(f"\n[Agent] Hit MAX_TURNS ({MAX_TURNS}). Stopping.")

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