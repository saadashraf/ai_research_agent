import logging
from typing import Optional

from src.agent.harness import run_agent
from src.mcp_client.client import MCPClient
from src.providers.base import AgentResult, ModelProvider

logger = logging.getLogger(__name__)


def _make_executor(client: MCPClient):
    """
    Returns an async callable that routes tool calls through MCP.
    Kept private — callers never interact with executors directly.
    """
    async def executor(tool_call):
        return await client.call_tool(tool_call.name, tool_call.arguments)
    return executor


class Agent:
    """
    A self-contained agent with a fixed identity and toolset.

    The caller provides a goal (system prompt) and queries.
    Everything else — MCP lifecycle, tool discovery, executor
    wiring — is handled internally.

    Usage:
        async with Agent(system=SYSTEM) as agent:
            result = await agent.run("What is 1234 * 5678?")
            print(result.answer)

    This is the unit the orchestrator talks to in Phase 5.
    One agent = one goal = one set of tools = one system prompt.
    """

    def __init__(
        self,
        system: Optional[str] = None,
        server: str = "src/mcp_client/server.py",
        max_turns: int = 5,
        provider: Optional[ModelProvider] = None,
    ):
        self._system = system
        self._server = server
        self._max_turns = max_turns
        self._provider = provider

        # Populated during __aenter__ — not available until context is entered
        self._client: Optional[MCPClient] = None
        self._tools = None
        self._executor = None

    async def __aenter__(self):
        """
        Starts the MCP server subprocess and runs the handshake.
        Discovers available tools so they're ready for every run() call.
        """
        self._client = MCPClient(self._server)
        await self._client.__aenter__()

        self._tools = await self._client.list_tools()
        self._executor = _make_executor(self._client)

        logger.debug(
            "[Agent] Ready. Tools: %s", [t.name for t in self._tools]
        )
        return self

    async def __aexit__(self, *args):
        """Shuts down the MCP server subprocess cleanly."""
        if self._client:
            await self._client.__aexit__(*args)

    async def run(self, query: str) -> AgentResult:
        """
        Run a single query through the agent loop.
        The caller provides the question — nothing else.

        Can be called multiple times within the same context block.
        Each call is independent — no memory carried between runs.
        """
        if self._client is None:
            raise RuntimeError(
                "Agent must be used as a context manager — "
                "use `async with Agent(...) as agent:`"
            )

        logger.debug("[Agent] Running query: %s", query)

        return await run_agent(
            user_query=query,
            system=self._system,
            tools=self._tools,
            executor=self._executor,
            max_turns=self._max_turns,
            provider=self._provider,
        )