import logging
from contextlib import AsyncExitStack
from typing import Optional

from src.agent.harness import run_agent
from src.mcp_client.client import MCPClient
from src.providers.base import AgentResult, ModelProvider

logger = logging.getLogger(__name__)


class Agent:
    """
    A self-contained agent with a fixed identity and toolset.

    Usage:
        async with Agent(system=SYSTEM) as agent:
            result = await agent.run("What is 1234 * 5678?")
            print(result.answer)

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
        self._stack: Optional[AsyncExitStack] = None
        self._client: Optional[MCPClient] = None
        self._tools = None
        self._executor = None

    async def __aenter__(self):
        """
        Starts the MCP server subprocess and runs the handshake.
        Discovers available tools so they're ready for every run() call.

        Each resource is entered through an AsyncExitStack, which records
        it for reverse-order cleanup the moment it enters successfully.
        If any later step fails, aclose() unwinds exactly what was entered
        so far — no orphaned server subprocess. Ownership of the stack is
        only handed to the instance via pop_all() once setup fully
        succeeds; until then a failure cleans up locally and re-raises.
        """
        stack = AsyncExitStack()
        try:
            self._client = await stack.enter_async_context(
                MCPClient(self._server)
            )

            self._tools = await self._client.list_tools()
            self._executor = self._client.call_tool

            logger.debug(
                "[Agent] Ready. Tools: %s", [t.name for t in self._tools]
            )

            # Setup succeeded — transfer cleanup duty to the instance so
            # __aexit__ owns it. pop_all() empties the local stack, so the
            # except below becomes a no-op if we get here.
            self._stack = stack.pop_all()
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(self, *args):
        """Shuts down the MCP server subprocess cleanly."""
        if self._stack:
            await self._stack.__aexit__(*args)

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