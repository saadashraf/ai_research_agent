"""
Tests for the Agent class (src/agent/agent.py) added in phase/4/mcp-client.

Design intentions being verified
--------------------------------
Agent is a thin, self-contained wrapper that:
  * is an async context manager — __aenter__ spins up an MCPClient, discovers
    tools once, and wires the client's call_tool as the executor;
  * exposes run(query) which simply delegates to run_agent with the stored
    config (system / tools / executor / max_turns / provider);
  * refuses to run() outside its context (helpful RuntimeError);
  * can run() many times within one context, reusing the tools discovered
    at enter time (no re-discovery per call);
  * NEVER orphans the server subprocess — if setup fails partway through
    __aenter__, whatever was already entered is torn down, and a failure
    inside the `async with` body still triggers cleanup.

No real MCP subprocess is spawned: MCPClient is replaced with a fake async
context manager, and run_agent is patched where it is used inside agent.py.
Async tests run under pytest-asyncio (asyncio_mode = auto in pytest.ini).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.agent import Agent
from src.providers.base import Tool


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeMCPClient:
    """
    Stand-in for MCPClient. Behaves as an async context manager and records
    its lifecycle so tests can assert it was entered and — crucially — always
    exited (no orphaned subprocess).
    """

    # The most recently constructed instance, so tests can inspect it after
    # the `async with` block has unwound.
    last_instance: "FakeMCPClient | None" = None

    def __init__(self, server, *, tools=None, list_tools_error=None):
        self.server = server
        self._tools = tools if tools is not None else []
        self._list_tools_error = list_tools_error
        self.entered = False
        self.exited = False
        self.exit_args = None
        self.list_tools_calls = 0
        FakeMCPClient.last_instance = self

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True
        self.exit_args = args
        return False  # never suppress

    async def list_tools(self):
        self.list_tools_calls += 1
        if self._list_tools_error is not None:
            raise self._list_tools_error
        return self._tools

    async def call_tool(self, tool_call):
        return f"result:{tool_call.name}"


def _patch_client(**kwargs):
    """
    Patch the MCPClient symbol referenced inside agent.py with a factory that
    builds a FakeMCPClient. kwargs (tools=, list_tools_error=) configure it.
    """
    def factory(server):
        return FakeMCPClient(server, **kwargs)

    return patch("src.agent.agent.MCPClient", side_effect=factory)


def _tool(name="calculator"):
    return Tool(name=name, description="desc", parameters=[])


# ---------------------------------------------------------------------------
# __aenter__ / __aexit__ lifecycle
# ---------------------------------------------------------------------------

class TestAgentLifecycle:
    async def test_enter_starts_and_returns_self(self):
        with _patch_client():
            async with Agent() as agent:
                assert isinstance(agent, Agent)
                assert FakeMCPClient.last_instance.entered is True

    async def test_tools_discovered_at_enter(self):
        tools = [_tool("calculator"), _tool("search")]
        with _patch_client(tools=tools):
            async with Agent() as agent:
                assert agent._tools == tools

    async def test_executor_wired_to_client_call_tool(self):
        with _patch_client(tools=[_tool()]):
            async with Agent() as agent:
                # bound methods compare equal when same instance + function
                assert agent._executor == FakeMCPClient.last_instance.call_tool

    async def test_server_path_passed_through_to_client(self):
        with _patch_client():
            async with Agent(server="custom/server.py"):
                assert FakeMCPClient.last_instance.server == "custom/server.py"

    async def test_default_server_used_when_unspecified(self):
        with _patch_client():
            async with Agent():
                assert FakeMCPClient.last_instance.server == "src/mcp_client/server.py"

    async def test_client_shut_down_on_normal_exit(self):
        with _patch_client(tools=[_tool()]):
            async with Agent():
                pass
        assert FakeMCPClient.last_instance.exited is True

    async def test_tools_discovered_exactly_once(self):
        with _patch_client(tools=[_tool()]):
            async with Agent():
                pass
        assert FakeMCPClient.last_instance.list_tools_calls == 1


# ---------------------------------------------------------------------------
# run() delegation to run_agent
# ---------------------------------------------------------------------------

class TestAgentRunDelegation:
    async def test_run_returns_run_agent_result(self):
        sentinel = object()
        with _patch_client(tools=[_tool()]), \
             patch("src.agent.agent.run_agent",
                   new=AsyncMock(return_value=sentinel)):
            async with Agent() as agent:
                assert await agent.run("q") is sentinel

    async def test_run_forwards_query_and_config(self):
        tools = [_tool("calculator")]
        provider = object()
        mock_run = AsyncMock(return_value=object())
        with _patch_client(tools=tools), \
             patch("src.agent.agent.run_agent", new=mock_run):
            async with Agent(system="be terse", max_turns=9,
                             provider=provider) as agent:
                await agent.run("what is 2+2?")

        kwargs = mock_run.await_args.kwargs
        assert kwargs["user_query"] == "what is 2+2?"
        assert kwargs["system"] == "be terse"
        assert kwargs["tools"] == tools
        assert kwargs["max_turns"] == 9
        assert kwargs["provider"] is provider
        # executor is the client's call_tool, discovered at enter
        assert kwargs["executor"] == FakeMCPClient.last_instance.call_tool

    async def test_run_forwards_defaults_when_unset(self):
        mock_run = AsyncMock(return_value=object())
        with _patch_client(tools=[_tool()]), \
             patch("src.agent.agent.run_agent", new=mock_run):
            async with Agent() as agent:
                await agent.run("q")

        kwargs = mock_run.await_args.kwargs
        assert kwargs["system"] is None
        assert kwargs["max_turns"] == 5     # documented default
        assert kwargs["provider"] is None

    async def test_run_callable_multiple_times_without_rediscovery(self):
        mock_run = AsyncMock(return_value=object())
        with _patch_client(tools=[_tool()]), \
             patch("src.agent.agent.run_agent", new=mock_run):
            async with Agent() as agent:
                await agent.run("first")
                await agent.run("second")
                await agent.run("third")

        assert mock_run.await_count == 3
        queries = [c.kwargs["user_query"] for c in mock_run.await_args_list]
        assert queries == ["first", "second", "third"]
        # tools discovered once at enter, not per run()
        assert FakeMCPClient.last_instance.list_tools_calls == 1


# ---------------------------------------------------------------------------
# Guard: run() outside the context manager
# ---------------------------------------------------------------------------

class TestAgentGuard:
    async def test_run_before_enter_raises_runtime_error(self):
        agent = Agent()  # never entered → no client, no subprocess
        with pytest.raises(RuntimeError, match="context manager"):
            await agent.run("q")

    async def test_run_after_exit_raises(self):
        """
        After the context exits, the client is cleared, so run() hits the same
        friendly RuntimeError as run()-before-enter — rather than failing deep
        inside a closed MCP session.
        """
        with _patch_client(tools=[_tool()]):
            async with Agent() as agent:
                pass
            assert FakeMCPClient.last_instance.exited is True
            with pytest.raises(RuntimeError, match="context manager"):
                await agent.run("q")


# ---------------------------------------------------------------------------
# Edge case: cleanup when setup fails — the orphaned-subprocess fix
# ---------------------------------------------------------------------------

class TestAgentEnterFailureCleanup:
    async def test_list_tools_failure_propagates(self):
        with _patch_client(list_tools_error=RuntimeError("discovery boom")):
            with pytest.raises(RuntimeError, match="discovery boom"):
                async with Agent():
                    pass

    async def test_client_torn_down_when_list_tools_fails(self):
        """
        The whole point of staging setup on AsyncExitStack: if list_tools()
        raises after the client has entered, the client is still exited —
        no orphaned server subprocess.
        """
        with _patch_client(list_tools_error=RuntimeError("discovery boom")):
            with pytest.raises(RuntimeError):
                async with Agent():
                    pass

        client = FakeMCPClient.last_instance
        assert client.entered is True   # subprocess was started
        assert client.exited is True    # ...and cleanly shut down

    async def test_no_client_entered_means_nothing_to_clean(self):
        """If MCPClient.__aenter__ itself fails, the failure still surfaces."""
        boom = RuntimeError("cannot spawn")

        class ExplodingClient(FakeMCPClient):
            async def __aenter__(self):
                self.entered = False
                raise boom

        with patch("src.agent.agent.MCPClient",
                   side_effect=lambda server: ExplodingClient(server)):
            with pytest.raises(RuntimeError, match="cannot spawn"):
                async with Agent():
                    pass


# ---------------------------------------------------------------------------
# Edge case: exception inside the `async with` body still cleans up
# ---------------------------------------------------------------------------

class TestAgentBodyException:
    async def test_body_exception_propagates(self):
        with _patch_client(tools=[_tool()]):
            with pytest.raises(ValueError, match="kaboom"):
                async with Agent():
                    raise ValueError("kaboom")

    async def test_body_exception_still_shuts_down_client(self):
        with _patch_client(tools=[_tool()]):
            with pytest.raises(ValueError):
                async with Agent():
                    raise ValueError("kaboom")
        assert FakeMCPClient.last_instance.exited is True


# ---------------------------------------------------------------------------
# Package export — Agent is reachable from the package root
# ---------------------------------------------------------------------------

class TestPackageExport:
    def test_agent_exported_from_src_agent(self):
        from src.agent import Agent as ExportedAgent
        assert ExportedAgent is Agent

    def test_run_agent_still_exported(self):
        from src.agent import run_agent  # noqa: F401  — sibling export intact
