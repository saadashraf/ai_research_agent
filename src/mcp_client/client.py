import asyncio
import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.providers.base import Tool, ToolParameter

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Connects to an MCP server over stdio and exposes two operations:
      - list_tools()  → your Tool dataclasses (same shape as Phase 3)
      - call_tool()   → string result (same shape as execute_tool)

    The harness calls these instead of execute_tool().
    Everything MCP-specific is contained here.
    """

    def __init__(self, server_script: str):
        """
        server_script: path to the MCP server entry point.
        e.g. "src/mcp_client/server.py"
        """
        self._server_script = server_script
        self._session: ClientSession | None = None
        self._tools: list[Tool] | None = None  # cached after first list_tools

    async def __aenter__(self):
        """Starts the server subprocess and runs the MCP handshake."""

        project_root = next(
            p for p in Path(__file__).parents
            if (p / "requirements.txt").exists()
        )
        server_script_abs = project_root / self._server_script

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script_abs)],
            env={**os.environ, "PYTHONPATH": str(project_root)},
        )

        # stdio_client is a context manager that:
        # 1. Spawns the server subprocess
        # 2. Connects stdin/stdout pipes
        # 3. Handles framing (Content-Length headers)
        self._cm = stdio_client(server_params)
        read, write = await self._cm.__aenter__()

        # ClientSession handles:
        # 1. The initialize handshake
        # 2. The initialized notification
        # 3. id matching on responses
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        logger.debug("[MCPClient] Connected to server: %s", self._server_script)
        return self

    async def __aexit__(self, *args):
        """Shuts down the session and the server subprocess cleanly."""
        if self._session:
            await self._session.__aexit__(*args)
        await self._cm.__aexit__(*args)

    async def list_tools(self) -> list[Tool]:
        """
        Fetch available tools from the server.
        Translates MCP tool definitions back into your Tool dataclasses
        so the harness never sees MCP types.
        Result is cached — server doesn't change tools mid-session.
        """
        if self._tools is not None:
            return self._tools

        response = await self._session.list_tools()

        tools = []
        for mcp_tool in response.tools:
            props = mcp_tool.inputSchema.get("properties", {})
            required = mcp_tool.inputSchema.get("required", [])

            parameters = [
                ToolParameter(
                    name=param_name,
                    type=prop.get("type", "string"),
                    description=prop.get("description", ""),
                    required=param_name in required,
                )
                for param_name, prop in props.items()
            ]

            tools.append(Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                parameters=parameters,
            ))

        self._tools = tools
        return tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool on the server and return the result as a string.
        Same return type as execute_tool() — harness needs no changes.
        Distinguishes tool errors from transport errors explicitly.
        """
        logger.debug("[MCPClient] call_tool: %s(%s)", name, arguments)

        response = await self._session.call_tool(name, arguments)

        # Tool error — the tool ran but reported failure
        # Pass it back as a string so the model can reason about it
        if response.isError:
            error_text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            logger.warning("[MCPClient] Tool error: %s", error_text)
            return f"Tool error: {error_text}"

        # Success — join all text content blocks into one string
        return " ".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )