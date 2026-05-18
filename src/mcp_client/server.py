import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPTool, TextContent

from src.tools import execute_tool, ALL_TOOLS
from src.providers.base import ToolCall

logger = logging.getLogger(__name__)

# Create the MCP server instance
# The name here is what clients see in the handshake
app = Server("ai-research-agent-tools")


@app.list_tools()
async def list_tools() -> list[MCPTool]:
    """
    Respond to tools/list requests.
    Translates your existing Tool dataclasses into MCP's format.
    The client calls this once on startup to discover available tools.
    """
    mcp_tools = []
    for tool in ALL_TOOLS:
        mcp_tools.append(MCPTool(
            name=tool.name,
            description=tool.description,
            inputSchema={
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in tool.parameters
                },
                "required": [p.name for p in tool.parameters if p.required],
            }
        ))
    return mcp_tools


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Respond to tools/call requests.
    Delegates to existing execute_tool()
    Returns a list of TextContent blocks (MCP's result format).
    """
    logger.debug("[MCP Server] call_tool: %s(%s)", name, arguments)

    # The server is a thin protocol adapter — business logic stays in tools/.
    tool_call = ToolCall(
        id="mcp_client-call",      # id is only meaningful client-side for matching
        name=name,
        arguments=arguments,
    )

    result = execute_tool(tool_call)
    return [TextContent(type="text", text=result)]


async def main():
    """Entry point — runs the server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())