"""
Tests for MCPClient (src/mcp_client/client.py).

The phase/4 diff changed call_tool's signature from (name, arguments) to a
single ToolCall, so the MCP executor now honors the same contract as the local
execute_tool — the harness can call either with executor(tool_call). These
tests pin that contract plus the result-translation behavior, and the
list_tools() translation that turns MCP tool defs into our Tool dataclasses.

No real subprocess is spawned: a fake ClientSession is injected directly as
client._session, since call_tool / list_tools only ever touch the session.
Async tests run under pytest-asyncio (asyncio_mode = auto in pytest.ini).
"""

from unittest.mock import AsyncMock

from src.mcp_client.client import MCPClient
from src.providers.base import Tool, ToolCall, ToolParameter


# ---------------------------------------------------------------------------
# Fakes mirroring the bits of the MCP response objects the client touches
# ---------------------------------------------------------------------------

class TextBlock:
    """A content block with a .text attribute (TextContent-like)."""
    def __init__(self, text):
        self.text = text


class NonTextBlock:
    """A content block WITHOUT .text (e.g. an image) — must be skipped."""


class FakeToolResponse:
    def __init__(self, content, is_error=False):
        self.content = content
        self.isError = is_error


class FakeMcpTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class FakeListToolsResponse:
    def __init__(self, tools):
        self.tools = tools


def _client_with_session(session):
    client = MCPClient("dummy/server.py")
    client._session = session
    return client


def _tc(name="calculator", arguments=None):
    return ToolCall(id="toolu_1", name=name,
                    arguments=arguments if arguments is not None else {"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# call_tool — the changed signature
# ---------------------------------------------------------------------------

class TestCallToolContract:
    async def test_unpacks_tool_call_into_name_and_arguments(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse([TextBlock("ok")])
        client = _client_with_session(session)

        await client.call_tool(_tc(name="calculator", arguments={"a": 5, "b": 6}))

        session.call_tool.assert_awaited_once_with("calculator", {"a": 5, "b": 6})

    async def test_accepts_single_tool_call_argument(self):
        """The whole point of the diff: one ToolCall in, string out."""
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse([TextBlock("42")])
        client = _client_with_session(session)

        result = await client.call_tool(_tc())
        assert result == "42"


class TestCallToolSuccess:
    async def test_returns_single_text_block(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse([TextBlock("hello")])
        result = await _client_with_session(session).call_tool(_tc())
        assert result == "hello"

    async def test_joins_multiple_text_blocks_with_space(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse(
            [TextBlock("foo"), TextBlock("bar"), TextBlock("baz")]
        )
        result = await _client_with_session(session).call_tool(_tc())
        assert result == "foo bar baz"

    async def test_skips_blocks_without_text(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse(
            [TextBlock("keep"), NonTextBlock(), TextBlock("this")]
        )
        result = await _client_with_session(session).call_tool(_tc())
        assert result == "keep this"

    async def test_empty_content_returns_empty_string(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse([])
        result = await _client_with_session(session).call_tool(_tc())
        assert result == ""


class TestCallToolError:
    async def test_error_response_returned_as_string_not_raised(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse(
            [TextBlock("bad arguments")], is_error=True
        )
        result = await _client_with_session(session).call_tool(_tc())
        assert result == "Tool error: bad arguments"

    async def test_error_text_joined_from_blocks(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse(
            [TextBlock("line one"), TextBlock("line two")], is_error=True
        )
        result = await _client_with_session(session).call_tool(_tc())
        assert result == "Tool error: line one line two"

    async def test_error_is_a_string_so_model_can_reason(self):
        session = AsyncMock()
        session.call_tool.return_value = FakeToolResponse(
            [TextBlock("nope")], is_error=True
        )
        result = await _client_with_session(session).call_tool(_tc())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# list_tools — translation of MCP tool defs into Tool dataclasses + caching
# ---------------------------------------------------------------------------

class TestListToolsTranslation:
    def _session_with_tools(self, mcp_tools):
        session = AsyncMock()
        session.list_tools.return_value = FakeListToolsResponse(mcp_tools)
        return session

    async def test_translates_into_tool_dataclasses(self):
        session = self._session_with_tools([
            FakeMcpTool("calculator", "does math", {
                "properties": {
                    "operation": {"type": "string", "description": "op"},
                    "a": {"type": "number", "description": "first"},
                },
                "required": ["operation", "a"],
            })
        ])
        tools = await _client_with_session(session).list_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], Tool)
        assert tools[0].name == "calculator"
        assert tools[0].description == "does math"
        assert all(isinstance(p, ToolParameter) for p in tools[0].parameters)

    async def test_required_flag_reflects_required_list(self):
        session = self._session_with_tools([
            FakeMcpTool("t", "d", {
                "properties": {
                    "must": {"type": "string", "description": ""},
                    "maybe": {"type": "string", "description": ""},
                },
                "required": ["must"],
            })
        ])
        params = {p.name: p for p in (await _client_with_session(session).list_tools())[0].parameters}
        assert params["must"].required is True
        assert params["maybe"].required is False

    async def test_type_defaults_to_string_when_missing(self):
        session = self._session_with_tools([
            FakeMcpTool("t", "d", {
                "properties": {"x": {"description": "no type given"}},
                "required": [],
            })
        ])
        param = (await _client_with_session(session).list_tools())[0].parameters[0]
        assert param.type == "string"

    async def test_none_description_becomes_empty_string(self):
        session = self._session_with_tools([
            FakeMcpTool("t", None, {"properties": {}, "required": []})
        ])
        tools = await _client_with_session(session).list_tools()
        assert tools[0].description == ""

    async def test_handles_tool_with_no_parameters(self):
        session = self._session_with_tools([
            FakeMcpTool("ping", "no args", {})
        ])
        tools = await _client_with_session(session).list_tools()
        assert tools[0].parameters == []

    async def test_result_is_cached(self):
        session = self._session_with_tools([
            FakeMcpTool("t", "d", {"properties": {}, "required": []})
        ])
        client = _client_with_session(session)

        first = await client.list_tools()
        second = await client.list_tools()

        assert first is second
        session.list_tools.assert_awaited_once()  # server hit only once
