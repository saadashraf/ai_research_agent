"""
Tests for the tools package: calculator, file_reader, web_search,
and the ALL_TOOLS / TOOL_REGISTRY / execute_tool dispatcher.
"""

import pytest

from src.providers.base import Tool, ToolCall
from src.tools import ALL_TOOLS, TOOL_REGISTRY, execute_tool
from src.tools.calculator import CALCULATOR_TOOL, run_calculator
from src.tools.file_reader import FILE_READER_TOOL, run_file_reader
from src.tools.web_search import WEB_SEARCH_TOOL, run_web_search


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

class TestCalculatorDefinition:
    def test_is_tool_instance(self):
        assert isinstance(CALCULATOR_TOOL, Tool)

    def test_name(self):
        assert CALCULATOR_TOOL.name == "calculator"

    def test_has_three_parameters(self):
        names = {p.name for p in CALCULATOR_TOOL.parameters}
        assert names == {"operation", "a", "b"}


class TestRunCalculator:
    def test_add(self):
        assert run_calculator("add", 2, 3) == "5"

    def test_subtract(self):
        assert run_calculator("subtract", 10, 4) == "6"

    def test_multiply(self):
        assert run_calculator("multiply", 6, 7) == "42"

    def test_divide(self):
        assert run_calculator("divide", 10, 2) == "5.0"

    def test_divide_by_zero(self):
        assert run_calculator("divide", 5, 0) == "Error: division by zero"

    def test_unknown_operation(self):
        result = run_calculator("modulo", 5, 2)
        assert "unknown operation" in result and "modulo" in result

    def test_negative_numbers(self):
        assert run_calculator("add", -3, -2) == "-5"

    def test_floats(self):
        assert run_calculator("add", 0.1, 0.2).startswith("0.3")


# ---------------------------------------------------------------------------
# file_reader
# ---------------------------------------------------------------------------

class TestFileReaderDefinition:
    def test_is_tool_instance(self):
        assert isinstance(FILE_READER_TOOL, Tool)

    def test_name(self):
        assert FILE_READER_TOOL.name == "read_file"

    def test_has_path_parameter(self):
        names = {p.name for p in FILE_READER_TOOL.parameters}
        assert names == {"path"}


class TestRunFileReader:
    def test_reads_file_contents(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world")
        assert run_file_reader(str(f)) == "hello world"

    def test_empty_file_returns_placeholder(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert run_file_reader(str(f)) == "(file is empty)"

    def test_whitespace_only_file_returns_placeholder(self, tmp_path):
        f = tmp_path / "whitespace.txt"
        f.write_text("   \n\t  ")
        assert run_file_reader(str(f)) == "(file is empty)"

    def test_missing_file(self, tmp_path):
        missing = tmp_path / "nope.txt"
        result = run_file_reader(str(missing))
        assert result.startswith("Error: file not found")
        assert str(missing) in result

    def test_permission_error(self, tmp_path, monkeypatch):
        def deny(*args, **kwargs):
            raise PermissionError()
        monkeypatch.setattr("builtins.open", deny)
        result = run_file_reader("anything.txt")
        assert result.startswith("Error: no read permission")

    def test_unexpected_error_caught(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")
        monkeypatch.setattr("builtins.open", boom)
        result = run_file_reader("x.txt")
        assert result.startswith("Error reading file:")
        assert "disk on fire" in result


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearchDefinition:
    def test_is_tool_instance(self):
        assert isinstance(WEB_SEARCH_TOOL, Tool)

    def test_name(self):
        assert WEB_SEARCH_TOOL.name == "web_search"

    def test_has_query_parameter(self):
        names = {p.name for p in WEB_SEARCH_TOOL.parameters}
        assert names == {"query"}

    def test_query_param_is_string_type(self):
        query = next(p for p in WEB_SEARCH_TOOL.parameters if p.name == "query")
        assert query.type == "string"

    def test_query_param_is_required(self):
        query = next(p for p in WEB_SEARCH_TOOL.parameters if p.name == "query")
        assert query.required is True

    def test_description_is_non_empty(self):
        assert WEB_SEARCH_TOOL.description.strip() != ""


class TestRunWebSearch:
    def test_returns_placeholder_with_query(self):
        assert run_web_search("python") == "Search results for: python"

    def test_handles_multiword_query(self):
        result = run_web_search("how to brew coffee")
        assert result == "Search results for: how to brew coffee"

    def test_handles_empty_string(self):
        assert run_web_search("") == "Search results for: "

    def test_handles_special_characters(self):
        assert run_web_search("a & b !?") == "Search results for: a & b !?"

    def test_returns_string(self):
        assert isinstance(run_web_search("x"), str)


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_tools_contains_three_tools(self):
        assert len(ALL_TOOLS) == 3

    def test_all_tools_includes_web_search(self):
        assert WEB_SEARCH_TOOL in ALL_TOOLS

    def test_all_tools_includes_calculator(self):
        assert CALCULATOR_TOOL in ALL_TOOLS

    def test_all_tools_includes_file_reader(self):
        assert FILE_READER_TOOL in ALL_TOOLS

    def test_registry_has_web_search(self):
        assert TOOL_REGISTRY["web_search"] is run_web_search

    def test_registry_has_calculator(self):
        assert TOOL_REGISTRY["calculator"] is run_calculator

    def test_registry_has_file_reader(self):
        assert TOOL_REGISTRY["read_file"] is run_file_reader

    def test_registry_keys_match_tool_names(self):
        tool_names = {t.name for t in ALL_TOOLS}
        assert set(TOOL_REGISTRY.keys()) == tool_names


class TestExecuteTool:
    def test_dispatches_to_web_search(self):
        call = ToolCall(id="t1", name="web_search", arguments={"query": "cats"})
        assert execute_tool(call) == "Search results for: cats"

    def test_dispatches_to_calculator(self):
        call = ToolCall(id="t2", name="calculator",
                        arguments={"operation": "add", "a": 1, "b": 2})
        assert execute_tool(call) == "3"

    def test_unknown_tool_returns_error(self):
        call = ToolCall(id="t3", name="nonexistent", arguments={})
        result = execute_tool(call)
        assert "no tool named" in result and "nonexistent" in result
