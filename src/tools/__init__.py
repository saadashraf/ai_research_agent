from src.tools.calculator import CALCULATOR_TOOL, run_calculator
from src.tools.file_reader import FILE_READER_TOOL, run_file_reader
from src.tools.web_search import WEB_SEARCH_TOOL, run_web_search
from src.providers.base import ToolCall

# All available tools — add new ones here as the project grows
ALL_TOOLS = [CALCULATOR_TOOL, FILE_READER_TOOL, WEB_SEARCH_TOOL]

# Maps tool name → the Python function that runs it
TOOL_REGISTRY = {
    "calculator": run_calculator,
    "read_file":  run_file_reader,
    "web_search": run_web_search,
}


def execute_tool(tool_call: ToolCall) -> str:
    """
    Given a ToolCall from the model, find and run the right function.
    Returns the result as a string to send back to Claude.
    """
    fn = TOOL_REGISTRY.get(tool_call.name)
    if not fn:
        return f"Error: no tool named '{tool_call.name}' is registered"
    return fn(**tool_call.arguments)