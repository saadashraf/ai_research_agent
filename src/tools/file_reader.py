import os
from src.providers.base import Tool, ToolParameter


FILE_READER_TOOL = Tool(
    name="read_file",
    description=(
        "Reads the contents of a local file and returns it as text. "
        "Use this when the user asks you to look at, summarise, or "
        "answer questions about a file on disk."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Relative or absolute path to the file to read",
        ),
    ],
)


def run_file_reader(path: str) -> str:
    """Read a file and return its contents, or a clear error message."""
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, "r", encoding="utf-8") as f:
            contents = f.read()
        return contents if contents.strip() else "(file is empty)"
    except FileNotFoundError:
        return f"Error: file not found at '{path}'"
    except PermissionError:
        return f"Error: no read permission for '{path}'"
    except Exception as e:
        return f"Error reading file: {e}"