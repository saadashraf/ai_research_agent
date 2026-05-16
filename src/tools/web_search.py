from src.providers.base import Tool, ToolParameter


# The tool DEFINITION
WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Searches the web for information on a given query. Use this when "
        "the user asks about current events, recent information, or anything "
        "that requires up-to-date knowledge beyond the model's training data."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query to look up on the web",
        ),
    ],
)


# The tool IMPLEMENTATION
def run_web_search(query: str) -> str:
    """Execute the web search and return results as a string."""
    return f"Search results for: {query}"
