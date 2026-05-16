from src.providers.base import Tool, ToolParameter


# The tool DEFINITION
CALCULATOR_TOOL = Tool(
    name="calculator",
    description=(
        "Performs basic arithmetic: addition, subtraction, multiplication, "
        "and division. Use this whenever the user asks for a calculation, "
        "even simple ones — never do math in your head."
    ),
    parameters=[
        ToolParameter(
            name="operation",
            type="string",
            description="One of: add, subtract, multiply, divide",
        ),
        ToolParameter(
            name="a",
            type="number",
            description="The first number",
        ),
        ToolParameter(
            name="b",
            type="number",
            description="The second number",
        ),
    ],
)


# The tool IMPLEMENTATION
def run_calculator(operation: str, a: float, b: float) -> str:
    """Execute the calculation and return result as a string."""
    ops = {
        "add":      lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide":   lambda: a / b if b != 0 else None,
    }

    if operation not in ops:
        return f"Error: unknown operation '{operation}'"

    result = ops[operation]()

    if result is None:
        return "Error: division by zero"

    return str(result)