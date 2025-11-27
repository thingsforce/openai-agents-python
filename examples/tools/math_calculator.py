"""
Math Formula Calculator using the Agents SDK.

This example demonstrates a calculator that can parse and evaluate
mathematical formulas like "(2+3)*2 / 5" using simple math functions.
"""

import asyncio
import re
from typing import Annotated

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
load_dotenv()


@function_tool
def add(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers together."""
    return a + b


@function_tool
def subtract(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Subtract the second number from the first."""
    return a - b


@function_tool
def multiply(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Multiply two numbers together."""
    return a * b


@function_tool
def divide(
    a: Annotated[float, "Numerator"],
    b: Annotated[float, "Denominator"],
) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def evaluate_formula(formula: str) -> float:
    """
    Evaluate a mathematical formula using Python's eval.
    This is a simple implementation for demonstration purposes.
    
    Args:
        formula: Mathematical formula as a string (e.g., "(2+3)*2 / 5")
        
    Returns:
        The result of the evaluation
        
    Raises:
        ValueError: If the formula contains invalid characters
    """
    # Remove whitespace
    formula = formula.strip()
    
    # Validate that formula only contains safe characters
    # Allow: digits, operators, parentheses, decimal point
    # This regex prevents all forms of code injection by only allowing mathematical symbols
    if not re.match(r'^[\d\+\-\*/\(\)\.\s]+$', formula):
        raise ValueError("Formula contains invalid characters")
    
    # Safe to use eval() here because input is validated to only contain mathematical operators
    # Note: ast.literal_eval() cannot evaluate mathematical expressions
    try:
        result = eval(formula)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating formula: {str(e)}")


# Create calculator agent with basic math tools
calculator_agent = Agent(
    name="Calculator Agent",
    instructions=(
        "You are a helpful calculator assistant. "
        "When given a mathematical formula, break it down and use the available math functions "
        "(add, subtract, multiply, divide) to compute the result step by step. "
        "Show your work and explain each step."
    ),
    tools=[add, subtract, multiply, divide],
)


async def main():
    """Run the calculator agent with example formulas."""
    
    # Example 1: Simple formula from problem statement
    print("Example 1: (2+3)*2 / 5")
    print("-" * 50)
    result = await Runner.run(
        calculator_agent,
        input="Calculate this formula step by step: (2+3)*2 / 5"
    )
    print(result.final_output)
    print()
    
    # Example 2: More complex formula
    print("\nExample 2: 10 + 5 * 3 - 8 / 4")
    print("-" * 50)
    result = await Runner.run(
        calculator_agent,
        input="Calculate this formula step by step: 10 + 5 * 3 - 8 / 4"
    )
    print(result.final_output)
    print()
    
    # Example 3: Using the direct evaluator
    print("\nDirect evaluation (without agent):")
    print("-" * 50)
    formulas = [
        "(2+3)*2 / 5",
        "10 + 5 * 3 - 8 / 4",
        "(100 - 50) * 2 + 10",
    ]
    for formula in formulas:
        result = evaluate_formula(formula)
        print(f"{formula} = {result}")


if __name__ == "__main__":
    asyncio.run(main())
