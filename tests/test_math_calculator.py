"""Tests for the math calculator tool."""

import pytest

from examples.tools.math_calculator import add, divide, evaluate_formula, multiply, subtract
from agents.tool_context import ToolContext


@pytest.mark.asyncio
async def test_add_tool():
    """Test the add function tool."""
    result = await add.on_invoke_tool(
        ToolContext(None, tool_name=add.name, tool_call_id="1", tool_arguments='{"a": 2, "b": 3}'),
        '{"a": 2, "b": 3}',
    )
    assert result == 5.0


@pytest.mark.asyncio
async def test_subtract_tool():
    """Test the subtract function tool."""
    result = await subtract.on_invoke_tool(
        ToolContext(
            None,
            tool_name=subtract.name,
            tool_call_id="1",
            tool_arguments='{"a": 10, "b": 3}',
        ),
        '{"a": 10, "b": 3}',
    )
    assert result == 7.0


@pytest.mark.asyncio
async def test_multiply_tool():
    """Test the multiply function tool."""
    result = await multiply.on_invoke_tool(
        ToolContext(
            None,
            tool_name=multiply.name,
            tool_call_id="1",
            tool_arguments='{"a": 4, "b": 5}',
        ),
        '{"a": 4, "b": 5}',
    )
    assert result == 20.0


@pytest.mark.asyncio
async def test_divide_tool():
    """Test the divide function tool."""
    result = await divide.on_invoke_tool(
        ToolContext(
            None, tool_name=divide.name, tool_call_id="1", tool_arguments='{"a": 20, "b": 4}'
        ),
        '{"a": 20, "b": 4}',
    )
    assert result == 5.0


@pytest.mark.asyncio
async def test_divide_by_zero():
    """Test that divide by zero raises an error."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        await divide.on_invoke_tool(
            ToolContext(
                None, tool_name=divide.name, tool_call_id="1", tool_arguments='{"a": 10, "b": 0}'
            ),
            '{"a": 10, "b": 0}',
        )


def test_evaluate_simple_formula():
    """Test evaluation of simple mathematical formulas."""
    assert evaluate_formula("2 + 3") == 5.0
    assert evaluate_formula("10 - 3") == 7.0
    assert evaluate_formula("4 * 5") == 20.0
    assert evaluate_formula("20 / 4") == 5.0


def test_evaluate_complex_formula():
    """Test evaluation of complex formulas with parentheses."""
    # From problem statement
    assert evaluate_formula("(2+3)*2 / 5") == 2.0
    
    # Additional test cases
    assert evaluate_formula("10 + 5 * 3") == 25.0
    assert evaluate_formula("(10 + 5) * 3") == 45.0
    assert evaluate_formula("100 - 50 * 2 + 10") == 10.0
    assert evaluate_formula("(100 - 50) * 2 + 10") == 110.0


def test_evaluate_with_decimals():
    """Test evaluation with decimal numbers."""
    assert evaluate_formula("2.5 + 3.5") == 6.0
    assert evaluate_formula("10.5 / 2") == 5.25
    assert evaluate_formula("3.5 * 2") == 7.0


def test_evaluate_invalid_formula():
    """Test that invalid formulas raise errors."""
    with pytest.raises(ValueError, match="Formula contains invalid characters"):
        evaluate_formula("2 + 3; import os")
    
    with pytest.raises(ValueError, match="Formula contains invalid characters"):
        evaluate_formula("2 + abc")
    
    with pytest.raises(ValueError, match="Error evaluating formula"):
        evaluate_formula("2 + + 3")


def test_evaluate_division_by_zero():
    """Test that division by zero in formulas raises an error."""
    with pytest.raises(ValueError, match="Error evaluating formula"):
        evaluate_formula("10 / 0")
