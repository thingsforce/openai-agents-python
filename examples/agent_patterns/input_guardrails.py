from __future__ import annotations
import asyncio
from pydantic import BaseModel
import re
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
from dotenv import load_dotenv, find_dotenv
load_dotenv()
# load_dotenv(find_dotenv(usecwd=True), override=True)

"""
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message or perform a local calculation.
"""

### Local math functions (copied from math_calculator.py)
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def evaluate_formula(formula: str) -> float:
    """
    Evaluate a mathematical formula using Python's eval.
    Only supports +, -, *, / and parentheses.
    Trailing '=' or whitespace is ignored.
    """
    formula = formula.strip()
    # Remove trailing '=' if present.
    if formula.endswith("="):
        formula = formula[:-1].strip()
    # Validate allowed characters.
    if not re.match(r'^[\d\+\-\*/\(\)\.\s]+$', formula):
        raise ValueError("Formula contains invalid characters")
    try:
        result = eval(formula)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating formula: {str(e)}")

### 1. An agent-based guardrail that is triggered if the user is asking to do math homework
class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions=(
        "Check if the user is asking you to do their math homework, "
        "or if the input contains any math operations, equations, or arithmetic expressions."
    ),
    output_type=MathHomeworkOutput,
)

@input_guardrail
async def math_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input
    is a math homework question or contains math expressions.
    """
    def contains_math_expression(text: str) -> bool:
        math_pattern = r"(\d+\s*[\+\-\*/]\s*\d+)|([a-zA-Z]\s*=\s*[\d\+\-\*/\s]+)"
        return bool(re.search(math_pattern, text))

    if isinstance(input, list):
        input_text = " ".join(str(item.get("content", "")) for item in input)
    else:
        input_text = str(input)

    if contains_math_expression(input_text):
        return GuardrailFunctionOutput(
            output_info={"reasoning": "Detected math expression in input.", "is_math_homework": True},
            tripwire_triggered=True,
        )

    result = await Runner.run(guardrail_agent, input, context=context.context)
    final_output = result.final_output_as(MathHomeworkOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_math_homework,
    )

def split_input_by_math_expressions(text: str):
    """
    Splits the input into normal text and math expressions.
    Returns a list of dicts with 'type' and 'content'.
    """
    # Updated pattern to match expressions with parentheses
    math_pattern = r"([\(\)\d\.\s\+\-\*/]+)"
    matches = list(re.finditer(math_pattern, text))
    result = []
    last_end = 0
    for match in matches:
        start, end = match.span()
        math_text = match.group().strip()
        if math_text and re.match(r'^[\(\)\d\.\s\+\-\*/]+$', math_text):
            if start > last_end:
                normal_text = text[last_end:start].strip()
                if normal_text:
                    result.append({"type": "normal", "content": normal_text})
            result.append({"type": "math", "content": math_text})
            last_end = end
    if last_end < len(text):
        normal_text = text[last_end:].strip()
        if normal_text:
            result.append({"type": "normal", "content": normal_text})
    return result

async def main():
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        input_guardrails=[math_guardrail],
    )

    while True:
        user_input = input("Enter a message: ")
        if not user_input:
            print("No more input. exiting loop.")
            break

        input_parts = split_input_by_math_expressions(user_input)
        for part in input_parts:
            print(f"Processing part: {part['content']} (type: {part['type']})")
            single_input_data = [
                {
                    "role": "user",
                    "content": part["content"],
                    "type": "message",
                }
            ]
            try:
                result = await Runner.run(agent, single_input_data)
                print(result.final_output)
            except InputGuardrailTripwireTriggered:
                if part["type"] == "math":
                    try:
                        calc_result = evaluate_formula(part["content"])
                        print(f"Local calculation result: {calc_result}")
                    except Exception as e:
                        print(f"Local calculation error: {e}")
                else:
                    message = "Sorry, I can't help you with your math homework."
                    print(message)

if __name__ == "__main__":
    asyncio.run(main())