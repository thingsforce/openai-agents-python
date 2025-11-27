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
from dotenv import load_dotenv
load_dotenv()

"""
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.
"""


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
# async def math_guardrail(
#     context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
# ) -> GuardrailFunctionOutput:
#     """This is an input guardrail function, which happens to call an agent to check if the input
#     is a math homework question.
#     """
#     result = await Runner.run(guardrail_agent, input, context=context.context)
#     final_output = result.final_output_as(MathHomeworkOutput)

#     return GuardrailFunctionOutput(
#         output_info=final_output,
#         tripwire_triggered=final_output.is_math_homework,
#     )

async def math_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input
    is a math homework question or contains math expressions.
    """
    # Pre-processing: detect math expressions using regex.
    def contains_math_expression(text: str) -> bool:
        # Matches simple arithmetic expressions like "2 + 2", "x = 3 * 7", etc.
        math_pattern = r"(\d+\s*[\+\-\*/]\s*\d+)|([a-zA-Z]\s*=\s*[\d\+\-\*/\s]+)"
        return bool(re.search(math_pattern, text))

    # Normalize input to string for checking.
    if isinstance(input, list):
        input_text = " ".join(str(item.get("content", "")) for item in input)
    else:
        input_text = str(input)

    if contains_math_expression(input_text):
        # Tripwire triggered if math expression detected.
        return GuardrailFunctionOutput(
            output_info={"reasoning": "Detected math expression in input.", "is_math_homework": True},
            tripwire_triggered=True,
        )

    # Otherwise, call the agent as before.
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
    # Pattern matches arithmetic expressions and equations.
    math_pattern = r"(\d+\s*[\+\-\*/]\s*\d+|[a-zA-Z]\s*=\s*[\d\+\-\*/\s]+)"
    matches = list(re.finditer(math_pattern, text))
    result = []
    last_end = 0
    for match in matches:
        start, end = match.span()
        # Add normal text before math expression.
        if start > last_end:
            normal_text = text[last_end:start].strip()
            if normal_text:
                result.append({"type": "normal", "content": normal_text})
        # Add math expression.
        math_text = match.group().strip()
        if math_text:
            result.append({"type": "math", "content": math_text})
        last_end = end
    # Add any remaining normal text.
    if last_end < len(text):
        normal_text = text[last_end:].strip()
        if normal_text:
            result.append({"type": "normal", "content": normal_text})
    return result


### 2. The run loop
async def main():
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        input_guardrails=[math_guardrail],
    )

    input_data: list[TResponseInputItem] = []

    while True:

        user_input = input("Enter a message: ")
        if not  user_input:
            print("No more input. exiting loop.")
            break

        # Preprocess user_input to split into normal and math parts.
        input_parts = split_input_by_math_expressions(user_input)
        for part in input_parts:
            print(f"Processing part: {part['content']} (type: {part['type']})") # Debugging output
            # input_data.append( ... ) # Original code
            single_input_data = [
                {
                    "role": "user",
                    "content": part["content"],
                    # "type": part["type"],  # can't be used by agent SDK.
                    "type": "message",  # Optional: for debugging or downstream logic.
                }
            ]
            try:
                # result = await Runner.run(agent, input_data)
                result = await Runner.run(agent, single_input_data)
                print(result.final_output)
                input_data = result.to_input_list()
            except InputGuardrailTripwireTriggered:
                message = "Sorry, I can't help you with your math homework."
                print(message)
                input_data.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "type": "message",  # Optional: for debugging or downstream logic.  
                    }
                )
        input_data = []

    # print("program ended.")

    # Sample run:
    # Enter a message: What's the capital of California?
    # The capital of California is Sacramento.
    # Enter a message: Can you help me solve for x: 2x + 5 = 11
    # Sorry, I can't help you with your math homework.


if __name__ == "__main__":
    asyncio.run(main())
