import asyncio

from agents import Agent, ItemHelpers, Runner, trace
from dotenv import load_dotenv
load_dotenv()


"""
This example shows the parallelization pattern. We run the agent three times in parallel, and pick
the best result.
"""

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
)

translation_picker = Agent(
    name="translation_picker",
    model="gpt-4o",
    instructions="You pick the best Italian translation from the given options.",
)

def get_agent_model(agent_instance):
    """Try multiple ways to extract the model name"""
    # Common attribute names
    possible_attrs = ['model', 'llm', '_model', 'model_name', 'model_id']
    
    for attr in possible_attrs:
        if hasattr(agent_instance, attr):
            value = getattr(agent_instance, attr)
            if value:
                return value
    
    # Check config if exists
    if hasattr(agent_instance, 'config'):
        config = agent_instance.config
        if hasattr(config, 'get') and callable(config.get):
            return config.get('model')
    
    return "Unknown"

async def main():
    print(f"Italian_agent using model: {get_agent_model(italian_agent)}")
    print(f"Translation_picker using model: {get_agent_model(translation_picker)}")

    msg = input("Hi! Enter a message, and we'll translate it to Italian.\n\n")

    # Ensure the entire workflow is a single trace
    with trace("Parallel translation"):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(
                italian_agent,
                msg,
            ),
            Runner.run(
                italian_agent,
                msg,
            ),
            Runner.run(
                italian_agent,
                msg,
            ),
        )

        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
            ItemHelpers.text_message_outputs(res_3.new_items),
        ]

        translations = "\n\n".join(outputs)
        print(f"\n\nTranslations:\n\n{translations}")

        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
        )

    print("\n\n-----")

    print(f"Best translation: {best_translation.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
