import json
from dotenv import load_dotenv
from google import genai
from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

load_dotenv()
client = genai.Client(http_options={"timeout": 1200000})

MODEL = "gemini-3.8-flash"
MAX_STEPS = 10   # safety limit so the agent can't loop forever

SYSTEM_PROMPT = """You are a careful data analyst.
You have tools to explore an ecommerce dataset.
Always look at the data with tools before answering.
Never guess table or column names. Check them first.
Keep the final answer short and plain."""


def run_agent(question):
    # ---- First call: question + menu of tools ----
    interaction = client.interactions.create(
        model=MODEL,
        input=SYSTEM_PROMPT + "\n\nQuestion: " + question,
        tools=TOOL_DECLARATIONS,
    )

    # ---- The loop ----
    for step in range(MAX_STEPS):

        # Did the model ask for any tools this turn?
        calls = [s for s in interaction.steps if s.type == "function_call"]

        # No tool requests means the model is done. Return its answer.
        if not calls:
            return interaction.output_text

        # Run every tool the model asked for
        results = []
        for call in calls:
            print(f"  [step {step + 1}] model called: {call.name}({call.arguments})")

            func = TOOL_FUNCTIONS.get(call.name)
            if func is None:
                output = {"error": f"Unknown tool '{call.name}'"}
            else:
                output = func(**(call.arguments or {}))

            # Package the result so the model knows which request it answers
            results.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text",
                            "text": json.dumps(output, default=str)}],
            })

        # Send results back. previous_interaction_id = "continue the same chat"
        interaction = client.interactions.create(
            model=MODEL,
            input=results,
            tools=TOOL_DECLARATIONS,
            previous_interaction_id=interaction.id,
        )

    return "Stopped: hit the step limit without a final answer."


if __name__ == "__main__":
    q = input("Ask a question about the data: ")
    print("\nThinking...\n")
    print("\nANSWER:\n" + run_agent(q))