import json
from dotenv import load_dotenv
from google import genai
from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS
import time
import threading

load_dotenv()
client = genai.Client(http_options={"timeout": 12000000})

def call_model(**kwargs):
    """Call Gemini with retries, and show how long we've been waiting."""
    for attempt in range(4):
        t0 = time.perf_counter()          # stopwatch start for this call
        stop = threading.Event()          # a signal to switch the ticker off

        # Runs in the background: prints a message every 5 seconds
        def ticker():
            while not stop.wait(5):       # wait 5s; keep going until stop is set
                print(f"    ...still waiting for Gemini "
                      f"({time.perf_counter() - t0:.0f}s)")

        threading.Thread(target=ticker, daemon=True).start()

        try:
            reply = client.interactions.create(**kwargs)
        except Exception as e:
            stop.set()                    # stop the ticker before handling the error
            msg = str(e)
            busy = "503" in msg or "429" in msg or "high demand" in msg
            if not busy or attempt == 3:
                raise
            wait = 5 * (2 ** attempt)
            print(f"    (Gemini busy, retrying in {wait}s...)")
            time.sleep(wait)
            continue                      # go back to the top and try again

        stop.set()                        # got a reply: stop the ticker
        print(f"    (Gemini replied in {time.perf_counter() - t0:.1f}s)")
        return reply

MODEL = "gemini-3.8-flash"
MAX_STEPS = 10   # safety limit so the agent can't loop forever

SYSTEM_PROMPT = """You are a careful data analyst working with an ecommerce dataset.

How to work:
1. Use list_tables and describe_table to find the right tables and columns.
2. Use run_python to calculate the answer. Never do maths in your head.
3. If code fails, read the error, fix the code and try again.

Rules:
- Never guess table or column names. Check them first.
- State any assumption you made (for example, only delivered orders counted).
- Final answer: the number first, then one or two lines on how you got it."""


def run_agent(question):
    start = time.perf_counter()

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
            if call.name == "run_python":
                print(f"\n  [step {step + 1} | {time.perf_counter() - start:.0f}s] running code:\n{call.arguments.get('code')}\n")
            else:
                print(f"  [step {step + 1} | {time.perf_counter() - start:.0f}s] model called: {call.name}({call.arguments})")

            func = TOOL_FUNCTIONS.get(call.name)
            if func is None:
                output = {"error": f"Unknown tool '{call.name}'"}
            else:
                output = func(**(call.arguments or {}))
            if "error" in output:
                print(f"    -> error: {output['error'][:200]}")

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
    start = time.perf_counter()
    answer = run_agent(q)
    print("\nANSWER:\n" + answer)
    print(f"\n(Total time: {time.perf_counter() - start:.1f}s)")