import os
import pandas as pd

import subprocess   # runs code as a separate program
import sys          # tells us which Python is running (the .venv one)

DATA_DIR = "data"   # folder where the CSVs live


# ---------- TOOL 1 ----------
def list_tables():
    """Return the names of all tables (CSV files) available."""
    files = os.listdir(DATA_DIR)                     # every file in /data
    tables = [f.replace(".csv", "") for f in files   # drop the .csv ending
              if f.endswith(".csv")]                 # ignore non-CSV files
    return {"tables": tables}


# ---------- TOOL 2 ----------
def describe_table(table_name):
    """Return row count, column names + types, and 3 sample rows."""
    path = os.path.join(DATA_DIR, table_name + ".csv")

    # If the model asks for a table that doesn't exist, we don't crash.
    # We send back a helpful error so the model can fix its own mistake.
    if not os.path.exists(path):
        return {"error": f"No table named '{table_name}'. "
                         f"Call list_tables to see valid names."}

    df = pd.read_csv(path)
    return {
        "rows": len(df),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(3).astype(str).to_dict(orient="records"),
    }

# ---------- TOOL 3 ----------

# Words the agent's code is NOT allowed to contain.
# This blocks file access, system commands, and writing files.
BLOCKED = [
    "import os", "import sys", "import subprocess", "import shutil",
    "open(", "__import__", "eval(", "exec(",
    "remove(", "rmtree", "to_csv", "to_excel",
]

# Code we add in front of every run, so the agent doesn't have to.
PREAMBLE = '''
import pandas as pd
import numpy as np
pd.set_option("display.width", 200)

def load(table_name):
    return pd.read_csv("data/" + table_name + ".csv")
'''


def run_python(code):
    """Run the agent's pandas code in a separate process and return what it printed."""

    # 1. Safety check: refuse code containing blocked words
    for word in BLOCKED:
        if word in code:
            return {"error": f"Blocked: code contains '{word}'. "
                             f"Only use pandas/numpy with load() and print()."}

    # 2. Hide the API key from the agent's code
    safe_env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    safe_env["PYTHONIOENCODING"] = "utf-8"   # so Portuguese text in the data prints fine

    # 3. Run the code as a separate program, with a time limit
    try:
        result = subprocess.run(
            [sys.executable, "-c", PREAMBLE + "\n" + code],
            capture_output=True,   # collect what it prints instead of showing it
            text=True,
            encoding="utf-8",
            timeout=30,            # kill it after 30 seconds
            env=safe_env,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Code took over 30 seconds and was stopped. Simplify it."}

    # 4. If the code crashed, send back the last lines of the error
    if result.returncode != 0:
        last_lines = result.stderr.strip().splitlines()[-5:]
        return {"error": "\n".join(last_lines)}

    # 5. If it ran but printed nothing, tell the agent
    output = result.stdout.strip()
    if not output:
        return {"error": "Code ran but printed nothing. Use print() to show results."}

    # 6. Cut very long output so we don't flood the model
    if len(output) > 2000:
        output = output[:2000] + "\n...(output cut at 2000 characters)"

    return {"output": output}


# ---------- THE MENU WE SHOW THE MODEL ----------
# The model never sees our Python code. It only sees these descriptions.
# So the description is basically the instructions for when to use each tool.
TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "list_tables",
        "description": "Lists every data table available. "
                       "Call this first to see what data exists.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "describe_table",
        "description": "Shows row count, column names with data types, "
                       "and 3 sample rows for one table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Exact name from list_tables, "
                                   "e.g. olist_orders_dataset",
                }
            },
            "required": ["table_name"],
        },
    },
    {
        "type": "function",
        "name": "run_python",
        "description": "Runs Python (pandas/numpy) code to calculate answers from the data. "
                       "Load a table with load('table_name'), "
                       "e.g. orders = load('olist_orders_dataset'). "
                       "pd and np are already imported. "
                       "Each run starts fresh and remembers nothing from earlier runs, "
                       "so load every table you need each time. "
                       "Use print() to show results. Print small summaries, not whole tables.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Must use print() to show the result.",
                }
            },
            "required": ["code"],
        },
    },
]

# Maps the name the model asks for -> the real Python function to run
TOOL_FUNCTIONS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_python": run_python,
}