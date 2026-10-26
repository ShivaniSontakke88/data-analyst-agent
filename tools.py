import os
import pandas as pd

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
]

# Maps the name the model asks for -> the real Python function to run
TOOL_FUNCTIONS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
}