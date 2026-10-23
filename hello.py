from dotenv import load_dotenv   # reads the .env file
from google import genai          # Google's Gemini library
import pandas as pd

load_dotenv()            # loads GEMINI_API_KEY into memory
client = genai.Client()  # finds the key on its own; no need to pass it

# 1. Test the LLM: send a question, get text back
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="In 2 lines, what is an AI agent?"
)
print(response.text)

# 2. Test the data
orders = pd.read_csv("data/olist_orders_dataset.csv")
print(orders.shape)   # (rows, columns)