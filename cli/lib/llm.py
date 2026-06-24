import os
from google import genai
from dotenv import load_dotenv
from lib.prompts import get_spelling_prompt, get_rewriting_prompt

MODEL = "gemini-2.5-flash"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def generate_content(prompt: str):
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text

def correct_spelling(query: str):
    return generate_content(get_spelling_prompt(query))

def rewrite_query(query: str):
    return generate_content(get_rewriting_prompt(query))