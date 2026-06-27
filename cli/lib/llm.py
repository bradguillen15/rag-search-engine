import os
import json
from google import genai
from dotenv import load_dotenv
from lib.prompts import load_prompt

MODEL = "gemini-2.5-flash"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def generate_content(prompt: str):
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text

def augmented_prompt(query: str, enhance: str):
    return generate_content(load_prompt(enhance, query=query))

def llm_judge(query: str, formatted_results: str):
    prompt = load_prompt("llm_judge", query=query, formatted_results=formatted_results)
    return json.loads(generate_content(prompt) or "[]")

def answer_question(query: str, docs: list[dict]) -> str:
    formatted_docs = "\n\n".join(
        f"Title: {doc['title']}\n{doc['description']}" for doc in docs
    )
    prompt = load_prompt("answer_question", query=query, docs=formatted_docs)
    return generate_content(prompt)
    