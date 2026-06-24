import os
from google import genai
from dotenv import load_dotenv
from lib.prompts import load_prompt

MODEL = "gemini-2.5-flash"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def individual_rerank(query: str, documents: dict) -> list[dict]:
    prompt = load_prompt("individual_rerank", query=query)

    results = []
    for doc in documents:
        _prompt = prompt.format(
            title=doc["title"],
            description=doc["description"],
        )
        response = client.models.generate_content(model=MODEL, contents=_prompt)
        results.append({**doc, "rerank_response": float(response.text or 0)})

    return sorted(results, key=lambda x: x["rerank_response"], reverse=True)
    