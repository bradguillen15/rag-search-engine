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
    
def batch_rerank(query: str, documents: list[dict]) -> list[dict]:
    prompt = load_prompt("batch_rerank", query=query)

    doc_list_str = []
    movie_template = "<movie_id>{doc_id}</movie_id> <movie_title>{title}</movie_title> - <movie_description>{description}</movie_description>"
    for doc in documents:
        doc_list_str.append(movie_template.format(doc_id=doc['doc_id'], title=doc['title'], description=doc['description']))

    _prompt = prompt.format(
        query=query,
        doc_list_str="\n".join(doc_list_str),
    )

    response = client.models.generate_content(model=MODEL, contents=_prompt)
    response_parsed = json.loads(response.text or "[]")

    doc_by_id = {doc["doc_id"]: doc for doc in documents}
    results = []
    for idx, doc_id in enumerate(response_parsed):
        results.append({**doc_by_id[doc_id], "rerank_response": float(len(response_parsed) - idx)})

    return results
