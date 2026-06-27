from lib.llm import answer_question
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def rag(query: str):
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    rag_results = answer_question(query, rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['title']}")

    print("RAG Response:")
    print(rag_results)