from lib.llm import answer_question, answer_detailed_question, summarize_results, citations
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def doc_answer_question(query: str):
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    rag_results = answer_question(query, rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['title']}")

    print("RAG Response:")
    print(rag_results)

def doc_summarization(query, limit=5):
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    rag_results = summarize_results(query, rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['title']}")

    print("LLM Summary:")
    print(rag_results)

def doc_citations(query, limit=5):
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    rag_results = citations(query, rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['title']}")

    print("LLM Answer:")
    print(rag_results)

def doc_answer_detailed_question(query, limit=5):
    hs = HybridSearch(load_movies())
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    rag_results = answer_detailed_question(query, rrf_results)

    print("Search Results:")
    for result in rrf_results:
        print(f"- {result['title']}")

    print("Answer:")
    print(rag_results)
