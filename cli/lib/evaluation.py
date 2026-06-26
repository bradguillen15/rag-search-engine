import json
from lib.search_utils import DATA_PATH
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch

def load_test_cases() -> list[dict]:
    with open(DATA_PATH / "golden_dataset.json", "r") as f:
        test_cases = json.load(f)["test_cases"]
    return test_cases

def evaluate(limit):
    print(f"k={limit}")
    test_cases = load_test_cases()
    movies = load_movies()

    hs = HybridSearch(movies)

    for test_case in test_cases:
        query = test_case["query"]
        expected_results = test_case["relevant_docs"]
        rrf_results = hs.rrf_search(query, k=60, limit=limit)
        relevant_count = 0
        for rrf_result in rrf_results:
            relevant_count += rrf_result["title"] in expected_results
        precision = relevant_count / limit

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join([result['title'] for result in rrf_results])}")
        print(f"  - Relevant: {', '.join(expected_results)}")