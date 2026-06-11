import sys

from lib.search_utils import Movie, BM25_K1, BM25_B
from lib.inverted_index import InvertedIndex
from lib.text_utils import tokenize_text


def bm25_search_command(query: str, limit: int = 5) -> None:
    index = InvertedIndex()
    index.load()
    results = index.bm25_search(query, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. ({result['doc_id']}) {result['title']} - Score: {result['score']:.2f}")

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> None:
    index = InvertedIndex()
    index.load()
    bm25_tf = index.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")

def bm25_idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()
    bm25idf = index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def tf_idf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    index.load()
    tf_idf = index.get_tf_idf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def tf(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    index.load()
    result = index.get_tf(doc_id, term)
    print(result)


def idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()
    idf = index.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, n_results: int = 5) -> list[Movie]:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Index not found. Run 'build' first to create the index.")
        sys.exit(1)

    matching_doc_ids: set[int] = set()
    for qt in tokenize_text(query):
        matching_doc_ids.update(index.get_documents(qt))

    result: list[Movie] = []
    for doc_id in sorted(matching_doc_ids):
        result.append(index.docmap[doc_id])
        if len(result) == n_results:
            break

    return result
