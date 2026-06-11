import sys

from lib.search_utils import Movie
from lib.inverted_index import InvertedIndex
from lib.text_utils import tokenize_text


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
