import os
import math
import pickle
from collections import defaultdict, Counter

from lib.search_utils import load_movies, CACHE_PATH
from lib.text_utils import tokenize_text, tokenize_term


class InvertedIndex:
    def __init__(self):
        self.index_path = CACHE_PATH / "index.pkl"
        self.index = defaultdict(set)

        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.docmap = {}

        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        return list(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_term(term)
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        token = tokenize_term(term)

        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.docmap_path.exists():
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_path}")
        if not self.term_frequencies_path.exists():
            raise FileNotFoundError(f"Term frequencies file not found: {self.term_frequencies_path}")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
