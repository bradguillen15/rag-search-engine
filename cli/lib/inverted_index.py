import os
import math
import pickle
from collections import defaultdict, Counter

from lib.search_utils import load_movies, CACHE_PATH, BM25_K1, BM25_B
from lib.text_utils import tokenize_text, tokenize_term


class InvertedIndex:
    def __init__(self):
        self.index_path = CACHE_PATH / "index.pkl"
        self.index = defaultdict(set)

        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.docmap = {}

        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.term_frequencies = defaultdict(Counter)

        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        
        if len(lengths) == 0:
            return 0.0

        return sum(lengths) / len(lengths)

    def get_documents(self, term: str) -> list[int]:
        return list(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_term(term)
        return self.term_frequencies[doc_id][token]

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        if tf == 0:
            return 0.0

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_idf(self, term: str) -> float:
        token = tokenize_term(term)

        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_term(term)
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(token))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        query_tokens = tokenize_text(query)

        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
    
        sorted_scores = sorted(
            scores.items(), 
            key=lambda x: x[1],
            reverse=True
        )

        results = sorted_scores[:limit]
        formatted_results = []

        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            formatted_results.append({
                "doc_id": doc_id,
                "title": title,
                "score": score
            })

        return formatted_results

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.docmap_path.exists():
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_path}")
        if not self.term_frequencies_path.exists():
            raise FileNotFoundError(f"Term frequencies file not found: {self.term_frequencies_path}")
        if not self.doc_lengths_path.exists():
            raise FileNotFoundError(f"Doc lengths file not found: {self.doc_lengths_path}")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
