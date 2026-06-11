import os
import sys
import math
import pickle
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from lib.search_utils import Movie, load_movies, load_stop_words, CACHE_PATH

stemmer = PorterStemmer()

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

def clean_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def tokenize_text(text: str) -> list[str]:
    cleaned_text = clean_text(text)
    stop_words = load_stop_words()

    tokens: list[str] = []
    for token in cleaned_text.split():
        if token and token not in stop_words:
            tokens.append(stemmer.stem(token))
            
    return tokens

def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected 1 token, got {len(tokens)}")
    return tokens[0]

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
    