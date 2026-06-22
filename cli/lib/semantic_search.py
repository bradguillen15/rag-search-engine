import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_PATH
from lib.vector_utils import cosine_similarity


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = CACHE_PATH / "embeddings.npy"

    def build_embeddings(self, documents: list[str]):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(self.documents)

    def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_emb = self.generate_embedding(query)

        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            similarity = cosine_similarity(query_emb, doc_emb)
            similarities.append((similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sc, doc in similarities[:limit]:
            results.append({
                "score": sc,
                "title": doc["title"],
                "description": doc["description"],
            })
        return results
