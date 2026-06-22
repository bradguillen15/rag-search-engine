import json

import numpy as np

from lib.chunking import semantic_chunking
from lib.search_utils import CACHE_PATH
from lib.semantic_search import SemanticSearch
from lib.vector_utils import cosine_similarity

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = CACHE_PATH / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            if doc['description'].strip() == "":
                continue
            _chunks = semantic_chunking(
                doc['description'],
                overlap=1,
                max_chunk_size=4
            )
            all_chunks.extend(_chunks)
            for chunk_idx in range(len(_chunks)):
                chunk_metadata.append({
                    'movie_id': movie_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(_chunks),
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(chunk_metadata),
            }, f, indent=4)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_emb = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = {}
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata[idx]
            midx, cidx = metadata['movie_id'], metadata['chunk_idx']
            sim = cosine_similarity(query_emb, chunk_embedding)
            chunk_scores.append({
                'movie_id': midx,
                'chunk_id': cidx,
                'score': sim,
            })
            movie_scores[midx] = max(movie_scores.get(midx, 0), sim) 
            movie_scores_sorted = sorted(
                movie_scores.items(), 
                key=lambda x: x[1], reverse=True
            )
            results = []
            for midx, score in movie_scores_sorted[:limit]:
                doc = self.documents[midx]
                results.append({
                    "id": doc['id'],
                    "title": doc['title'],
                    "document": doc['description'][:100],
                    "score": round(score, 4),
                    "metadata": {}
                })
        return results


