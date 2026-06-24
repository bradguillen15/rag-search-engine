import os
from lib.search_utils import load_movies
from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save() 

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        results = self.idx.bm25_search(query, limit)
        for result in results:
            result["description"] = self.idx.docmap[result["doc_id"]]["description"]
        return results

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bn25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = combine_search_results(bn25_results, sem_results, alpha)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bn25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = rrf_combine_search_results(bn25_results, sem_results, k)
        return combined_results[:limit]

def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
    hs = HybridSearch(load_movies())
    results = hs.weighted_search(query, alpha, limit)
    for idx, result in enumerate(results):
        print(f"{idx + 1}. {result['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['sem_score']:.3f}")
        print(f"  {result['description'][:100]}...")
    return results

def rrf_search(query: str, k: int = 60, limit: int = 5) -> list[dict]:
    hs = HybridSearch(load_movies())
    results = hs.rrf_search(query, k, limit)
    for idx, result in enumerate(results):
        print(f"{idx + 1}. {result['title']}")
        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['sem_rank']}")
        print(f"  {result['description'][:100]}...")
    return results


def hybrid_score(bn25_score: float, sem_score: float, alpha: float) -> float:
    return alpha * bn25_score + (1 - alpha) * sem_score

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = [r['score'] for r in results]
    norm_scores = normalize_scores(scores)
    for idx, result in enumerate(results):
        result['normalized_score'] = norm_scores[idx]
    return results

def rrf_score(rank: int, k: int) -> float:
    return 1 / (rank + k)

def rrf_final_score(r1: int, r2: int, k: int) -> int:
    score = 0
    if r1:
        score += rrf_score(r1, k)
    if r2:
        score += rrf_score(r2, k)
    return score

def rrf_combine_search_results(bm25_results: list[dict], sem_results: list[dict], k: int) -> list[dict]:
    scores = {}
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result['doc_id']
        scores[doc_id] = {
            "doc_id": doc_id,
            "bm25_rank": rank,
            "bm25_score": rrf_score(rank, k),
            "sem_rank": None,
            "sem_score": None,
            "title": result['title'],
            "description": result['description'],
        }
    for rank, result in enumerate(sem_results, start=1):
        doc_id = result.get('doc_id', result['id'])
        if doc_id not in scores:
            scores[doc_id] = {
                "doc_id": doc_id,
                "bm25_rank": None,
                "bm25_score": None,
                "sem_rank": rank,
                "sem_score": rrf_score(rank, k),
                "title": result['title'],
                "description": result.get('description', result.get('document', '')),
            }
        else:
            scores[doc_id]['sem_rank'] = rank
            scores[doc_id]['sem_score'] = rrf_score(rank, k)
    
    for doc_id in scores.keys():
        scores[doc_id]['rrf_score'] = rrf_final_score(
            scores[doc_id]['bm25_rank'],
            scores[doc_id]['sem_rank'],
            k
        )
    
    return sorted(list(scores.values()), key=lambda x: x['rrf_score'], reverse=True)

def combine_search_results(bm25_results: list[dict], sem_results: list[dict], alpha: float = 0.5) -> list[dict]:
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results([
        {"doc_id": r["id"], "title": r["title"], "description": r["document"], "score": r["score"]}
        for r in sem_results
    ])

    combined_nom = {}
    for nom in bm25_norm:
        doc_id = nom['doc_id']
        combined_nom[doc_id] = {
            "doc_id": doc_id,
            "bm25_score": nom['normalized_score'],
            "sem_score": 0,
            "title": nom['title'],
            "description": nom['description'],
        }
    for norm in sem_norm:
        doc_id = norm['doc_id']

        if doc_id not in combined_nom:
            combined_nom[doc_id] = {
                "doc_id": doc_id,
                "bm25_score": 0,
                "sem_score": 0,
                "title": norm['title'],
                "description": norm['description'],
            }
        combined_nom[doc_id]['sem_score'] = norm['normalized_score']

    for k, v in combined_nom.items():
        combined_nom[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'], alpha)

    return sorted(combined_nom.values(), key=lambda x: x['hybrid_score'], reverse=True)

def normalize_scores(scores: list[dict]) -> list[dict]:
    if not scores: return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.] * len(scores)

    score_range = max_score - min_score

    return [(score - min_score) / score_range for score in scores]