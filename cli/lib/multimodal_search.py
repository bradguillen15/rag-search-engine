from torch._tensor import Tensor


from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.vector_utils import cosine_similarity
from lib.search_utils import load_movies

class MultiModalSearch:
    def __init__(self, documents: list[dict], model_name : str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: str, limit: int = 5):
        image_emb = self.embed_image(image_path)

        similarities = []
        for idx, text_emb in enumerate(self.embeddings):
            similarities.append((idx, cosine_similarity(image_emb, text_emb)))

        sorted_sims = sorted(similarities, key=lambda x: x[1], reverse=True)

        results = []

        for doc_idx, score in sorted_sims[:limit]:
            _doc = self.documents[doc_idx]
            results.append({
                "title": _doc["title"],
                "description": _doc["description"],
                "doc_id": doc_idx,
                "score": score, 
            })
        return results

def image_search(image_path: str, limit: int = 5):
    movies = load_movies()
    ms = MultiModalSearch(movies)
    results = ms.search_with_image(image_path, limit)
    for index, result in enumerate(results):
        print(f"{index + 1}. {result['title']} (similarity: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")

def verify_image_embedding(image_path: str):
    ms = MultiModalSearch()
    embedding= ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")