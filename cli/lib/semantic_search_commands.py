from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.chunking import fixed_sized_chunking, semantic_chunking
from lib.search_utils import load_movies
from lib.semantic_search import SemanticSearch

def search_chunked(query: str, limit: int = 5) -> list[dict]:
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)

    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document'][:100]}...")

def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def chunk_text_semantic(text: str, overlap: int = 0, max_chunk_size: int = 4) -> list[str]:
    chunks = semantic_chunking(text, overlap, max_chunk_size)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def chunk_text(text: str, overlap, chunk_size: int = 200) -> list[str]:
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def search_command(query: str, limit: int = 5) -> list[dict]:
    ss = SemanticSearch()
    ss.load_or_create_embeddings(load_movies())
    search_results = ss.search(query, limit)

    for index, result in enumerate(search_results):
        print(f"{index + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"{result['description'][:100]}")

def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
