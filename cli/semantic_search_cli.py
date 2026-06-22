import argparse
from lib.semantic_search_commands import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_text,
    chunk_text_semantic,
    embed_chunks,
    search_chunked,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the embedding model loads properly")

    embed_parser = subparsers.add_parser("embed_text", help="Text with an embedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embedding model loads or creates embeddings properly")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to be encoded")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text into semantic chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Create embeddings for semantic search")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for movies using chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.overlap, args.max_chunk_size)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "search":
            search_command(args.query, args.limit)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()