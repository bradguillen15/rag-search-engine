import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
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

    args = parser.parse_args()

    match args.command:
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()