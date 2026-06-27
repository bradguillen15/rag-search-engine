import argparse
from lib.rag import query_answering, doc_summarization

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarization_parser = subparsers.add_parser("summarize", help="Perform document summarization")
    summarization_parser.add_argument("query", type=str, help="Search query for summarization")
    summarization_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query_answering(args.query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()