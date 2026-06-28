import argparse
from lib.rag import doc_answer_question, doc_summarization, doc_citations, doc_answer_detailed_question

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarization_parser = subparsers.add_parser("summarize", help="Perform document summarization")
    summarization_parser.add_argument("query", type=str, help="Search query for summarization")
    summarization_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    citations_parser = subparsers.add_parser("citations", help="Perform document citations")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    question_parser = subparsers.add_parser("question", help="Perform question answering")
    question_parser.add_argument("query", type=str, help="Search query for question answering")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            doc_answer_question(args.query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "citations":
            doc_citations(args.query, args.limit)
        case "question":
            doc_answer_detailed_question(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()