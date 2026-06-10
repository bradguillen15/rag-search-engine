import argparse
from lib.keyword_search import search_command, build_command, tf

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Show term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", type=str, help="Search term to find count for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies_searched = search_command(args.query)
            for movie in movies_searched:
                print(f"{movie['id']}. {movie['title']}")
        case "build":
            build_command()
        case "tf":
            tf(args.doc_id, args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()