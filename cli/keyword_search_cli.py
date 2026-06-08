import argparse
from lib.keyword_search import search_command

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies_searched = search_command(args.query)

            print(f"Searching for: {args.query}")
            for index, movie in enumerate(movies_searched[:5]):
                print(f"{index + 1}. {movie['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()