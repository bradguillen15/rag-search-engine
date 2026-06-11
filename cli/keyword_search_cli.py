import argparse
from lib.keyword_search import search_command, build_command, tf, idf_command, tf_idf_command

def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Calculate Term Frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", type=str, help="Search term to find count for")

    idf_parser = subparsers.add_parser("idf", help="Calculate Inverse Document Frequency for a term")
    idf_parser.add_argument("term", type=str, help="Search term to find IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tfidf_parser.add_argument("term", type=str, help="Search term to find TF-IDF for")

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
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tf_idf_command(args.doc_id, args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()