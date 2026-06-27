import argparse
from lib.rerank import cross_encoder_rerank
from lib.hybrid_search import normalize_scores, weighted_search, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normalize a list of strings")
    norm_parser.add_argument("scores", type=float, nargs='+', help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="A hybrid search with weighted average scores")
    weighted_search_parser.add_argument("query", type=str, help="User query to find relevant docs for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="% if weight for nm25")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="# of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="A hybrid search with RRF scores")
    rrf_search_parser.add_argument("query", type=str, help="User query to find relevant docs for")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K parameter for RRF")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="# of results to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank method")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Run LLM as a judge in the results")

    args = parser.parse_args()

    match args.command:
        case "cross_encoder":
            cross_encoder_rerank(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for normalized_score in normalized_scores:
                print(f"* {normalized_score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()