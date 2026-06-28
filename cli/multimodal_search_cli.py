import argparse
from lib.multimodal_search import verify_image_embedding, image_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Image Embedding")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Path to the image to verify")

    image_search_parser = subparsers.add_parser("image_search", help="Search for movies using image")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image to search")
    image_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            image_search(args.image_path, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()