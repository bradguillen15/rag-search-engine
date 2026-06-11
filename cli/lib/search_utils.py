import json
from pathlib import Path
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CACHE_PATH = PROJECT_ROOT / "cache"

def load_movies() -> list[Movie]:
    with open(DATA_PATH / "movies.json", "r") as f:
        data: dict[str, list[Movie]] = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(DATA_PATH / "stopwords.txt", "r") as f:
        return f.read().splitlines()