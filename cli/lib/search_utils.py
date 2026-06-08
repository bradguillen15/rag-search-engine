import json
from pathlib import Path
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"


def load_movies() -> list[Movie]:
    with open(DATA_PATH / "movies.json", "r") as f:
        data: dict[str, list[Movie]] = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(DATA_PATH / "stopwords.txt", "r") as f:
        return f.read().splitlines()