import string
from nltk.stem import PorterStemmer

from lib.search_utils import load_stop_words

stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> list[str]:
    cleaned_text = clean_text(text)
    stop_words = load_stop_words()

    tokens: list[str] = []
    for token in cleaned_text.split():
        if token and token not in stop_words:
            tokens.append(stemmer.stem(token))

    return tokens


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected 1 token, got {len(tokens)}")
    return tokens[0]
