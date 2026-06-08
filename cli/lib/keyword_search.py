import string
from nltk.stem import PorterStemmer
from lib.search_utils import Movie, load_movies, load_stop_words

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

def has_matching_token(query_tokens: list[str], movie_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False
            

def search_command(query: str, n_results: int = 5) -> list[Movie]:
    movies = load_movies()
    
    movies_searched = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, movie_tokens):
            movies_searched.append(movie)
        if len(movies_searched) == n_results:
            break

    return movies_searched
    