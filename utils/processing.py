from typing import List
from nltk.stem import PorterStemmer
# import Stemmer
from collections import Counter
import re

alphanum = r"[a-zA-Z0-9_-]*"
non_alphanum = r"[^a-zA-Z0-9_-]"

stemmer = PorterStemmer()
# stemmer = Stemmer.Stemmer("english")
# load stopwords
with open("./data/stopwords.txt", "r") as stopwords_file:
    stopwords = set(stopwords_file.read().strip().split("\n"))


def tokenize_text(text: str) -> List[str]:
    """
    lowercases everything, converts non-alphanumeric chars into newlines
    returns an array of tokens, split by newline (drops empty strings)
    """
    text = text.lower()
    text = re.sub(non_alphanum, "\n", text)
    return list(filter(lambda token: bool(token), text.split("\n")))


def remove_stopwords(collection: List[str]) -> List[str]:
    """
    Drops all entries that are in the stopword list
    """
    return list(filter(lambda token: token not in stopwords, collection))


def normalize(collection: List[str]) -> List[str]:
    return list(map(lambda token: stemmer.stem(token), collection))


def count_tokens(collection: List[str]):
    """
    Returns a dictionary of counts for each token in the collection
    """
    return Counter(collection)


def process_tokens(tokens: List[str]):
    ts = remove_stopwords(tokens)
    ts = normalize(ts)
    return ts

def generate_vocab_growth_data(tokens: List[str], n: int):
    """
    Computes the size of unique vocab of a text every `n` words
    """
    # collection = tokenize_text(text)
    data = []
    vocab = set()
    for i, token in enumerate(tokens):
        if i % n == 0:
            data.append([i, len(vocab)])
        vocab.add(token)
    return data


if __name__ == "__main__":
    with open("data/bible.txt", "r") as file:
        data = file.read()
