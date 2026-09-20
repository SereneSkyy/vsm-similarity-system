import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text):
    """Lowercase and strip non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem_tokens(tokens):
    return [STEMMER.stem(t) for t in tokens]


def preprocess(text):
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> stem.
    Returns a list of processed term tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def preprocess_to_string(text):
    """
    Same pipeline, but returns a single space-joined string instead of
    a token list. scikit-learn's TfidfVectorizer expects raw strings
    (it does its own internal tokenization), so this is the version
    vsm.py will actually use.
    """
    return " ".join(preprocess(text))


if __name__ == "__main__":
    sample = "Information Retrieval systems help users find relevant documents efficiently!"
    print("Original:", sample)
    print("Token list:", preprocess(sample))
    print("As string (for TF-IDF):", preprocess_to_string(sample))