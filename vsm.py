import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_to_string

DOCS_DIR = "docs"
TFIDF_MATRIX_FILE = "tfidf_matrix.json"
SIMILARITY_FILE = "similarity_matrix.json"
DOC_MAP_FILE = "doc_map.json"


def load_documents(docs_dir=DOCS_DIR):
    """
    Reads all .txt files, applies preprocessing, and returns:
    - doc_map: {doc_id: filename}
    - processed_texts: [preprocessed_string, ...] (same order as doc_map)
    """
    doc_map = {}
    processed_texts = []
    filenames = sorted(f for f in os.listdir(docs_dir) if f.endswith(".txt"))

    for doc_id, filename in enumerate(filenames, start=1):
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        doc_map[doc_id] = filename
        processed_texts.append(preprocess_to_string(raw_text))

    return doc_map, processed_texts


def build_tfidf_matrix(processed_texts):
    """
    Converts preprocessed document strings into a TF-IDF matrix.
    Each row = one document. Each column = one term in the vocabulary.
    Each cell = that term's TF-IDF weight in that document.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    vocabulary = vectorizer.get_feature_names_out()
    return tfidf_matrix, vocabulary, vectorizer


def compute_similarity_matrix(tfidf_matrix):
    """
    Computes pairwise cosine similarity between every pair of documents.
    Returns an NxN matrix where entry [i][j] = similarity between doc i and doc j.
    Diagonal is always 1.0 (a document is identical to itself).
    """
    return cosine_similarity(tfidf_matrix)


def get_top_similar_pairs(similarity_matrix, doc_map, top_n=5):
    """
    Returns the top_n most similar DISTINCT document pairs
    (excludes self-similarity and duplicate pairs).
    """
    n = similarity_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):  # only upper triangle -> no duplicates/self-pairs
            pairs.append((i + 1, j + 1, similarity_matrix[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    results = []
    for doc_id1, doc_id2, score in pairs[:top_n]:
        results.append({
            "doc1": doc_map[doc_id1],
            "doc2": doc_map[doc_id2],
            "similarity": round(float(score), 4)
        })
    return results


def save_results(similarity_matrix, doc_map):
    with open(SIMILARITY_FILE, "w", encoding="utf-8") as f:
        json.dump(similarity_matrix.tolist(), f, indent=2)
    with open(DOC_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, indent=2)
    print(f"Saved similarity matrix -> {SIMILARITY_FILE}")
    print(f"Saved document map -> {DOC_MAP_FILE}")


def build_and_save():
    doc_map, processed_texts = load_documents()
    tfidf_matrix, vocabulary, vectorizer = build_tfidf_matrix(processed_texts)
    similarity_matrix = compute_similarity_matrix(tfidf_matrix)
    save_results(similarity_matrix, doc_map)

    print(f"\nDocuments processed: {len(doc_map)}")
    print(f"Vocabulary size (TF-IDF features): {len(vocabulary)}")

    return similarity_matrix, doc_map, vocabulary


def print_similarity_matrix(similarity_matrix, doc_map):
    print("\nCosine Similarity Matrix:")
    filenames = [doc_map[i] for i in sorted(doc_map.keys())] if isinstance(list(doc_map.keys())[0], int) \
        else [doc_map[str(i)] for i in range(1, len(doc_map) + 1)]

    short_names = [f[:15] for f in filenames]
    header = "".ljust(18) + "".join(name.ljust(18) for name in short_names)
    print(header)
    for i, row in enumerate(similarity_matrix):
        row_str = short_names[i].ljust(18)
        row_str += "".join(f"{val:.3f}".ljust(18) for val in row)
        print(row_str)


if __name__ == "__main__":
    similarity_matrix, doc_map, vocabulary = build_and_save()
    print_similarity_matrix(similarity_matrix, doc_map)

    print("\nTop 5 most similar document pairs:")
    top_pairs = get_top_similar_pairs(similarity_matrix, doc_map, top_n=5)
    for pair in top_pairs:
        print(f"  {pair['doc1']}  <->  {pair['doc2']}   similarity = {pair['similarity']}")