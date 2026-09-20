"""
TECH 400 - Week 3 Assignment
Vector Space Model - Document Similarity System

Runs the full pipeline end-to-end:
  1. Load and preprocess documents from docs/
  2. Build the TF-IDF matrix
  3. Compute the cosine similarity matrix
  4. Print the full similarity matrix
  5. Print the top-N most similar document pairs
"""

from vsm import build_and_save, get_top_similar_pairs, print_similarity_matrix

TOP_N = 5

if __name__ == "__main__":
    print("=" * 55)
    print(" Vector Space Model - Document Similarity System")
    print(" TECH 400 - Week 3")
    print("=" * 55)

    similarity_matrix, doc_map, vocabulary = build_and_save()

    print_similarity_matrix(similarity_matrix, doc_map)

    print(f"\nTop {TOP_N} most similar document pairs:")
    top_pairs = get_top_similar_pairs(similarity_matrix, doc_map, top_n=TOP_N)
    for pair in top_pairs:
        print(f"  {pair['doc1']}  <->  {pair['doc2']}   similarity = {pair['similarity']}")