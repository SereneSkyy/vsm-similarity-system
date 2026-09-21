# Vector Space Model — Document Similarity System

**TECH 400 — Week 3 Assignment**

A simple document similarity system built in Python. It scrapes a small
collection of documents from Wikipedia, preprocesses the text, builds a
TF-IDF (term frequency–inverse document frequency) representation of each
document, and computes pairwise **cosine similarity** between every document
in the collection using the Vector Space Model.

## Project Structure

```
vsm-similarity-system/
├── docs/                 # 8 scraped documents (corpus)
├── scraper.py            # Scrapes documents from Wikipedia
├── preprocess.py         # Tokenization, stopword removal, stemming
├── vsm.py                # Builds TF-IDF matrix + cosine similarity
├── main.py                # Entry point: runs the full pipeline
├── requirements.txt        # Python dependencies
├── similarity_matrix.json   # Generated similarity scores (doc x doc)
├── doc_map.json               # Generated doc ID -> filename map
└── README.md                    # This file
```

## What It Does

1. **Scrapes** 8 documents from Wikipedia (topics deliberately chosen so some
   pairs are related and some are not, to make similarity results meaningful).
2. **Preprocesses** each document: lowercasing, punctuation removal,
   tokenization, stopword removal, and stemming (NLTK).
3. **Builds a TF-IDF matrix**: each document becomes a vector, where each
   dimension is a term's TF-IDF weight (scikit-learn `TfidfVectorizer`).
4. **Computes cosine similarity** between every pair of documents, producing
   an 8x8 similarity matrix.
5. **Prints results**: the full similarity matrix and the top-5 most similar
   document pairs.

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/SereneSkyy/vsm-similarity-system.git
cd vsm-similarity-system

# 2. Set up a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1      # Windows (PowerShell)
source venv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 5. Scrape the documents (already included in docs/, but can be re-run)
python scraper.py

# 6. Run the full pipeline
python main.py
```

`main.py` will build the TF-IDF matrix, compute similarity scores, and print
the full similarity matrix along with the top-5 most similar document pairs
to the terminal.

## References

- Lee, D. L., Chuang, H., & Seamons, K. (1997). Document ranking and the
  vector-space model. _IEEE Software_, 14(2), 67–75.
- Ramos, J. (2003, December). Using tf-idf to determine word relevance in
  document queries. In _Proceedings of the First Instructional Conference on
  Machine Learning_ (Vol. 242, No. 1, pp. 29–48).
