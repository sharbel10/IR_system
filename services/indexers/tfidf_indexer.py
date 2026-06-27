import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class TFIDFIndexer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.doc_ids = []  # row i in the matrix corresponds to doc_ids[i]
        self.models_dir = BASE_DIR / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # OFFLINE INDEXING: build the index once from the full corpus and persist it to disk.
    def fit_and_save(self, docs_df):
        print("Training TF-IDF Matrix...")
        # Preprocessed (tokenized/stemmed) text for every document — input to the vectorizer.
        processed_corpus = docs_df['processed_text'].fillna("").tolist()
        # Parallel list of IDs so search results can return doc_id instead of matrix row index.
        self.doc_ids = [str(uid) for uid in docs_df['doc_id'].tolist()]
        # fit_transform: learn vocabulary + IDF from the corpus, then build the document-term matrix.
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)

        # Persist vectorizer (vocabulary/IDF), precomputed matrix, and ID mapping for later search.
        with open(self.models_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.models_dir / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(self.models_dir / "tfidf_doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)
        print("TF-IDF Matrix saved successfully.")

    def load_models(self):
        # Reload saved index artifacts so search() works without rebuilding the corpus.
        with open(self.models_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.models_dir / "tfidf_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self.models_dir / "tfidf_doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)

    # ONLINE SEARCH: score a query against the prebuilt index (no re-fitting on the query).
    def search(self, processed_query, top_k=10):
        if self.tfidf_matrix is None:
            self.load_models()

        # Query must be preprocessed like documents — the index was built on processed_text.
        # transform (not fit_transform): reuse the vocabulary/IDF from indexing; do not relearn from one query.
        query_vector = self.vectorizer.transform([processed_query])

        # Dot product between each document row and the query column = TF-IDF cosine-like score.
        scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results