import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class TFIDFIndexer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.doc_ids = []
        self.models_dir = BASE_DIR / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_save(self, docs_df):
        print("Training TF-IDF Matrix...")
        processed_corpus = docs_df['processed_text'].fillna("").tolist()
        self.doc_ids = [str(uid) for uid in docs_df['doc_id'].tolist()]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)
        
        with open(self.models_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.models_dir / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(self.models_dir / "tfidf_doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)
        print("TF-IDF Matrix saved successfully.")

    def load_models(self):
        with open(self.models_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.models_dir / "tfidf_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self.models_dir / "tfidf_doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)

    def search(self, processed_query, top_k=10):
        if self.tfidf_matrix is None:
            self.load_models()
        
        query_vector = self.vectorizer.transform([processed_query])
        
        scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results