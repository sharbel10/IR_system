import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class BM25Indexer:
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []  # row i in the BM25 index corresponds to doc_ids[i]
        self.models_dir = BASE_DIR / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # OFFLINE INDEXING: tokenize the corpus, build BM25 statistics, and save the model.
    def fit_and_save(self, docs_df):
        print("Building BM25 Index...")
        self.doc_ids = [str(uid) for uid in docs_df['doc_id'].tolist()]
        # BM25 expects a list of token lists, not raw strings — split preprocessed text on whitespace.
        corpus_tokenized = [str(text).split() for text in docs_df['processed_text'].fillna("").tolist()]

        # BM25Okapi precomputes document frequencies, lengths, and average doc length from the corpus.
        self.bm25 = BM25Okapi(corpus_tokenized)

        # Save the fitted BM25 model and the doc_id mapping for query-time retrieval.
        with open(self.models_dir / "bm25_model.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.models_dir / "bm25_doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)
        print("BM25 Index saved successfully.")

    def load_models(self):
        with open(self.models_dir / "bm25_model.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.models_dir / "bm25_doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)

    # ONLINE SEARCH: score the query against every document using the saved BM25 index.
    def search(self, processed_query, k1=1.5, b=0.75, top_k=10):
        if self.bm25 is None:
            self.load_models()

        # k1 controls term-frequency saturation; b controls document-length normalization (tuned at query time).
        self.bm25.k1 = k1
        self.bm25.b = b

        # Split preprocessed query into tokens — same representation as indexed documents.
        query_tokens = str(processed_query).split()
        # Compare query tokens against the BM25 index built from processed document tokens.
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results