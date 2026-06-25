import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class EmbeddingIndexer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.doc_ids = []
        self.models_dir = BASE_DIR / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_save(self, docs_df):
        print("Generating BERT Embeddings...")
        self.doc_ids = [str(uid) for uid in docs_df['doc_id'].tolist()]
        corpus = docs_df['text'].fillna("").tolist()
        
        self.embeddings = self.model.encode(corpus, show_progress_bar=True, batch_size=128)
        
        np.save(self.models_dir / "embeddings.npy", self.embeddings)
        with open(self.models_dir / "embedding_doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)
        print("Embeddings saved successfully.")

    def load_models(self):
        self.embeddings = np.load(self.models_dir / "embeddings.npy")
        with open(self.models_dir / "embedding_doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)

    def search(self, raw_query, top_k=10):
        if self.embeddings is None:
            self.load_models()
            
        query_vector = self.model.encode([raw_query])
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(similarities[idx])))
        return results