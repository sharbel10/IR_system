from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "data" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def build_clusters(n_clusters=5):
    print("Loading embeddings...")
    embeddings = np.load(MODELS_DIR / "embeddings.npy")

    with open(MODELS_DIR / "embedding_doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    print(f"Building KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters_df = pd.DataFrame({
        "doc_id": doc_ids,
        "cluster": labels
    })

    clusters_df.to_csv(PROCESSED_DIR / "document_clusters.csv", index=False)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")

    print("Clustering saved successfully!")
    print(f"Saved: {PROCESSED_DIR / 'document_clusters.csv'}")
    print(f"Saved: {MODELS_DIR / 'kmeans_model.pkl'}")

if __name__ == "__main__":
    build_clusters(n_clusters=5)