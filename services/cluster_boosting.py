from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class ClusterBoostingService:
    """Apply cluster-based score boosting to retrieval results.

    This is boosting, not filtering: all documents returned by the base retriever
    remain in the result set. Documents whose cluster matches the predicted query
    cluster receive a multiplicative score boost, then the list is re-sorted.
    """

    def __init__(self):
        self.processed_dir = BASE_DIR / "data" / "processed"
        self.models_dir = BASE_DIR / "data" / "models"
        self._kmeans = None
        self._doc_clusters = None
        self._encoder = None

    def _load_resources(self):
        if self._kmeans is None:
            kmeans_path = self.models_dir / "kmeans_model.pkl"
            if not kmeans_path.exists():
                raise FileNotFoundError(
                    f"Saved KMeans model not found: {kmeans_path}. Run clustering.py first."
                )
            self._kmeans = joblib.load(kmeans_path)

        if self._doc_clusters is None:
            clusters_path = self.processed_dir / "document_clusters.csv"
            if not clusters_path.exists():
                raise FileNotFoundError(
                    f"Cluster assignments not found: {clusters_path}. Run clustering.py first."
                )
            clusters_df = pd.read_csv(clusters_path)
            clusters_df["doc_id"] = clusters_df["doc_id"].astype(str)
            self._doc_clusters = dict(
                zip(clusters_df["doc_id"], clusters_df["cluster"].astype(int))
            )

        if self._encoder is None:
            # Same SentenceTransformer model used when document embeddings were built.
            self._encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def predict_query_cluster(self, raw_query):
        """Encode the query and assign it to a cluster with the saved KMeans model."""
        self._load_resources()
        query_embedding = self._encoder.encode([raw_query])
        return int(self._kmeans.predict(query_embedding)[0])

    def apply_cluster_boost(self, results, query_cluster, boost_factor=1.10, top_k=None):
        """Boost matching-cluster scores and re-rank. No documents are removed."""
        self._load_resources()

        boosted_results = []
        for doc_id, original_score in results:
            doc_id = str(doc_id)
            doc_cluster = self._doc_clusters.get(doc_id)
            final_score = float(original_score)

            # Cluster-based boosting (not filtering): multiply score when clusters match.
            if doc_cluster is not None and doc_cluster == query_cluster:
                final_score *= boost_factor

            boosted_results.append(
                {
                    "doc_id": doc_id,
                    "original_score": float(original_score),
                    "final_score": final_score,
                    "doc_cluster": doc_cluster,
                }
            )

        boosted_results.sort(key=lambda item: item["final_score"], reverse=True)
        if top_k is not None:
            boosted_results = boosted_results[:top_k]
        return boosted_results
