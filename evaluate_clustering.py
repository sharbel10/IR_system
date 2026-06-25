import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLUSTERS_FILE = PROCESSED_DIR / "document_clusters.csv"
CHART_PATH = PROCESSED_DIR / "cluster_distribution.png"


def load_cluster_assignments():
    if not CLUSTERS_FILE.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {CLUSTERS_FILE}\n"
            "Run clustering.py first to generate document_clusters.csv."
        )

    clusters_df = pd.read_csv(CLUSTERS_FILE)
    clusters_df["doc_id"] = clusters_df["doc_id"].astype(str)
    clusters_df["cluster"] = clusters_df["cluster"].astype(int)
    return clusters_df


def count_documents_per_cluster(clusters_df):
    counts = (
        clusters_df.groupby("cluster", sort=True)
        .size()
        .reset_index(name="document_count")
        .sort_values("cluster")
    )
    return counts


def print_cluster_counts(counts_df):
    print("Documents per cluster:")
    print("-" * 40)
    for _, row in counts_df.iterrows():
        print(f"Cluster {int(row['cluster'])}: {int(row['document_count'])} documents")
    print("-" * 40)
    print(f"Total documents: {int(counts_df['document_count'].sum())}")
    print(f"Total clusters:  {len(counts_df)}")


def generate_cluster_chart(counts_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    cluster_labels = [f"Cluster {int(c)}" for c in counts_df["cluster"]]
    ax.bar(cluster_labels, counts_df["document_count"], color="steelblue", edgecolor="black")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Documents")
    ax.set_title("Document Distribution Across Clusters")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for i, count in enumerate(counts_df["document_count"]):
        ax.text(i, count, str(int(count)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()


def evaluate_clustering():
    print("Loading saved cluster assignments...")
    print("=" * 40)

    clusters_df = load_cluster_assignments()
    counts_df = count_documents_per_cluster(clusters_df)

    print_cluster_counts(counts_df)
    generate_cluster_chart(counts_df)

    print(f"\n[INFO] Cluster distribution chart saved to: {CHART_PATH}")


if __name__ == "__main__":
    sys.path.append(str(BASE_DIR))
    evaluate_clustering()
