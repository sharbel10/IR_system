"""
EXPERIMENTAL ATTEMPT (not the main enhancement).

Cluster Boosting slightly worsened metrics on BEIR webis-touche2020, so it is kept
only as a documented experiment. The main "after enhancement" result is the
Weighted Hybrid / LTR feature in evaluate_ltr.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from evaluate_baseline import (
    calculate_metrics,
    get_qrels_query_ids,
    load_evaluation_data,
)
from services.cluster_boosting import ClusterBoostingService
from services.query_processor import QueryProcessor

DEFAULT_BOOST_FACTOR = 1.10
K = 10


def retrieve_with_cluster_boost(base_results, raw_query, cluster_boosting, boost_factor, top_k):
    """Apply cluster boosting to base (doc_id, score) results and return ranked doc IDs."""
    query_cluster = cluster_boosting.predict_query_cluster(raw_query)
    boosted = cluster_boosting.apply_cluster_boost(
        base_results,
        query_cluster=query_cluster,
        boost_factor=boost_factor,
        top_k=top_k,
    )
    return [item["doc_id"] for item in boosted]


def generate_charts(summary_df):
    metrics = ["MAP", "Recall", "Precision@10", "nDCG@10"]
    models = summary_df["Model Name"].tolist()

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric)

    ax.set_ylabel("Scores")
    ax.set_title("IR System Enhanced Evaluation Metrics (After Cluster Boosting)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    chart_path = BASE_DIR / "enhanced_evaluation_metrics.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Evaluation chart successfully saved to: {chart_path}")


def evaluate_system(boost_factor=DEFAULT_BOOST_FACTOR):
    print("Initializing Evaluation System (Enhanced - After Cluster Boosting)...")
    print("=" * 75)

    processor = QueryProcessor()
    cluster_boosting = ClusterBoostingService()
    queries, ground_truth = load_evaluation_data()

    qrels_query_ids = get_qrels_query_ids(ground_truth)
    unique_qrels_count = len(qrels_query_ids)

    print(f"Unique queries in qrels: {unique_qrels_count}")
    print(f"Cluster boost factor:    {boost_factor}")
    print("-" * 75)

    models = ["BM25 + Cluster Boost", "Hybrid + Cluster Boost"]
    results = {model: {"AP": [], "Recall": [], "P@10": [], "nDCG": []} for model in models}

    evaluated_count = 0
    skipped_count = 0
    skipped_missing_query_text = 0
    skipped_no_relevant_docs = 0

    for q_id in qrels_query_ids:
        relevant_docs = ground_truth[q_id]
        if not relevant_docs:
            skipped_count += 1
            skipped_no_relevant_docs += 1
            continue

        q_text = queries.get(q_id)
        if not q_text or not str(q_text).strip():
            skipped_count += 1
            skipped_missing_query_text += 1
            continue

        processed_query = processor.pipeline.preprocess_text(q_text)

        bm25_base = processor.hybrid_indexer.bm25_indexer.search(processed_query, top_k=K)
        hybrid_base = processor.process_and_search(q_text, search_mode="parallel", top_k=K)

        bm25_boosted_ids = retrieve_with_cluster_boost(
            bm25_base, q_text, cluster_boosting, boost_factor, K
        )
        hybrid_boosted_ids = retrieve_with_cluster_boost(
            hybrid_base, q_text, cluster_boosting, boost_factor, K
        )

        for model, retrieved_ids in zip(
            models, [bm25_boosted_ids, hybrid_boosted_ids]
        ):
            ap, recall, p_10, ndcg = calculate_metrics(retrieved_ids, relevant_docs, k=K)
            results[model]["AP"].append(ap)
            results[model]["Recall"].append(recall)
            results[model]["P@10"].append(p_10)
            results[model]["nDCG"].append(ndcg)

        evaluated_count += 1
        if evaluated_count % 200 == 0:
            print(f"Processed {evaluated_count}/{unique_qrels_count} qrels queries...")

    print("-" * 75)
    print(f"Unique queries in qrels: {unique_qrels_count}")
    print(f"Evaluated queries:       {evaluated_count}")
    print(f"Skipped queries:         {skipped_count}")
    if skipped_count > 0:
        if skipped_missing_query_text:
            print(f"  - missing query text:  {skipped_missing_query_text}")
        if skipped_no_relevant_docs:
            print(f"  - no relevant docs:    {skipped_no_relevant_docs}")
    print("-" * 75)

    print("\nFinal Evaluation Metrics Summary (After Cluster Boosting):")
    print("=" * 75)
    print(f"{'Model Name':<25} | {'MAP':<10} | {'Recall':<10} | {'Precision@10':<12} | {'nDCG@10':<10}")
    print("-" * 75)

    summary_data = []
    for model in models:
        mean_map = np.mean(results[model]["AP"]) if results[model]["AP"] else 0.0
        mean_recall = np.mean(results[model]["Recall"]) if results[model]["Recall"] else 0.0
        mean_p10 = np.mean(results[model]["P@10"]) if results[model]["P@10"] else 0.0
        mean_ndcg = np.mean(results[model]["nDCG"]) if results[model]["nDCG"] else 0.0
        print(
            f"{model:<25} | {mean_map:<10.4f} | {mean_recall:<10.4f} | "
            f"{mean_p10:<12.4f} | {mean_ndcg:<10.4f}"
        )

        summary_data.append(
            {
                "Model Name": model,
                "MAP": mean_map,
                "Recall": mean_recall,
                "Precision@10": mean_p10,
                "nDCG@10": mean_ndcg,
            }
        )
    print("=" * 75)

    df_summary = pd.DataFrame(summary_data)
    csv_path = BASE_DIR / "evaluation_enhanced_results.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[INFO] Evaluation summary data saved to: {csv_path}")

    generate_charts(df_summary)


if __name__ == "__main__":
    evaluate_system()
