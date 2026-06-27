"""
Weighted Hybrid / LTR evaluation (main "after enhancement" feature).

Lightweight Learning-To-Rank: instead of fusing TF-IDF, BM25, and Embedding with
equal weight, we learn per-ranker weights for a weighted Reciprocal Rank Fusion.

Honest protocol:
  - Split the qrels queries into a validation set (~25%) and a held-out test set.
  - Tune the fusion weights ONLY on the validation set.
  - Report final metrics on the held-out test set.
  - Compare against baseline BM25 and baseline (equal-weight) Hybrid on the SAME test set.

We do not tune on the test set, so the reported "after" numbers are not overfit.
"""

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from evaluate_baseline import calculate_metrics, get_qrels_query_ids, load_evaluation_data
from services.query_processor import QueryProcessor

K = 10
VAL_RATIO = 0.25
RANDOM_SEED = 42

# Candidate weights per ranker; the grid is the LTR "search space".
WEIGHT_CHOICES = {
    "tfidf": [0.0, 0.5, 1.0],
    "bm25": [1.0, 2.0, 3.0],
    "embedding": [0.0, 0.5, 1.0],
}
TUNING_METRIC = "nDCG@10"  # metric used to pick the best weights on validation


def split_query_ids(qrels_query_ids, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    ids = list(qrels_query_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return set(ids[:n_val]), set(ids[n_val:])


def mean_metrics(buckets):
    if not buckets["AP"]:
        return {"MAP": 0.0, "Recall": 0.0, "Precision@10": 0.0, "nDCG@10": 0.0}
    return {
        "MAP": float(np.mean(buckets["AP"])),
        "Recall": float(np.mean(buckets["Recall"])),
        "Precision@10": float(np.mean(buckets["P@10"])),
        "nDCG@10": float(np.mean(buckets["nDCG"])),
    }


def evaluate(processor, queries, ground_truth, query_ids, scorer):
    """scorer(processed_query, raw_query) -> list of ranked doc_ids."""
    buckets = {"AP": [], "Recall": [], "P@10": [], "nDCG": []}
    for q_id in query_ids:
        relevant = ground_truth[q_id]
        if not relevant:
            continue
        q_text = queries.get(q_id)
        if not q_text or not str(q_text).strip():
            continue
        processed = processor.pipeline.preprocess_text(q_text)
        retrieved = scorer(processed, q_text)
        ap, recall, p10, ndcg = calculate_metrics(retrieved, relevant, k=K)
        buckets["AP"].append(ap)
        buckets["Recall"].append(recall)
        buckets["P@10"].append(p10)
        buckets["nDCG"].append(ndcg)
    return mean_metrics(buckets)


def bm25_scorer(processor):
    def scorer(processed, raw):
        return [d for d, _ in processor.hybrid_indexer.bm25_indexer.search(processed, top_k=K)]
    return scorer


def hybrid_equal_scorer(processor):
    def scorer(processed, raw):
        return [d for d, _ in processor.hybrid_indexer.search_parallel(raw, processed, top_k=K)]
    return scorer


def weighted_scorer(processor, weights):
    def scorer(processed, raw):
        return [
            d
            for d, _ in processor.hybrid_indexer.search_weighted(raw, processed, weights=weights, top_k=K)
        ]
    return scorer


def tune_weights(processor, queries, ground_truth, val_ids):
    rows = []
    best = None
    keys = ["tfidf", "bm25", "embedding"]
    for combo in product(*[WEIGHT_CHOICES[k] for k in keys]):
        weights = dict(zip(keys, combo))
        if all(v == 0 for v in weights.values()):
            continue
        metrics = evaluate(processor, queries, ground_truth, val_ids, weighted_scorer(processor, weights))
        row = {**{f"w_{k}": weights[k] for k in keys}, **metrics}
        rows.append(row)
        if best is None or metrics[TUNING_METRIC] > best[1][TUNING_METRIC]:
            best = (weights, metrics)
    return pd.DataFrame(rows), best


def save_chart(summary_df):
    metrics = ["MAP", "Recall", "Precision@10", "nDCG@10"]
    models = summary_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric)

    ax.set_ylabel("Scores")
    ax.set_title("Before vs After: Weighted Hybrid / LTR (held-out test queries)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = BASE_DIR / "ltr_evaluation_metrics.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Chart saved to: {chart_path}")


def main():
    print("Initializing Weighted Hybrid / LTR evaluation...")
    print("=" * 75)

    processor = QueryProcessor()
    queries, ground_truth = load_evaluation_data()
    qrels_query_ids = get_qrels_query_ids(ground_truth)

    val_ids, test_ids = split_query_ids(qrels_query_ids)
    print(f"Total qrels queries:  {len(qrels_query_ids)}")
    print(f"Validation (tuning):  {len(val_ids)}")
    print(f"Held-out test:        {len(test_ids)}")
    print(f"Tuning metric:        {TUNING_METRIC}")
    print("-" * 75)

    print("Tuning fusion weights on the validation split...")
    sweep_df, best = tune_weights(processor, queries, ground_truth, val_ids)
    best_weights, best_val_metrics = best
    sweep_path = BASE_DIR / "evaluation_ltr_validation_sweep.csv"
    sweep_df.sort_values(TUNING_METRIC, ascending=False).to_csv(sweep_path, index=False)

    print(f"\nBest weights on validation ({TUNING_METRIC}): {best_weights}")
    print(f"Validation metrics: {best_val_metrics}")
    print(f"[INFO] Full validation sweep saved to: {sweep_path}")
    print("-" * 75)

    print("Evaluating on held-out TEST queries (before vs after)...")
    summary = []
    summary.append({"Model": "BM25 (baseline)",
                    **evaluate(processor, queries, ground_truth, test_ids, bm25_scorer(processor))})
    summary.append({"Model": "Hybrid equal (baseline)",
                    **evaluate(processor, queries, ground_truth, test_ids, hybrid_equal_scorer(processor))})

    w = best_weights
    label = f"Weighted Hybrid / LTR (tfidf={w['tfidf']}, bm25={w['bm25']}, emb={w['embedding']})"
    summary.append({"Model": label,
                    **evaluate(processor, queries, ground_truth, test_ids, weighted_scorer(processor, w))})

    summary_df = pd.DataFrame(summary)

    print("\nHeld-out TEST results:")
    print("=" * 75)
    print(f"{'Model':<55} | {'MAP':<8} | {'Recall':<8} | {'P@10':<8} | {'nDCG@10':<8}")
    print("-" * 75)
    for _, r in summary_df.iterrows():
        print(f"{r['Model']:<55} | {r['MAP']:<8.4f} | {r['Recall']:<8.4f} | {r['Precision@10']:<8.4f} | {r['nDCG@10']:<8.4f}")
    print("=" * 75)

    csv_path = BASE_DIR / "evaluation_ltr_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to: {csv_path}")
    save_chart(summary_df)

    # Honest verdict based on held-out test, not validation.
    bm25_ndcg = summary_df.iloc[0]["nDCG@10"]
    ltr_ndcg = summary_df.iloc[2]["nDCG@10"]
    print("-" * 75)
    if ltr_ndcg > bm25_ndcg:
        print(f"VERDICT: Weighted Hybrid / LTR improved nDCG@10 over BM25 on held-out test "
              f"({ltr_ndcg:.4f} vs {bm25_ndcg:.4f}). Use it as the main enhancement.")
    else:
        print(f"VERDICT: Weighted Hybrid / LTR did NOT beat BM25 on held-out test "
              f"({ltr_ndcg:.4f} vs {bm25_ndcg:.4f}). Report this honestly; BM25 remains the strongest model.")


if __name__ == "__main__":
    main()
