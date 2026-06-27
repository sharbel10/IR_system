"""
Honest tuning / diagnostics for post-retrieval enhancements.

Uses a fixed train/validation/test-style split from qrels queries:
  - validation (~25%): pick hyperparameters
  - test (remaining): report final numbers once

Do NOT tune directly on the full 49-query set and claim improvement without this split.
"""

import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from evaluate_baseline import calculate_metrics, get_qrels_query_ids, load_evaluation_data
from services.cluster_boosting import ClusterBoostingService
from services.query_processor import QueryProcessor

K = 10
VAL_RATIO = 0.25
RANDOM_SEED = 42
BOOST_FACTORS = [1.02, 1.05, 1.10, 1.20, 1.50, 2.0]
CANDIDATE_POOLS = [10, 50, 100, 200]
RRF_WEIGHT_GRID = [
    {"name": "equal (baseline hybrid)", "w_tfidf": 1.0, "w_bm25": 1.0, "w_embed": 1.0},
    {"name": "bm25-heavy", "w_tfidf": 0.5, "w_bm25": 2.0, "w_embed": 0.5},
    {"name": "bm25-only-ish", "w_tfidf": 0.0, "w_bm25": 3.0, "w_embed": 0.5},
    {"name": "no-embedding", "w_tfidf": 1.0, "w_bm25": 2.0, "w_embed": 0.0},
]
BM25_PARAM_GRID = [(1.2, 0.6), (1.5, 0.75), (1.8, 0.85), (2.0, 0.9)]


def split_query_ids(qrels_query_ids, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    ids = list(qrels_query_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    test_ids = set(ids[n_val:])
    return val_ids, test_ids


def mean_metrics(metric_lists):
    if not metric_lists["AP"]:
        return {key: 0.0 for key in ["MAP", "Recall", "Precision@10", "nDCG@10"]}
    return {
        "MAP": float(np.mean(metric_lists["AP"])),
        "Recall": float(np.mean(metric_lists["Recall"])),
        "Precision@10": float(np.mean(metric_lists["P@10"])),
        "nDCG@10": float(np.mean(metric_lists["nDCG"])),
    }


def evaluate_query_ids(processor, cluster_boosting, queries, ground_truth, query_ids, scorer):
    buckets = {"AP": [], "Recall": [], "P@10": [], "nDCG": []}
    for q_id in sorted(query_ids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        relevant = ground_truth[q_id]
        if not relevant:
            continue
        q_text = queries.get(q_id)
        if not q_text or not str(q_text).strip():
            continue
        retrieved_ids = scorer(processor, cluster_boosting, q_id, q_text)
        ap, recall, p10, ndcg = calculate_metrics(retrieved_ids, relevant, k=K)
        buckets["AP"].append(ap)
        buckets["Recall"].append(recall)
        buckets["P@10"].append(p10)
        buckets["nDCG"].append(ndcg)
    return buckets


def search_parallel_weighted(
    hybrid_indexer,
    raw_query,
    processed_query,
    w_tfidf=1.0,
    w_bm25=1.0,
    w_embed=1.0,
    k1=1.5,
    b=0.75,
    top_k=10,
    c=60,
    pool=100,
):
    tfidf_res = hybrid_indexer.tfidf_indexer.search(processed_query, top_k=pool)
    bm25_res = hybrid_indexer.bm25_indexer.search(processed_query, k1=k1, b=b, top_k=pool)
    embed_res = hybrid_indexer.embedding_indexer.search(raw_query, top_k=pool)

    rrf_scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(tfidf_res, start=1):
        rrf_scores[doc_id] += w_tfidf / (c + rank)
    for rank, (doc_id, _) in enumerate(bm25_res, start=1):
        rrf_scores[doc_id] += w_bm25 / (c + rank)
    for rank, (doc_id, _) in enumerate(embed_res, start=1):
        rrf_scores[doc_id] += w_embed / (c + rank)

    return [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def apply_cluster_boost_ids(base_results, raw_query, cluster_boosting, boost_factor, top_k):
    query_cluster = cluster_boosting.predict_query_cluster(raw_query)
    boosted = cluster_boosting.apply_cluster_boost(
        base_results,
        query_cluster=query_cluster,
        boost_factor=boost_factor,
        top_k=top_k,
    )
    return [item["doc_id"] for item in boosted], query_cluster


def run_diagnostics(processor, cluster_boosting, queries, ground_truth, query_ids, boost_factor=1.10):
    print("\n" + "=" * 75)
    print("DIAGNOSTICS: why cluster boosting may not help")
    print("=" * 75)

    cluster_boosting._load_resources()

    top10_changed = 0
    order_only_changed = 0
    unchanged = 0
    total = 0

    rel_in_query_cluster = 0
    rel_total = 0
    irr_in_query_cluster_top10 = 0
    irr_total_top10 = 0

    for q_id in query_ids:
        relevant = ground_truth[q_id]
        if not relevant:
            continue
        q_text = queries.get(q_id)
        if not q_text or not str(q_text).strip():
            continue

        processed = processor.pipeline.preprocess_text(q_text)
        base = processor.hybrid_indexer.bm25_indexer.search(processed, top_k=K)
        boosted_ids, query_cluster = apply_cluster_boost_ids(
            base, q_text, cluster_boosting, boost_factor, K
        )
        base_ids = [doc_id for doc_id, _ in base]

        total += 1
        if base_ids == boosted_ids:
            unchanged += 1
        elif set(base_ids) == set(boosted_ids):
            order_only_changed += 1
        else:
            top10_changed += 1

        for doc_id in relevant:
            rel_total += 1
            doc_cluster = cluster_boosting._doc_clusters.get(str(doc_id))
            if doc_cluster is not None and doc_cluster == query_cluster:
                rel_in_query_cluster += 1

        for doc_id in base_ids:
            if doc_id not in relevant:
                irr_total_top10 += 1
                doc_cluster = cluster_boosting._doc_clusters.get(str(doc_id))
                if doc_cluster is not None and doc_cluster == query_cluster:
                    irr_in_query_cluster_top10 += 1

    print(f"Queries analyzed: {total}")
    print(
        f"BM25 top-10 unchanged after boost: {unchanged}/{total} "
        f"({100 * unchanged / total:.1f}%)"
    )
    print(
        f"Same 10 docs, different order only: {order_only_changed}/{total} "
        f"({100 * order_only_changed / total:.1f}%)"
    )
    print(
        f"Top-10 document set changed: {top10_changed}/{total} "
        f"({100 * top10_changed / total:.1f}%)"
    )
    if rel_total:
        print(
            f"Relevant docs sharing predicted query cluster: "
            f"{rel_in_query_cluster}/{rel_total} ({100 * rel_in_query_cluster / rel_total:.1f}%)"
        )
    if irr_total_top10:
        print(
            f"Irrelevant docs in BM25 top-10 that share query cluster (boosted): "
            f"{irr_in_query_cluster_top10}/{irr_total_top10} "
            f"({100 * irr_in_query_cluster_top10 / irr_total_top10:.1f}%)"
        )
    print(
        "\nInterpretation: if top-10 sets rarely change, Recall@10 and Precision@10 "
        "cannot improve. MAP/nDCG can only move via reordering."
    )


def tune_cluster_boost(processor, cluster_boosting, queries, ground_truth, val_ids):
    rows = []
    for boost_factor in BOOST_FACTORS:
        for pool in CANDIDATE_POOLS:
            def scorer(proc, boost_svc, q_id, q_text):
                processed = proc.pipeline.preprocess_text(q_text)
                base = proc.hybrid_indexer.bm25_indexer.search(processed, top_k=pool)
                boosted_ids, _ = apply_cluster_boost_ids(
                    base, q_text, boost_svc, boost_factor, K
                )
                return boosted_ids

            metrics = mean_metrics(
                evaluate_query_ids(processor, cluster_boosting, queries, ground_truth, val_ids, scorer)
            )
            rows.append(
                {
                    "method": "BM25 + cluster boost",
                    "boost_factor": boost_factor,
                    "candidate_pool": pool,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def tune_rrf_weights(processor, queries, ground_truth, val_ids, k1=1.5, b=0.75):
    rows = []
    for cfg in RRF_WEIGHT_GRID:
        def scorer(proc, _boost, q_id, q_text):
            processed = proc.pipeline.preprocess_text(q_text)
            return search_parallel_weighted(
                proc.hybrid_indexer,
                q_text,
                processed,
                w_tfidf=cfg["w_tfidf"],
                w_bm25=cfg["w_bm25"],
                w_embed=cfg["w_embed"],
                k1=k1,
                b=b,
                top_k=K,
            )

        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, val_ids, scorer)
        )
        rows.append({"method": "weighted hybrid RRF", "config": cfg["name"], **metrics})
    return pd.DataFrame(rows)


def tune_bm25_params(processor, queries, ground_truth, val_ids):
    rows = []
    for k1, b in BM25_PARAM_GRID:
        def scorer(proc, _boost, q_id, q_text):
            processed = proc.pipeline.preprocess_text(q_text)
            return [
                doc_id
                for doc_id, _ in proc.hybrid_indexer.bm25_indexer.search(
                    processed, k1=k1, b=b, top_k=K
                )
            ]

        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, val_ids, scorer)
        )
        rows.append({"method": "BM25", "k1": k1, "b": b, **metrics})
    return pd.DataFrame(rows)


def tune_query_expansion(processor, queries, ground_truth, val_ids):
    def scorer(proc, _boost, q_id, q_text):
        corrected = proc.refinement_service.suggest_correction(q_text)
        processed = proc.pipeline.preprocess_text(corrected)
        processed = proc.refinement_service.expand_with_synonyms(processed)
        return [
            doc_id
            for doc_id, _ in proc.hybrid_indexer.bm25_indexer.search(processed, top_k=K)
        ]

    metrics = mean_metrics(
        evaluate_query_ids(processor, None, queries, ground_truth, val_ids, scorer)
    )
    return {"method": "BM25 + synonym expansion", **metrics}


def pick_best(df, metric="nDCG@10"):
    return df.sort_values(metric, ascending=False).iloc[0]


def evaluate_best_on_test(processor, cluster_boosting, queries, ground_truth, test_ids, best_configs):
    rows = []

    # Baselines on test
    def bm25_baseline(proc, _b, _qid, q_text):
        processed = proc.pipeline.preprocess_text(q_text)
        return [d for d, _ in proc.hybrid_indexer.bm25_indexer.search(processed, top_k=K)]

    def hybrid_baseline(proc, _b, _qid, q_text):
        return [d for d, _ in proc.process_and_search(q_text, search_mode="parallel", top_k=K)]

    for label, scorer in [
        ("BM25 baseline", bm25_baseline),
        ("Hybrid baseline (equal RRF)", hybrid_baseline),
    ]:
        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, test_ids, scorer)
        )
        rows.append({"Model": label, **metrics})

    # Best cluster boost (if any beat baseline on validation)
    cb = best_configs.get("cluster_boost")
    if cb is not None:
        boost_factor = cb["boost_factor"]
        pool = int(cb["candidate_pool"])

        def cluster_scorer(proc, boost_svc, _qid, q_text):
            processed = proc.pipeline.preprocess_text(q_text)
            base = proc.hybrid_indexer.bm25_indexer.search(processed, top_k=pool)
            boosted_ids, _ = apply_cluster_boost_ids(base, q_text, boost_svc, boost_factor, K)
            return boosted_ids

        metrics = mean_metrics(
            evaluate_query_ids(processor, cluster_boosting, queries, ground_truth, test_ids, cluster_scorer)
        )
        rows.append(
            {
                "Model": f"BM25 + cluster boost (bf={boost_factor}, pool={pool})",
                **metrics,
            }
        )

    # Best BM25 params
    bp = best_configs.get("bm25")
    if bp is not None:
        k1, b = bp["k1"], bp["b"]

        def bm25_tuned(proc, _boost, _qid, q_text):
            processed = proc.pipeline.preprocess_text(q_text)
            return [
                d
                for d, _ in proc.hybrid_indexer.bm25_indexer.search(processed, k1=k1, b=b, top_k=K)
            ]

        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, test_ids, bm25_tuned)
        )
        rows.append({"Model": f"BM25 tuned (k1={k1}, b={b})", **metrics})

    # Best RRF weights
    rw = best_configs.get("rrf")
    if rw is not None:
        cfg = next(c for c in RRF_WEIGHT_GRID if c["name"] == rw["config"])

        def rrf_scorer(proc, _boost, _qid, q_text):
            processed = proc.pipeline.preprocess_text(q_text)
            return search_parallel_weighted(
                proc.hybrid_indexer,
                q_text,
                processed,
                w_tfidf=cfg["w_tfidf"],
                w_bm25=cfg["w_bm25"],
                w_embed=cfg["w_embed"],
                top_k=K,
            )

        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, test_ids, rrf_scorer)
        )
        rows.append({"Model": f"Weighted hybrid ({cfg['name']})", **metrics})

    qe = best_configs.get("query_expansion")
    if qe is not None:
        def qe_scorer(proc, _boost, _qid, q_text):
            corrected = proc.refinement_service.suggest_correction(q_text)
            processed = proc.pipeline.preprocess_text(corrected)
            processed = proc.refinement_service.expand_with_synonyms(processed)
            return [d for d, _ in proc.hybrid_indexer.bm25_indexer.search(processed, top_k=K)]

        metrics = mean_metrics(
            evaluate_query_ids(processor, None, queries, ground_truth, test_ids, qe_scorer)
        )
        rows.append({"Model": "BM25 + synonym expansion", **metrics})

    return pd.DataFrame(rows)


def save_chart(summary_df, path, title):
    metrics = ["MAP", "Recall", "Precision@10", "nDCG@10"]
    models = summary_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric)

    ax.set_ylabel("Scores")
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    print("Loading evaluation data and models...")
    processor = QueryProcessor()
    cluster_boosting = ClusterBoostingService()
    queries, ground_truth = load_evaluation_data()
    qrels_query_ids = get_qrels_query_ids(ground_truth)

    val_ids, test_ids = split_query_ids(qrels_query_ids)
    print(f"Total qrels queries: {len(qrels_query_ids)}")
    print(f"Validation queries:    {len(val_ids)}  (used for tuning)")
    print(f"Test queries:          {len(test_ids)}  (held-out final report)")

    run_diagnostics(processor, cluster_boosting, queries, ground_truth, qrels_query_ids)

    print("\n" + "=" * 75)
    print("VALIDATION TUNING (do not report these as final test results)")
    print("=" * 75)

    cb_df = tune_cluster_boost(processor, cluster_boosting, queries, ground_truth, val_ids)
    cb_baseline_val = mean_metrics(
        evaluate_query_ids(
            processor,
            None,
            queries,
            ground_truth,
            val_ids,
            lambda proc, _b, _qid, q_text: [
                d
                for d, _ in proc.hybrid_indexer.bm25_indexer.search(
                    proc.pipeline.preprocess_text(q_text), top_k=K
                )
            ],
        )
    )
    print("\nBM25 validation baseline:", cb_baseline_val)
    print("\nTop cluster-boost configs on validation (by nDCG@10):")
    print(cb_df.sort_values("nDCG@10", ascending=False).head(6).to_string(index=False))

    rrf_df = tune_rrf_weights(processor, queries, ground_truth, val_ids)
    hybrid_baseline_val = mean_metrics(
        evaluate_query_ids(
            processor,
            None,
            queries,
            ground_truth,
            val_ids,
            lambda proc, _b, _qid, q_text: [
                d for d, _ in proc.process_and_search(q_text, search_mode="parallel", top_k=K)
            ],
        )
    )
    print("\nHybrid validation baseline:", hybrid_baseline_val)
    print("\nWeighted hybrid configs on validation:")
    print(rrf_df.sort_values("nDCG@10", ascending=False).to_string(index=False))

    bm25_df = tune_bm25_params(processor, queries, ground_truth, val_ids)
    print("\nBM25 parameter grid on validation:")
    print(bm25_df.sort_values("nDCG@10", ascending=False).to_string(index=False))

    qe_metrics = tune_query_expansion(processor, queries, ground_truth, val_ids)
    print("\nBM25 + synonym expansion on validation:", qe_metrics)

    best_configs = {
        "cluster_boost": pick_best(cb_df) if not cb_df.empty else None,
        "rrf": pick_best(rrf_df) if not rrf_df.empty else None,
        "bm25": pick_best(bm25_df) if not bm25_df.empty else None,
        "query_expansion": qe_metrics,
    }

    print("\n" + "=" * 75)
    print("HELD-OUT TEST RESULTS (report these honestly)")
    print("=" * 75)

    test_df = evaluate_best_on_test(
        processor, cluster_boosting, queries, ground_truth, test_ids, best_configs
    )
    print(test_df.to_string(index=False))

    val_tuning_path = BASE_DIR / "evaluation_tuning_validation.csv"
    test_results_path = BASE_DIR / "evaluation_tuning_test_results.csv"
    cb_df.to_csv(val_tuning_path, index=False)
    test_df.to_csv(test_results_path, index=False)

    chart_path = BASE_DIR / "tuning_comparison_chart.png"
    save_chart(test_df, chart_path, "Held-out Test Metrics: Baselines vs Best-Tuned Settings")

    print(f"\n[INFO] Validation sweep saved to: {val_tuning_path}")
    print(f"[INFO] Held-out test results saved to: {test_results_path}")
    print(f"[INFO] Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
