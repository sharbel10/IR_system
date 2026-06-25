import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.query_processor import QueryProcessor

def load_evaluation_data():
    processed_path = BASE_DIR / 'data' / 'processed'
    queries_file = processed_path / 'queries.csv'
    qrels_file = processed_path / 'qrels.csv'
    
    if not queries_file.exists() or not qrels_file.exists():
        raise FileNotFoundError("Evaluation files (queries.csv or qrels.csv) are missing in data/processed/")
        
    df_queries = pd.read_csv(queries_file)
    df_qrels = pd.read_csv(qrels_file)
    
    queries = dict(zip(df_queries['query_id'].astype(str), df_queries['text'].astype(str)))
    
    ground_truth = defaultdict(set)
    for _, row in df_qrels.iterrows():
        q_id = str(row['query_id'])
        d_id = str(row['doc_id'])
        ground_truth[q_id].add(d_id)
        
    return queries, ground_truth


def get_qrels_query_ids(ground_truth):
    """All unique query IDs from qrels, sorted numerically when possible."""
    def sort_key(q_id):
        try:
            return (0, int(q_id))
        except ValueError:
            return (1, q_id)

    return sorted(ground_truth.keys(), key=sort_key)

def calculate_metrics(retrieved_ids, relevant_ids, k=10):
    if not relevant_ids:
        return 0.0, 0.0, 0.0, 0.0
        
    retrieved_k = retrieved_ids[:k]
    true_positives_k = [d for d in retrieved_k if d in relevant_ids]
    precision_k = len(true_positives_k) / k
    
    true_positives_all = [d for d in retrieved_ids if d in relevant_ids]
    recall = len(true_positives_all) / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
    
    ap_numerator = 0.0
    num_relevant_retrieved = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_retrieved += 1
            ap_numerator += num_relevant_retrieved / (i + 1)
    ap = ap_numerator / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
    
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k))])
    ndcg_k = dcg / idcg if idcg > 0 else 0.0
    
    return ap, recall, precision_k, ndcg_k

def generate_charts(summary_df):
    metrics = ['MAP', 'Recall', 'Precision@10', 'nDCG@10']
    models = summary_df['Model Name'].tolist()
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, summary_df[metric], width, label=metric)
        
    ax.set_ylabel('Scores')
    ax.set_title('IR System Baseline Evaluation Metrics (Before Enhancements)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    chart_path = BASE_DIR / 'baseline_evaluation_metrics.png'
    plt.savefig(chart_path, dpi=300)
    print(f"\n[INFO] Evaluation chart successfully saved to: {chart_path}")

def evaluate_system():
    print("Initializing Evaluation System (Baseline - Before Enhancements)...")
    print("=" * 75)
    
    processor = QueryProcessor()
    queries, ground_truth = load_evaluation_data()

    qrels_query_ids = get_qrels_query_ids(ground_truth)
    unique_qrels_count = len(qrels_query_ids)

    print(f"Unique queries in qrels: {unique_qrels_count}")
    print("-" * 75)

    models = ['TF-IDF', 'BM25', 'Embedding', 'Hybrid']
    results = {model: {'AP': [], 'Recall': [], 'P@10': [], 'nDCG': []} for model in models}

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

        tfidf_res = [d[0] for d in processor.hybrid_indexer.tfidf_indexer.search(processed_query, top_k=10)]
        bm25_res = [d[0] for d in processor.hybrid_indexer.bm25_indexer.search(processed_query, top_k=10)]
        embed_res = [d[0] for d in processor.hybrid_indexer.embedding_indexer.search(q_text, top_k=10)]
        hybrid_res = [d[0] for d in processor.process_and_search(q_text, search_mode="parallel", top_k=10)]

        for model, res in zip(models, [tfidf_res, bm25_res, embed_res, hybrid_res]):
            ap, recall, p_10, ndcg = calculate_metrics(res, relevant_docs, k=10)
            results[model]['AP'].append(ap)
            results[model]['Recall'].append(recall)
            results[model]['P@10'].append(p_10)
            results[model]['nDCG'].append(ndcg)

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

    print("\nFinal Evaluation Metrics Summary (Before Enhancements):")
    print("=" * 75)
    print(f"{'Model Name':<15} | {'MAP':<10} | {'Recall':<10} | {'Precision@10':<12} | {'nDCG@10':<10}")
    print("-" * 75)
    
    summary_data = []
    for model in models:
        mean_map = np.mean(results[model]['AP']) if results[model]['AP'] else 0.0
        mean_recall = np.mean(results[model]['Recall']) if results[model]['Recall'] else 0.0
        mean_p10 = np.mean(results[model]['P@10']) if results[model]['P@10'] else 0.0
        mean_ndcg = np.mean(results[model]['nDCG']) if results[model]['nDCG'] else 0.0
        print(f"{model:<15} | {mean_map:<10.4f} | {mean_recall:<10.4f} | {mean_p10:<12.4f} | {mean_ndcg:<10.4f}")
        
        summary_data.append({
            'Model Name': model,
            'MAP': mean_map,
            'Recall': mean_recall,
            'Precision@10': mean_p10,
            'nDCG@10': mean_ndcg
        })
    print("=" * 75)
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = BASE_DIR / 'evaluation_baseline_results.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"[INFO] Evaluation summary data saved to: {csv_path}")
    
    generate_charts(df_summary)

if __name__ == "__main__":
    evaluate_system()