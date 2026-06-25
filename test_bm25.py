import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.indexers.bm25_indexer import BM25Indexer

def main():
    processed_path = BASE_DIR / 'data' / 'processed'
    documents_file = processed_path / 'documents.csv'
    queries_file = processed_path / 'queries.csv'
    
    if not documents_file.exists() or not queries_file.exists():
        print("Error: Processed CSV files not found. Run preprocessing first.")
        return
        
    print("Loading processed documents...")
    docs_df = pd.read_csv(documents_file)
    queries_df = pd.read_csv(queries_file)
    
    indexer = BM25Indexer()
    indexer.fit_and_save(docs_df)
    
    print("\n--- Testing BM25 Search System ---")
    sample_query = queries_df['processed_query'].iloc[0]
    print(f"Testing Query: {sample_query}")
    
    print("\n[Test A] Using Default Parameters (k1=1.5, b=0.75):")
    results_a = indexer.search(sample_query, k1=1.5, b=0.75, top_k=3)
    for rank, (doc_id, score) in enumerate(results_a, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Score = {score:.4f}")
        
    print("\n[Test B] Changing Parameters Dynamically (k1=2.0, b=0.85):")
    results_b = indexer.search(sample_query, k1=2.0, b=0.85, top_k=3)
    for rank, (doc_id, score) in enumerate(results_b, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Score = {score:.4f}")

if __name__ == "__main__":
    main()