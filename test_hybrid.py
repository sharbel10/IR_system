import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.indexers.hybrid_indexer import HybridIndexer

def main():
    processed_path = BASE_DIR / 'data' / 'processed'
    queries_file = processed_path / 'queries.csv'
    
    if not queries_file.exists():
        print("Error: Processed CSV files not found. Run preprocessing first.")
        return
        
    queries_df = pd.read_csv(queries_file)
    raw_query = queries_df['text'].iloc[0]
    processed_query = queries_df['processed_query'].iloc[0]
    
    indexer = HybridIndexer()
    
    print("\n--- Testing Parallel Hybrid Search (RRF Fusion) ---")
    parallel_results = indexer.search_parallel(raw_query, processed_query, top_k=3)
    for rank, (doc_id, score) in enumerate(parallel_results, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | RRF Score = {score:.6f}")
        
    print("\n--- Testing Serial Hybrid Search (Multi-stage Re-ranking) ---")
    serial_results = indexer.search_serial(raw_query, processed_query, top_k=3)
    for rank, (doc_id, score) in enumerate(serial_results, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Re-ranked Cosine Score = {score:.4f}")

if __name__ == "__main__":
    main()