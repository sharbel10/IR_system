import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.indexers.tfidf_indexer import TFIDFIndexer

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
    
    indexer = TFIDFIndexer()
    indexer.fit_and_save(docs_df)
    
    print("\n--- Testing TF-IDF Search System ---")
    sample_query = queries_df['processed_query'].iloc[0]
    raw_query_text = queries_df['text'].iloc[0]
    print(f"Testing Query (Raw)      : {raw_query_text}")
    print(f"Testing Query (Processed): {sample_query}")
    
    results = indexer.search(sample_query, top_k=5)
    print(f"\nTop 5 Retrieved Results:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Similarity Score = {score:.4f}")

if __name__ == "__main__":
    main()