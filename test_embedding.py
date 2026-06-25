import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.indexers.embedding_indexer import EmbeddingIndexer

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
    
  
    print("Taking a sample of 10,000 documents for safe verification...")
    sample_docs_df = docs_df.head(10000).copy()
    
    indexer = EmbeddingIndexer()
    indexer.fit_and_save(sample_docs_df)
    
    print("\n--- Testing BERT Embedding Search System ---")
    raw_query_text = queries_df['text'].iloc[0]
    print(f"Testing Raw Semantic Query: {raw_query_text}")
    
    results = indexer.search(raw_query_text, top_k=3)
    print(f"\nTop 3 Semantically Retrieved Results:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Cosine Similarity = {score:.4f}")

if __name__ == "__main__":
    main()