import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.indexers.tfidf_indexer import TFIDFIndexer
from services.indexers.bm25_indexer import BM25Indexer
from services.indexers.embedding_indexer import EmbeddingIndexer

def run_pipeline_indexing():
    processed_path = BASE_DIR / 'data' / 'processed'
    documents_file = processed_path / 'documents.csv'
    
    if not documents_file.exists():
        print("Error: processed/documents.csv not found. Run preprocessing first.")
        return
        
    print("Loading processed documents for global indexing...")
    docs_df = pd.read_csv(documents_file)
    
   
    tfidf = TFIDFIndexer()
    tfidf.fit_and_save(docs_df)
    
  
    bm25 = BM25Indexer()
    bm25.fit_and_save(docs_df)
    
   
    embedding = EmbeddingIndexer()
  
    embedding.fit_and_save(docs_df) 
    
    print("\nAll Indices Built and Saved Successfully under 'data/models/'!")

if __name__ == "__main__":
    run_pipeline_indexing()