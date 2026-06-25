
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.query_processor import QueryProcessor

def main():
    print("Executing Query Matching & Ranking Verification...")
    print("=" * 70)
    
    processor = QueryProcessor()
    
   
    user_query = "What are the revenues of stoks and investmnt"
    print(f"User Query: '{user_query}'\n")
    
    corrected_query = processor.refinement_service.suggest_correction(user_query)
    processed_query = processor.pipeline.preprocess_text(corrected_query)
    
    print(f"Refined Query    : '{corrected_query}'")
    print(f"Processed Tokens : '{processed_query}'")
    print("-" * 70)
    
    print("1. VSM (TF-IDF) Matching & Top-3 Ranking:")
    tfidf_results = processor.hybrid_indexer.tfidf_indexer.search(processed_query, top_k=3)
    for rank, (doc_id, score) in enumerate(tfidf_results, start=1):
        print(f"   Rank {rank}: Doc ID = {doc_id} -> Score/Similarity = {score:.6f}")
        
    print("\n2. Probabilistic (BM25) Matching & Top-3 Ranking:")
    bm25_results = processor.hybrid_indexer.bm25_indexer.search(processed_query, top_k=3)
    for rank, (doc_id, score) in enumerate(bm25_results, start=1):
        print(f"   Rank {rank}: Doc ID = {doc_id} -> BM25 Score = {score:.4f}")
        
    print("\n3. Dense Semantic (Embedding Cosine) Matching & Top-3 Ranking:")
    embed_results = processor.hybrid_indexer.embedding_indexer.search(corrected_query, top_k=3)
    for rank, (doc_id, score) in enumerate(embed_results, start=1):
        print(f"   Rank {rank}: Doc ID = {doc_id} -> Cosine Similarity = {score:.6f}")
    
    print("-" * 70)
    
    print("4. Final Unified Ranking via Reciprocal Rank Fusion (RRF):")
    final_hybrid_results = processor.process_and_search(user_query, search_mode="parallel", top_k=3)
    for rank, (doc_id, rrf_score) in enumerate(final_hybrid_results, start=1):
        print(f"    Rank {rank}: Doc ID = {doc_id} -> Combined RRF Score = {rrf_score:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()