import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

 
from services.query_processor import QueryProcessor

def main():
    processor = QueryProcessor()
    
    user_query = "How do non-controlling interests affect the balance sheets?"
    print(f"1. User Typed (Raw UI Input): '{user_query}'")
    
    processed = processor.pipeline.preprocess_text(user_query)
    print(f"2. System Processed Query   : '{processed}'")
    print("-" * 50)
    
    print("3. Executing Hybrid Parallel Search via QueryProcessor...")
    results = processor.process_and_search(user_query, search_mode="parallel", top_k=3)
    
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}: Doc ID = {doc_id} | Hybrid Score = {score:.6f}")

if __name__ == "__main__":
    main()