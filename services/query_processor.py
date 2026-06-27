import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from services.preprocessing import PreprocessingService 
from services.indexers.hybrid_indexer import HybridIndexer
from services.query_refinement import QueryRefinementService

class QueryProcessor:
    def __init__(self):
        self.hybrid_indexer = HybridIndexer()
        self.pipeline = PreprocessingService() 
        self.refinement_service = QueryRefinementService()

    def process_and_search(self, raw_query, search_mode="parallel", k1=1.5, b=0.75, top_k=10, apply_expansion=False, weights=None):
        corrected_query = self.refinement_service.suggest_correction(raw_query)

        # Requirement 4 — query processing: normalize the query with the same steps as documents.
        processed_query = self.pipeline.preprocess_text(corrected_query)
        
        if apply_expansion:
            processed_query = self.refinement_service.expand_with_synonyms(processed_query)
            
        if search_mode == "tfidf":
            return self.hybrid_indexer.tfidf_indexer.search(processed_query, top_k=top_k)
        elif search_mode == "bm25":
            return self.hybrid_indexer.bm25_indexer.search(processed_query, k1=k1, b=b, top_k=top_k)
        elif search_mode == "embedding":
            return self.hybrid_indexer.embedding_indexer.search(corrected_query, top_k=top_k)  # raw text for embeddings
        elif search_mode == "parallel":
            return self.hybrid_indexer.search_parallel(corrected_query, processed_query, k1=k1, b=b, top_k=top_k)
        elif search_mode == "serial":
            return self.hybrid_indexer.search_serial(corrected_query, processed_query, k1=k1, b=b, top_k=top_k)
        elif search_mode == "ltr":
            # Weighted Hybrid / LTR: configurable weights tuned on a validation split.
            return self.hybrid_indexer.search_weighted(corrected_query, processed_query, weights=weights, k1=k1, b=b, top_k=top_k)
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")