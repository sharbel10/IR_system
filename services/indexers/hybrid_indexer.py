import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from services.indexers.tfidf_indexer import TFIDFIndexer
from services.indexers.bm25_indexer import BM25Indexer
from services.indexers.embedding_indexer import EmbeddingIndexer

class HybridIndexer:
    def __init__(self):
        self.tfidf_indexer = TFIDFIndexer()
        self.bm25_indexer = BM25Indexer()
        self.embedding_indexer = EmbeddingIndexer()

    def search_parallel(self, raw_query, processed_query, k1=1.5, b=0.75, top_k=10, c=60):
        tfidf_res = self.tfidf_indexer.search(processed_query, top_k=100)
        bm25_res = self.bm25_indexer.search(processed_query, k1=k1, b=b, top_k=100)
        embed_res = self.embedding_indexer.search(raw_query, top_k=100)
        
        rrf_scores = defaultdict(float)
        
        for rank, (doc_id, _) in enumerate(tfidf_res, start=1):
            rrf_scores[doc_id] += 1.0 / (c + rank)
            
        for rank, (doc_id, _) in enumerate(bm25_res, start=1):
            rrf_scores[doc_id] += 1.0 / (c + rank)
            
        for rank, (doc_id, _) in enumerate(embed_res, start=1):
            rrf_scores[doc_id] += 1.0 / (c + rank)
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def search_serial(self, raw_query, processed_query, k1=1.5, b=0.75, candidate_pool=500, top_k=10):
        candidates = self.bm25_indexer.search(processed_query, k1=k1, b=b, top_k=candidate_pool)
        candidate_ids = [doc_id for doc_id, _ in candidates]
        
        if not candidate_ids:
            return []
            
        if self.embedding_indexer.embeddings is None:
            self.embedding_indexer.load_models()
            
        query_vector = self.embedding_indexer.model.encode([raw_query])
        
        results = []
        for doc_id in candidate_ids:
            if doc_id in self.embedding_indexer.doc_ids:
                idx = self.embedding_indexer.doc_ids.index(doc_id)
                doc_vector = self.embedding_indexer.embeddings[idx].reshape(1, -1)
                
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(query_vector, doc_vector)[0][0]
                results.append((doc_id, float(sim)))
                
        sorted_docs = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs