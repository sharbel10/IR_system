from pathlib import Path

import streamlit as st

from services.preprocessing import PreprocessingService
from services.indexers.tfidf_indexer import TFIDFIndexer
from services.indexers.bm25_indexer import BM25Indexer
from services.indexers.embedding_indexer import EmbeddingIndexer
from services.indexers.hybrid_indexer import HybridIndexer
from services.cluster_boosting import ClusterBoostingService
from services.database import DocumentDatabase

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="IR Search System", layout="wide")
st.title("Information Retrieval Search System")


@st.cache_resource
def load_services():
    preprocessing = PreprocessingService()
    tfidf = TFIDFIndexer()
    bm25 = BM25Indexer()
    embedding = EmbeddingIndexer()
    hybrid = HybridIndexer()
    cluster_boosting = ClusterBoostingService()
    document_db = DocumentDatabase()
    return preprocessing, tfidf, bm25, embedding, hybrid, cluster_boosting, document_db


preprocessing, tfidf, bm25, embedding, hybrid, cluster_boosting, document_db = load_services()


def run_base_search(search_type, query, processed_query, k1, b, top_k):
    if search_type == "TF-IDF":
        return tfidf.search(processed_query, top_k=top_k)
    if search_type == "BM25":
        return bm25.search(processed_query, k1=k1, b=b, top_k=top_k)
    if search_type == "Embedding":
        return embedding.search(query, top_k=top_k)
    if search_type == "Hybrid Parallel":
        return hybrid.search_parallel(
            raw_query=query,
            processed_query=processed_query,
            k1=k1,
            b=b,
            top_k=top_k,
        )
    return hybrid.search_serial(
        raw_query=query,
        processed_query=processed_query,
        k1=k1,
        b=b,
        top_k=top_k,
    )


st.sidebar.header("Search Settings")
dataset = st.sidebar.selectbox("Dataset", ["BEIR / Webis Touche 2020"])
search_type = st.sidebar.selectbox(
    "Search Model",
    [
        "TF-IDF",
        "BM25",
        "Embedding",
        "Hybrid Parallel",
        "Hybrid Serial",
    ],
)

use_clustering = st.sidebar.checkbox(
    "Enable Additional Feature: Document Clustering",
    value=False,
    help="When enabled, matching-cluster documents receive a score boost and are re-ranked.",
)

cluster_boost_factor = 1.10
if use_clustering:
    cluster_boost_factor = st.sidebar.slider(
        "Cluster Boost Factor",
        min_value=1.0,
        max_value=2.0,
        value=1.10,
        step=0.05,
        help="Multiplier applied to scores when a document cluster matches the query cluster.",
    )

k1 = st.sidebar.slider("BM25 k1", 0.5, 3.0, 1.5, 0.1)
b = st.sidebar.slider("BM25 b", 0.0, 1.0, 0.75, 0.05)
top_k = st.sidebar.slider("Top K Results", 1, 10, 10)

query = st.text_input("Enter your query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        processed_query = preprocessing.preprocess_text(query)

        with st.spinner("Searching..."):
            base_results = run_base_search(
                search_type=search_type,
                query=query,
                processed_query=processed_query,
                k1=k1,
                b=b,
                top_k=top_k,
            )

            query_cluster = None
            if use_clustering:
                query_cluster = cluster_boosting.predict_query_cluster(query)
                display_results = cluster_boosting.apply_cluster_boost(
                    base_results,
                    query_cluster=query_cluster,
                    boost_factor=cluster_boost_factor,
                    top_k=top_k,
                )
            else:
                display_results = [
                    {
                        "doc_id": str(doc_id),
                        "original_score": float(score),
                        "final_score": float(score),
                        "doc_cluster": None,
                    }
                    for doc_id, score in base_results
                ]

        st.subheader("Top Retrieved Documents")
        st.write(f"Processed Query: `{processed_query}`")

        if use_clustering and query_cluster is not None:
            st.write(f"**Query Cluster:** {query_cluster}")
            st.caption(
                "Cluster boosting is active: documents in the same cluster as the query "
                "receive a score multiplier, then all results are re-ranked. No documents are filtered out."
            )

        for rank, result in enumerate(display_results, start=1):
            doc_id = result["doc_id"]
            text = document_db.get_document_text(doc_id)
            if text is None:
                text = "Document text not found in database."

            if use_clustering:
                doc_cluster_label = (
                    result["doc_cluster"] if result["doc_cluster"] is not None else "N/A"
                )
                title = (
                    f"Rank {rank} | Doc ID: {doc_id} | "
                    f"Original Score: {result['original_score']:.6f} | "
                    f"Final Score: {result['final_score']:.6f} | "
                    f"Cluster: {doc_cluster_label}"
                )
            else:
                title = (
                    f"Rank {rank} | Doc ID: {doc_id} | Score: {result['final_score']:.6f}"
                )

            with st.expander(title):
                st.write(text)
