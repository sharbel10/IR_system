from pathlib import Path

import streamlit as st

from services.query_processor import QueryProcessor
from services.cluster_boosting import ClusterBoostingService
from services.database import DocumentDatabase

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="IR Search System", layout="wide")
st.title("Information Retrieval Search System")


@st.cache_resource
def load_services():
    query_processor = QueryProcessor()
    cluster_boosting = ClusterBoostingService()
    document_db = DocumentDatabase()
    return query_processor, cluster_boosting, document_db


query_processor, cluster_boosting, document_db = load_services()


def ui_search_type_to_mode(search_type):
    return {
        "TF-IDF": "tfidf",
        "BM25": "bm25",
        "Embedding": "embedding",
        "Hybrid Parallel": "parallel",
        "Hybrid Serial": "serial",
        "Weighted Hybrid / LTR": "ltr",
    }[search_type]


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
        "Weighted Hybrid / LTR",
    ],
)

# Weighted Hybrid / LTR: configurable per-ranker weights for the fusion step.
# Defaults favor BM25, the strongest single ranker on this dataset (tuned on validation).
ltr_weights = {"tfidf": 1.0, "bm25": 2.0, "embedding": 0.5}
if search_type == "Weighted Hybrid / LTR":
    st.sidebar.markdown("**LTR Fusion Weights**")
    ltr_weights = {
        "tfidf": st.sidebar.slider("Weight: TF-IDF", 0.0, 3.0, 1.0, 0.1),
        "bm25": st.sidebar.slider("Weight: BM25", 0.0, 3.0, 2.0, 0.1),
        "embedding": st.sidebar.slider("Weight: Embedding", 0.0, 3.0, 0.5, 0.1),
    }

apply_expansion = st.sidebar.checkbox(
    "Enable Query Expansion (Synonyms)",
    value=False,
    help="Adds WordNet synonyms to the processed query before sparse retrieval (TF-IDF, BM25, hybrid).",
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
        # Requirement 5 — raw_query is refined first (spelling correction), then preprocessed for sparse search.
        corrected_query = query_processor.refinement_service.suggest_correction(query)
        processed_query = query_processor.pipeline.preprocess_text(corrected_query)
        if apply_expansion:
            processed_query = query_processor.refinement_service.expand_with_synonyms(processed_query)

        with st.spinner("Searching..."):
            # QueryProcessor: corrected query → preprocessing → retrieval (TF-IDF/BM25 use processed_query; Embedding uses corrected raw text).
            base_results = query_processor.process_and_search(
                raw_query=query,
                search_mode=ui_search_type_to_mode(search_type),
                k1=k1,
                b=b,
                top_k=top_k,
                apply_expansion=apply_expansion,
                weights=ltr_weights,
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
        st.write(f"**Original Query:** `{query}`")
        st.write(f"**Corrected Query:** `{corrected_query}`")
        st.write(f"**Processed Query:** `{processed_query}`")

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
