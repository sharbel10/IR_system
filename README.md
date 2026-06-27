Project Structure

This project is organized in a modular way so that each part of the Information Retrieval pipeline has a clear responsibility. The system starts from data preprocessing, then builds indexes for different retrieval models, processes user queries, retrieves and ranks documents, and finally displays the results through a Streamlit interface.

IR_Project/
│
├── app.py
├── run_indexing.py
├── build_documents_db.py
├── clustering.py
│
├── services/
│   ├── preprocessing.py
│   ├── query_processor.py
│   ├── query_refinement.py
│   ├── cluster_boosting.py
│   ├── database.py
│   │
│   └── indexers/
│       ├── tfidf_indexer.py
│       ├── bm25_indexer.py
│       ├── embedding_indexer.py
│       └── hybrid_indexer.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── documents.db
│
├── evaluate_baseline.py
├── evaluate_ltr.py
├── evaluate_enhanced.py
├── evaluate_tuning.py
├── evaluate_clustering.py
│
└── test_*.py
