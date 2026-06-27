import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["IR_DATASETS_HOME"] = str(BASE_DIR / "data" / "raw")

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ir_datasets is only needed to build the corpus (build_pipeline). It is imported lazily
# there so search/evaluation scripts can import this service without that dependency.

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class PreprocessingService:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.processed_path = BASE_DIR / "data" / "processed"
        self.processed_path.mkdir(parents=True, exist_ok=True)

    # Shared preprocessing for documents and live queries — keeps term forms aligned at search time.
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""

        # Same pipeline for docs and queries: lowercase → tokenize → remove stopwords → lemmatize.
        text = text.lower()
        tokens = re.findall(r"\w+", text)

        processed_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(processed_tokens)

    def build_pipeline(self, dataset_name="beir/webis-touche2020"):
        import ir_datasets  # lazy import: only required when building the corpus

        print(f"Starting pipeline for dataset: '{dataset_name}'")
        print("Loading dataset via ir_datasets...")

        dataset = ir_datasets.load(dataset_name)

        print("Extracting Qrels...")
        qrels_df = pd.DataFrame(dataset.qrels_iter())
        qrels_df["query_id"] = qrels_df["query_id"].astype(str)
        qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

        print("Extracting and preprocessing Queries...")
        queries_df = pd.DataFrame(dataset.queries_iter())
        queries_df["query_id"] = queries_df["query_id"].astype(str)

        queries_df = queries_df[
            queries_df["query_id"].isin(qrels_df["query_id"])
        ].copy()

        # Store a preprocessed version of each dataset query for evaluation with sparse retrievers.
        queries_df["processed_query"] = queries_df["text"].apply(self.preprocess_text)

        print("Extracting and preprocessing ALL Documents...")
        docs_list = []

        for i, doc in enumerate(dataset.docs_iter(), start=1):
            doc_id = str(doc.doc_id)

            # Some datasets may use title/text fields
            title = getattr(doc, "title", "")
            text = getattr(doc, "text", "")

            original_text = f"{title} {text}".strip()
            processed_text = self.preprocess_text(original_text)

            docs_list.append(
                {
                    "doc_id": doc_id,
                    "text": original_text,
                    "processed_text": processed_text,
                }
            )

            if i % 50000 == 0:
                print(f"Processed {i} documents...")

        docs_df = pd.DataFrame(docs_list)

        print("Saving processed datasets to 'data/processed/'...")
        docs_df.to_csv(self.processed_path / "documents.csv", index=False)
        queries_df.to_csv(self.processed_path / "queries.csv", index=False)
        qrels_df.to_csv(self.processed_path / "qrels.csv", index=False)

        print("Preprocessing pipeline complete successfully!")
        print(f"Total Documents Saved: {len(docs_df)}")
        print(f"Total Queries Saved: {queries_df['query_id'].nunique()}")
        print(f"Total Qrels Queries: {qrels_df['query_id'].nunique()}")


if __name__ == "__main__":
    service = PreprocessingService()
    service.build_pipeline()