from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "data" / "documents.db"
csv_path = BASE_DIR / "data" / "processed" / "documents.csv"

df = pd.read_csv(csv_path)
df["doc_id"] = df["doc_id"].astype(str)

conn = sqlite3.connect(db_path)
df[["doc_id", "text"]].to_sql("documents", conn, if_exists="replace", index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)")
conn.commit()
conn.close()

print("documents.db created successfully")