import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class DocumentDatabase:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else BASE_DIR / "data" / "documents.db"

    def get_document_text(self, doc_id):
        if not self.db_path.exists():
            return None

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT text FROM documents WHERE doc_id = ?",
                (str(doc_id),),
            ).fetchone()
        return row[0] if row else None
