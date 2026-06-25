import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.preprocessing import PreprocessingService

def test_committee():
    service = PreprocessingService()
    
    print("--- Test 1: Real-time Text Preprocessing ---")
    sample_text = "The quick brown foxes are jumping over the lazy dogs!"
    processed = service.preprocess_text(sample_text)
    print(f"Original Text : {sample_text}")
    print(f"Processed Text: {processed}\n")
    
    print("--- Test 2: Local CSV Files Check ---")
    processed_path = BASE_DIR / 'data' / 'processed'
    files = ['documents.csv', 'queries.csv', 'qrels.csv']
    
    for f in files:
        file_path = processed_path / f
        if file_path.exists():
            df_check = pd.read_csv(file_path, nrows=3)
            print(f"✔ {f} exists locally. Rows loaded for preview: {len(df_check)}")
            print(f"Columns: {list(df_check.columns)}")
            print("-" * 30)
        else:
            print(f"{f} is missing from the processed directory.")

if __name__ == "__main__":
    test_committee()