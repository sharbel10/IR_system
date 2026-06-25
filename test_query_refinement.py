import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from services.query_processor import QueryProcessor

def main():
    processor = QueryProcessor()
    
    broken_query = "What are the revenues of stoks and investmnt"
    print(f"User Input with Typos : '{broken_query}'")
    
    corrected = processor.refinement_service.suggest_correction(broken_query)
    print(f"System Auto-Corrected : '{corrected}'")
    print("-" * 60)
    
    processed_base = processor.pipeline.preprocess_text(corrected)
    print(f"Standard Processed    : '{processed_base}'")
    
    expanded = processor.refinement_service.expand_with_synonyms(processed_base)
    print(f"Expanded with Synonyms: '{expanded}'")

if __name__ == "__main__":
    main()