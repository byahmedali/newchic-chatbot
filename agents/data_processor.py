import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from database.vector_store import VectorStore
from database.embeddings import EmbeddingGenerator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class DataProcessor:
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        # Add more text cleaning logic as needed
        return text

    def normalize_prices(self, price: float) -> float:
        """Normalize price values."""
        try:
            return float(price)
        except (ValueError, TypeError):
            return 0.0

    def process_row(self, row: pd.Series) -> Tuple[str, Dict]:
        """Process a single row of data."""
        # Combine relevant fields for embedding
        text_fields = [
            str(row.get('name', '')),
            str(row.get('description', '')),
            str(row.get('category', '')),
            str(row.get('brand', ''))
        ]
        combined_text = ' '.join([self.clean_text(field) for field in text_fields if field])
        
        # Create metadata
        metadata = {
            'id': str(row.get('id', '')),
            'category': str(row.get('category', '')),
            'price': self.normalize_prices(row.get('current_price', 0)),
            'brand': str(row.get('brand', '')),
            'likes_count': int(row.get('likes_count', 0)),
            'is_new': bool(row.get('is_new', False))
        }
        
        return combined_text, metadata

    def process_csv(self, file_path: Path) -> Dict:
        """Process a single CSV file."""
        try:
            self.logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            
            # Process rows
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                document, metadata = self.process_row(row)
                documents.append(document)
                metadatas.append(metadata)
                ids.append(f"{file_path.stem}_{idx}")
            
            # Generate embeddings and store in vector database
            self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "file_name": file_path.name,
                "rows_processed": len(df),
                "embeddings_generated": len(documents)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: str, max_workers: int = 4) -> List[Dict]:
        """Process all CSV files in a directory."""
        directory = Path(directory_path)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file_path in directory.glob("*.csv"):
                future = executor.submit(self.process_csv, file_path)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process file: {str(e)}")
        
        return results

    def generate_statistics(self, processing_results: List[Dict]) -> Dict:
        """Generate statistics about the processed data."""
        total_rows = sum(result['rows_processed'] for result in processing_results)
        total_embeddings = sum(result['embeddings_generated'] for result in processing_results)
        
        return {
            "total_files_processed": len(processing_results),
            "total_rows_processed": total_rows,
            "total_embeddings_generated": total_embeddings,
            "files_summary": processing_results
        }