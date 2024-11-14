import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Dict, List, Optional
import logging
from pathlib import Path
from database.embeddings import EmbeddingGenerator

class VectorStore:
    def __init__(self, embedding_generator: EmbeddingGenerator, persist_directory: str = "./data/vectorstore"):
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB with new configuration
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="product_catalog",
                metadata={"description": "Product catalog embeddings"}
            )
            
            self.logger.info("Vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add documents to the vector store."""
        try:
            # Generate embeddings in batches
            embeddings = self.embedding_generator.batch_generate(documents)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query_similar(self, query_text: str, filters: Optional[Dict] = None, n_results: int = 5) -> List[Dict]:
        """Query similar documents from the vector store."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate(query_text)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where=filters,
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection."""
        try:
            return {
                'total_documents': self.collection.count(),
                'metadata': self.collection.get(),
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection("product_catalog")
            self.logger.info("Collection deleted successfully")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            raise