import os
import logging
import logging.config
from pathlib import Path
from typing import Dict, Optional

from database.embeddings import EmbeddingGenerator
from database.vector_store import VectorStore
from api.groq_client import GroqClient
from agents.schema_analyzer import SchemaAnalyzer
from agents.data_processor import DataProcessor
from agents.query_agent import QueryAgent
from ui.gradio_app import ProductCatalogUI
from config.settings import Settings
from dotenv import load_dotenv

class ProductCatalogSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Create necessary directories
        Settings.create_directories()
        
        # Initialize basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize components
            self.initialize_components()
            self.logger.info("System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {str(e)}")
            raise

    def initialize_components(self):
        """Initialize all system components."""
        # Initialize Groq client first
        self.groq_client = GroqClient()
        
        # Initialize embedding generator with Groq client
        self.embedding_generator = EmbeddingGenerator(groq_client=self.groq_client)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,
            persist_directory=str(Settings.VECTOR_STORE_DIR)
        )
        
        # Initialize agents
        self.schema_analyzer = SchemaAnalyzer(self.groq_client)
        self.data_processor = DataProcessor(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        self.query_agent = QueryAgent(
            vector_store=self.vector_store,
            groq_client=self.groq_client
        )
        
        # Initialize UI
        self.ui = ProductCatalogUI(
            schema_analyzer=self.schema_analyzer,
            data_processor=self.data_processor,
            query_agent=self.query_agent
        )

    def process_initial_data(self, directory_path: Optional[str] = None) -> None:
        """Process initial data if provided."""
        if directory_path:
            try:
                self.logger.info(f"Processing initial data from {directory_path}")
                results = self.data_processor.process_directory(directory_path)
                self.logger.info(f"Processed {len(results)} files from initial data")
            except Exception as e:
                self.logger.error(f"Error processing initial data: {str(e)}")
                raise

    def start_ui(self, share: bool = False) -> None:
        """Start the Gradio interface."""
        try:
            self.logger.info("Starting Gradio interface")
            interface = self.ui.create_interface()
            interface.launch(
                server_name=Settings.API_HOST,
                server_port=Settings.API_PORT,
                share=share
            )
        except Exception as e:
            self.logger.error(f"Error starting UI: {str(e)}")
            raise

    def run(self, initial_data_dir: Optional[str] = None, share_ui: bool = False) -> None:
        """Run the complete system."""
        try:
            # Process initial data if provided
            if initial_data_dir:
                self.process_initial_data(initial_data_dir)
            
            # Start the UI
            self.start_ui(share=share_ui)
            
        except Exception as e:
            self.logger.error(f"Error running system: {str(e)}")
            raise

def main():
    try:
        # Initialize system
        system = ProductCatalogSystem()
        
        # Get initial data directory from environment
        initial_data_dir = os.getenv("INITIAL_DATA_DIR")
        
        # Run system
        system.run(
            initial_data_dir=initial_data_dir,
            share_ui=True
        )
        
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    main()