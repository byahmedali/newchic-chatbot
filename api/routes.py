from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, List
import logging
from pathlib import Path
import tempfile
import shutil
from agents.schema_analyzer import SchemaAnalyzer
from agents.data_processor import DataProcessor
from agents.query_agent import QueryAgent

app = FastAPI(title="Product Catalog API")
logger = logging.getLogger(__name__)

class ProductCatalogAPI:
    def __init__(self, schema_analyzer: SchemaAnalyzer, 
                 data_processor: DataProcessor, 
                 query_agent: QueryAgent):
        self.schema_analyzer = schema_analyzer
        self.data_processor = data_processor
        self.query_agent = query_agent

    async def upload_file(self, file: UploadFile) -> Dict:
        """Handle file upload and processing."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = Path(temp_file.name)
            
            # Analyze schema
            schema_analysis = self.schema_analyzer.analyze_csv(temp_path)
            
            # Process data
            processing_result = self.data_processor.process_csv(temp_path)
            
            # Clean up
            temp_path.unlink()
            
            return {
                "schema_analysis": schema_analysis,
                "processing_result": processing_result
            }
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def query_products(self, query: str) -> Dict:
        """Handle product queries."""
        try:
            result = self.query_agent.process_query(query)
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

def create_routes(api: ProductCatalogAPI) -> FastAPI:
    """Create FastAPI routes."""
    
    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        return await api.upload_file(file)

    @app.post("/query")
    async def query_products(query: str):
        return await api.query_products(query)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app