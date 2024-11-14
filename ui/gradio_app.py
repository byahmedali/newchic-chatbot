import gradio as gr
from typing import Dict, List
from pathlib import Path
import pandas as pd
import logging
from agents.schema_analyzer import SchemaAnalyzer
from agents.data_processor import DataProcessor
from agents.query_agent import QueryAgent

class ProductCatalogUI:
    def __init__(self, schema_analyzer: SchemaAnalyzer, 
                 data_processor: DataProcessor, 
                 query_agent: QueryAgent):
        self.schema_analyzer = schema_analyzer
        self.data_processor = data_processor
        self.query_agent = query_agent
        self.logger = logging.getLogger(__name__)

    def process_upload(self, files: List[str]) -> str:
        """Process uploaded CSV files."""
        try:
            results = []
            for file_path in files:
                # Analyze schema
                schema_analysis = self.schema_analyzer.analyze_csv(Path(file_path))
                
                # Process data
                processing_result = self.data_processor.process_csv(Path(file_path))
                
                results.append({
                    "file": Path(file_path).name,
                    "schema": schema_analysis,
                    "processing": processing_result
                })
            
            # Format results for display
            output = "Processing Results:\n\n"
            for result in results:
                output += f"File: {result['file']}\n"
                output += f"Rows Processed: {result['processing']['rows_processed']}\n"
                output += f"Embeddings Generated: {result['processing']['embeddings_generated']}\n"
                output += f"Schema Analysis: {result['schema']['schema_description'][:500]}...\n\n"
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error processing uploads: {str(e)}")
            return f"Error processing files: {str(e)}"

    def process_query(self, query: str) -> str:
        """Process user query."""
        try:
            result = self.query_agent.process_query(query)
            return result['response']
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def show_stats(self) -> str:
        """Display current system statistics."""
        try:
            query_history = self.query_agent.get_query_history()
            
            stats = {
                "total_queries": len(query_history),
                "recent_queries": query_history[-5:] if query_history else []
            }
            
            output = "System Statistics:\n\n"
            output += f"Total Queries Processed: {stats['total_queries']}\n\n"
            output += "Recent Queries:\n"
            for query in stats['recent_queries']:
                output += f"Q: {query['query']}\n"
                output += f"A: {query['response'][:200]}...\n\n"
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return f"Error retrieving statistics: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Product Catalog Assistant") as interface:
            gr.Markdown("# Product Catalog Assistant")
            
            with gr.Tab("Process Data"):
                with gr.Row():
                    file_input = gr.File(
                        file_count="multiple",
                        label="Upload CSV Files",
                        file_types=[".csv"]
                    )
                    process_button = gr.Button("Process Files")
                    
                process_output = gr.Textbox(
                    label="Processing Results",
                    lines=10,
                    max_lines=20
                )
                
                process_button.click(
                    fn=self.process_upload,
                    inputs=[file_input],
                    outputs=[process_output]
                )
            
            with gr.Tab("Query Products"):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask about products",
                        placeholder="e.g., What are the best-selling accessories under $50?"
                    )
                    query_button = gr.Button("Ask")
                    
                response_output = gr.Textbox(
                    label="Response",
                    lines=8,
                    max_lines=15
                )
                
                query_button.click(
                    fn=self.process_query,
                    inputs=[query_input],
                    outputs=[response_output]
                )
            
            with gr.Tab("Statistics"):
                stats_button = gr.Button("Show Statistics")
                stats_output = gr.Textbox(
                    label="System Statistics",
                    lines=10,
                    max_lines=20
                )
                
                stats_button.click(
                    fn=self.show_stats,
                    inputs=[],
                    outputs=[stats_output]
                )
            
            gr.Markdown("""
            ### Usage Instructions:
            1. Start by uploading your CSV files in the "Process Data" tab
            2. Wait for processing to complete
            3. Switch to "Query Products" to ask questions about your catalog
            4. Check system statistics in the "Statistics" tab
            """)
        
        return interface