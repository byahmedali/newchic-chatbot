import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from api.groq_client import GroqClient
from config.prompts import SCHEMA_ANALYSIS_PROMPT

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    sample_values: list
    null_percentage: float
    unique_values: int
    description: str

class SchemaAnalyzer:
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.logger = logging.getLogger(__name__)

    def analyze_column(self, df: pd.DataFrame, column: str) -> ColumnInfo:
        """Analyze a single column from the DataFrame."""
        series = df[column]
        return ColumnInfo(
            name=column,
            data_type=str(series.dtype),
            sample_values=series.head(3).tolist(),
            null_percentage=(series.isna().sum() / len(series)) * 100,
            unique_values=series.nunique(),
            description=self._generate_column_description(series)
        )

    def _generate_column_description(self, series: pd.Series) -> str:
        """Generate a description for a column using LLM."""
        sample_data = series.head(5).tolist()
        prompt = f"Describe this data column with samples: {sample_data}"
        response = self.groq_client.generate_response(prompt)
        return response

    def analyze_csv(self, file_path: Path) -> Dict:
        """Analyze the schema of a CSV file."""
        try:
            self.logger.info(f"Analyzing schema for {file_path}")
            df = pd.read_csv(file_path)
            
            # Analyze each column
            columns = {}
            for column in df.columns:
                columns[column] = self.analyze_column(df, column)
            
            # Generate overall schema description
            schema_description = self._generate_schema_description(columns)
            
            return {
                "file_name": file_path.name,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": {
                    col: {
                        "type": info.data_type,
                        "samples": info.sample_values,
                        "null_percentage": info.null_percentage,
                        "unique_values": info.unique_values,
                        "description": info.description
                    }
                    for col, info in columns.items()
                },
                "schema_description": schema_description
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            raise

    def _generate_schema_description(self, columns: Dict[str, ColumnInfo]) -> str:
        """Generate an overall description of the schema using LLM."""
        schema_info = {
            col: {
                "type": info.data_type,
                "samples": info.sample_values,
                "description": info.description
            }
            for col, info in columns.items()
        }
        
        prompt = SCHEMA_ANALYSIS_PROMPT.format(schema=json.dumps(schema_info, indent=2))
        response = self.groq_client.generate_response(prompt)
        return response

    def analyze_directory(self, directory_path: str) -> Dict[str, Dict]:
        """Analyze all CSV files in a directory."""
        directory = Path(directory_path)
        results = {}
        
        for file_path in directory.glob("*.csv"):
            try:
                results[file_path.name] = self.analyze_csv(file_path)
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {str(e)}")
                results[file_path.name] = {"error": str(e)}
        
        return results

    def get_schema_summary(self, schema_analysis: Dict) -> str:
        """Generate a human-readable summary of the schema analysis."""
        return f"""
        File: {schema_analysis['file_name']}
        Total Rows: {schema_analysis['total_rows']}
        Total Columns: {schema_analysis['total_columns']}
        
        Schema Description:
        {schema_analysis['schema_description']}
        
        Columns:
        {json.dumps(schema_analysis['columns'], indent=2)}
        """