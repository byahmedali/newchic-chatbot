SCHEMA_ANALYSIS_PROMPT = """
Analyze the following database schema and provide insights:
{schema}

Please describe:
1. The purpose and structure of this dataset
2. Relationships between different columns
3. Data quality considerations
4. Potential use cases for this data
"""

QUERY_PROMPTS = {
    'intent_analysis': """
    Analyze the following user query and extract the intent and parameters:
    Query: {query}
    
    Return a Python dictionary with:
    {
        'type': str,  # 'search', 'compare', 'analyze'
        'filters': {
            'category': Optional[str],
            'price_range': Optional[tuple],
            'brand': Optional[str]
        },
        'sort': Optional[str]
    }
    """,
    
    'response_generation': """
    Based on the following product catalog data:
    {context}
    
    User Query: {query}
    
    Please provide a helpful response that:
    1. Directly answers the user's question
    2. Includes specific product details
    3. Mentions prices and availability
    4. Suggests related items if relevant
    """,
    
    'product_analysis': """
    Analyze the following product details and provide insights:
    {product_data}
    
    Consider:
    1. Key features and benefits
    2. Price positioning
    3. Customer sentiment (based on likes)
    4. Comparable products
    """
}

ERROR_MESSAGES = {
    'file_processing': "Error processing file: {error}",
    'query_processing': "Error processing query: {error}",
    'embedding_generation': "Error generating embeddings: {error}",
    'api_error': "API Error: {error}"
}