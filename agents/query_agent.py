from typing import Dict, List, Optional
import logging
from database.vector_store import VectorStore
from api.groq_client import GroqClient
from config.prompts import QUERY_PROMPTS

class QueryAgent:
    def __init__(self, vector_store: VectorStore, groq_client: GroqClient):
        self.vector_store = vector_store
        self.groq_client = groq_client
        self.logger = logging.getLogger(__name__)
        self.query_history = []

    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze the user's query intent using LLM."""
        try:
            prompt = QUERY_PROMPTS['intent_analysis'].format(query=query)
            response = self.groq_client.generate_response(prompt)
            
            # Parse the structured response
            intent = eval(response)  # Safe since we control the LLM prompt
            return intent
        except Exception as e:
            self.logger.error(f"Error analyzing query intent: {str(e)}")
            return {
                "type": "general",
                "filters": {},
                "sort": None
            }

    def apply_filters(self, intent: Dict) -> Dict:
        """Convert intent into vector store filters."""
        filters = {}
        
        if 'category' in intent['filters']:
            filters['category'] = intent['filters']['category']
            
        if 'price_range' in intent['filters']:
            min_price, max_price = intent['filters']['price_range']
            filters['price'] = {'$gte': min_price, '$lte': max_price}
            
        if 'brand' in intent['filters']:
            filters['brand'] = intent['filters']['brand']
        
        return filters

    def get_relevant_products(self, query: str, intent: Dict) -> List[Dict]:
        """Get relevant products based on query and intent."""
        filters = self.apply_filters(intent)
        
        try:
            results = self.vector_store.query_similar(
                query_text=query,
                filters=filters,
                n_results=5
            )
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving products: {str(e)}")
            return []

    def format_product_context(self, products: List[Dict]) -> str:
        """Format product information for LLM context."""
        context = []
        for product in products:
            context.append(f"""
Product: {product['metadata'].get('name', 'N/A')}
Category: {product['metadata'].get('category', 'N/A')}
Price: ${product['metadata'].get('price', 0):.2f}
Brand: {product['metadata'].get('brand', 'N/A')}
Likes: {product['metadata'].get('likes_count', 0)}
            """.strip())
        
        return "\n\n".join(context)

    def generate_response(self, query: str, products: List[Dict]) -> str:
        """Generate a response using the LLM."""
        context = self.format_product_context(products)
        prompt = QUERY_PROMPTS['response_generation'].format(
            context=context,
            query=query
        )
        
        try:
            response = self.groq_client.generate_response(prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."

    def process_query(self, query: str) -> Dict:
        """Process a user query and return a response."""
        try:
            # Analyze intent
            intent = self.analyze_query_intent(query)
            
            # Get relevant products
            products = self.get_relevant_products(query, intent)
            
            # Generate response
            response = self.generate_response(query, products)
            
            # Store in history
            query_result = {
                "query": query,
                "intent": intent,
                "products_found": len(products),
                "response": response
            }
            self.query_history.append(query_result)
            
            return query_result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your query."
            }

    def get_query_history(self) -> List[Dict]:
        """Return the query history."""
        return self.query_history