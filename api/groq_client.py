from typing import Dict, Optional, List
import os
import logging
from groq import Groq
from config.settings import Settings

class GroqClient:
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or Settings.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key not found")
        
        self.client = Groq(api_key=self.api_key)
        self.model = Settings.MODEL_NAME

    def generate_response(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         temperature: float = Settings.TEMPERATURE) -> str:
        """Generate a response using the Groq API."""
        try:
            # Use default max_tokens if not specified
            if max_tokens is None:
                max_tokens = Settings.MAX_TOKENS

            # Ensure max_tokens doesn't exceed limit
            max_tokens = min(max_tokens, 8000)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information about products."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response from Groq: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Groq API."""
        try:
            prompt = f"Generate a numerical embedding vector for the following text. Return only the vector numbers in a Python list format: {text}"
            response = self.generate_response(prompt, max_tokens=2000)
            
            # Extract the vector from the response
            # This is a simplified approach; you might need to adjust based on actual response format
            try:
                # Remove any non-numeric characters and split into numbers
                vector = eval(response.strip())
                if isinstance(vector, list):
                    return vector
                return [0.0] * 384  # Return default vector if parsing fails
            except:
                return [0.0] * 384
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

    def batch_generate(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            try:
                response = self.generate_response(prompt, max_tokens=max_tokens)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Error in batch generation for prompt: {prompt[:50]}...")
                responses.append(str(e))
        
        return responses

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            "model": self.model,
            "max_tokens": Settings.MAX_TOKENS,
            "temperature": Settings.TEMPERATURE
        }