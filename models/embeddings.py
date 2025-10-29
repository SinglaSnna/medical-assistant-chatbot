import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Handles text embeddings for RAG system"""
    
    def __init__(self, model_name=None):
        """Initialize the embedding model"""
        try:
            model_name = model_name or EMBEDDING_MODEL
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading embedding model: {e}")
            raise
    
    def encode(self, texts):
        """
        Convert texts to embeddings
        
        Args:
            texts: Single string or list of strings
            
        Returns:
            numpy array of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return np.array(embeddings, dtype='float32')
        
        except Exception as e:
            print(f"✗ Error encoding texts: {e}")
            return np.array([])
    
    def get_dimension(self):
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()