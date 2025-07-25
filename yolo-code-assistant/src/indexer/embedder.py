"""Code embedding module using Jina embeddings."""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import config
from .chunker import CodeChunk


class CodeEmbedder:
    """Generate embeddings for code chunks using Jina v2 base code model."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedder.
        
        Args:
            model_name: Optional model name override
        """
        self.model_name = model_name or config.embedding_model_name
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        # Set max sequence length to prevent memory overflow
        self.model.max_seq_length = 512  # Limit to 512 tokens
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
            
        # Use smaller batch size and truncate to avoid memory issues
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=4,  # Further reduced to avoid memory overflow
            normalize_embeddings=True,
            device='cpu'  # Force CPU to avoid GPU memory issues
        )
        return embeddings.tolist()
        
    def embed_chunk(self, chunk: CodeChunk) -> List[float]:
        """Generate embedding for a code chunk.
        
        Args:
            chunk: CodeChunk to embed
            
        Returns:
            Embedding vector
        """
        # Use the search_text field which is optimized for search
        return self.embed_text(chunk.search_text)
        
    def embed_chunks(self, chunks: List[CodeChunk]) -> List[List[float]]:
        """Generate embeddings for multiple code chunks.
        
        Args:
            chunks: List of CodeChunks to embed
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
            
        # Extract search texts from chunks
        texts = [chunk.search_text for chunk in chunks]
        return self.embed_batch(texts)
        
    def prepare_chunks_for_storage(self, chunks: List[CodeChunk]) -> List[Dict[str, Any]]:
        """Prepare chunks with embeddings for storage.
        
        Args:
            chunks: List of CodeChunks
            
        Returns:
            List of dictionaries ready for MongoDB storage
        """
        # Generate embeddings
        embeddings = self.embed_chunks(chunks)
        
        # Combine chunks with embeddings
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = chunk.to_dict()
            doc['embedding'] = embedding
            documents.append(doc)
            
        return documents
        
    def compute_similarity(self, query_embedding: List[float], 
                         doc_embeddings: List[List[float]]) -> List[float]:
        """Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities.tolist()
