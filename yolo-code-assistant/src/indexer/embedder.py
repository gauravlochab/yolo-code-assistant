"""Code embedding module using Jina embeddings with strong typing."""

from dataclasses import asdict
from typing import List, Dict, Any, Protocol, Optional
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..config import config
from ..types import (
    CodeChunk,
    YOLOAssistantError,
)


class EmbeddingError(YOLOAssistantError):
    """Raised when embedding generation fails."""


class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 4
    NORMALIZE: bool = True
    DEVICE: str = 'cpu'  # Force CPU to avoid GPU memory issues


class Embedder(Protocol):
    """Protocol for embedding generators."""
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        ...
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


class CodeEmbedder:
    """Generate embeddings for code chunks using Jina v2 base code model."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the embedder.
        
        Args:
            model_name: Optional model name override
            
        Raises:
            EmbeddingError: If model initialization fails
        """
        try:
            self.model_name = model_name or config.embedding_model_name
            print(f"Loading embedding model: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = EmbeddingConfig.MAX_SEQUENCE_LENGTH
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embedding: NDArray = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=EmbeddingConfig.NORMALIZE,
                device=EmbeddingConfig.DEVICE
            )
            return embedding.tolist()
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
            
        Raises:
            EmbeddingError: If batch embedding fails
        """
        if not texts:
            return []
            
        try:
            embeddings: NDArray = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=EmbeddingConfig.BATCH_SIZE,
                normalize_embeddings=EmbeddingConfig.NORMALIZE,
                device=EmbeddingConfig.DEVICE
            )
            return embeddings.tolist()
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")
        
    def embed_chunk(self, chunk: CodeChunk) -> List[float]:
        """Generate embedding for a code chunk.
        
        Args:
            chunk: CodeChunk to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If chunk embedding fails
        """
        try:
            # Create search text from chunk
            search_text = self._create_search_text(chunk)
            return self.embed_text(search_text)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to embed chunk {chunk.name}: {e}")
        
    def embed_chunks(self, chunks: List[CodeChunk]) -> List[List[float]]:
        """Generate embeddings for multiple code chunks.
        
        Args:
            chunks: List of CodeChunks to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If chunks embedding fails
        """
        if not chunks:
            return []
            
        try:
            # Create search texts
            texts = [self._create_search_text(chunk) for chunk in chunks]
            return self.embed_batch(texts)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to embed chunks: {e}")
        
    def prepare_chunks_for_storage(
        self,
        chunks: List[CodeChunk]
    ) -> List[Dict[str, Any]]:
        """Prepare chunks with embeddings for storage.
        
        Args:
            chunks: List of CodeChunks
            
        Returns:
            List of dictionaries ready for MongoDB storage
            
        Raises:
            EmbeddingError: If preparation fails
        """
        try:
            # Generate embeddings
            embeddings = self.embed_chunks(chunks)
            
            # Combine chunks with embeddings
            documents = []
            for chunk, embedding in zip(chunks, embeddings):
                doc = asdict(chunk)
                doc['embedding'] = embedding
                documents.append(doc)
                
            return documents
            
        except Exception as e:
            raise EmbeddingError(f"Failed to prepare chunks for storage: {e}")
        
    def compute_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
            
        Raises:
            EmbeddingError: If similarity computation fails
        """
        try:
            # Convert to numpy arrays
            query_vec: NDArray = np.array(query_embedding)
            doc_vecs: NDArray = np.array(doc_embeddings)
            
            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities: NDArray = np.dot(doc_norms, query_norm)
            
            return similarities.tolist()
            
        except Exception as e:
            raise EmbeddingError(f"Failed to compute similarities: {e}")
            
    def _create_search_text(self, chunk: CodeChunk) -> str:
        """Create optimized search text from a code chunk.
        
        Args:
            chunk: CodeChunk to process
            
        Returns:
            Search-optimized text
        """
        parts = []
        
        # Add name with context
        if chunk.parent_name:
            parts.append(f"{chunk.parent_name}.{chunk.name}")
        else:
            parts.append(chunk.name)
            
        # Add docstring if available
        if chunk.docstring:
            parts.append(chunk.docstring)
            
        # Add code content
        parts.append(chunk.content)
        
        # Add file location
        parts.append(f"File: {chunk.location.file_path}")
        
        return "\n".join(parts)
