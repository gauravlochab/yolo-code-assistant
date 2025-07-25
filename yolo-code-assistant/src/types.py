"""Core type definitions for the YOLO Code Assistant.

This module contains the core data structures and protocols used throughout
the application. It provides type hints and runtime validation for key
components like code chunks, search results, and configurations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, Any, Literal
from datetime import datetime

class ChunkType(Enum):
    """Types of code chunks that can be extracted."""
    FUNCTION = auto()
    CLASS = auto()
    METHOD = auto()
    MODULE = auto()

@dataclass(frozen=True)
class CodeLocation:
    """Represents a location in source code."""
    file_path: str
    start_line: int
    end_line: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None

@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    content: str
    location: CodeLocation
    chunk_type: ChunkType
    name: str
    docstring: Optional[str] = None
    parent_name: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate chunk data on initialization."""
        if not self.content.strip():
            raise ValueError("Code chunk content cannot be empty")
        if not self.name.strip():
            raise ValueError("Code chunk name cannot be empty")

@dataclass
class SearchResult:
    """A search result with relevance score and context."""
    chunk: CodeChunk
    score: float
    context: Optional[str] = None

    def __post_init__(self):
        """Validate search result on initialization."""
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")

@dataclass
class GenerationConfig:
    """Configuration for LLM text generation."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        """Validate generation config on initialization."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top P must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")

class Chunker(Protocol):
    """Protocol for code chunking implementations."""
    def chunk_code(self, code: str, file_path: str) -> List[CodeChunk]:
        """Chunk code into semantic units.
        
        Args:
            code: Source code to chunk
            file_path: Path to source file
            
        Returns:
            List of code chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        ...

class VectorStore(Protocol):
    """Protocol for vector storage implementations."""
    def store(self, chunks: List[CodeChunk]) -> None:
        """Store code chunks in vector database.
        
        Args:
            chunks: Code chunks to store
            
        Raises:
            StorageError: If storage operation fails
        """
        ...

    def search(
        self, 
        query: str, 
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for relevant code chunks.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity score
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search operation fails
        """
        ...

class LLMClient(Protocol):
    """Protocol for LLM client implementations."""
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """Generate text using LLM.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text
            
        Raises:
            GenerationError: If text generation fails
        """
        ...

# Custom exceptions
class YOLOAssistantError(Exception):
    """Base exception for YOLO Code Assistant."""

class ChunkingError(YOLOAssistantError):
    """Raised when code chunking fails."""

class StorageError(YOLOAssistantError):
    """Raised when vector storage operations fail."""

class SearchError(YOLOAssistantError):
    """Raised when vector search fails."""

class GenerationError(YOLOAssistantError):
    """Raised when LLM text generation fails."""
