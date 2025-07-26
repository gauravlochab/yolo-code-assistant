"""MongoDB Atlas vector store for storing and retrieving code chunks."""

from dataclasses import asdict
from typing import List, Dict, Any, Optional, Protocol
from bson import ObjectId
from pymongo.errors import OperationFailure
import time

from ..config import config
from ..types import (
    CodeChunk,
    ChunkType,
    YOLOAssistantError,
)
from .mongodb_client import MongoDBClient


class StorageError(YOLOAssistantError):
    """Raised when vector storage operations fail."""


class VectorStoreConfig:
    """Configuration for vector store."""
    
    INDEX_NAME: str = "vector_search_index"
    INDEX_WAIT_TIME: int = 10  # seconds
    MAX_INDEX_ATTEMPTS: int = 30
    INDEX_CHECK_INTERVAL: int = 5  # seconds
    NUM_CANDIDATES_MULTIPLIER: int = 10


class VectorStore(Protocol):
    """Protocol for vector storage implementations."""
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks with embeddings."""
        ...
        
    def search_similar(
        self,
        query_embedding: List[float],
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        ...


class MongoDBVectorStore(MongoDBClient):
    """MongoDB Atlas vector store with strong typing and error handling."""
    
    def __init__(self, connection_string: Optional[str] = None) -> None:
        """Initialize vector store.
        
        Args:
            connection_string: Optional MongoDB connection string override
            
        Raises:
            StorageError: If initialization fails
        """
        try:
            super().__init__(connection_string)
            self.vector_index_name = VectorStoreConfig.INDEX_NAME
        except Exception as e:
            raise StorageError(f"Failed to initialize vector store: {e}")
        
    def ensure_vector_index(self) -> None:
        """Create vector search index for embeddings.
        
        Raises:
            StorageError: If index creation fails
        """
        try:
            self.ensure_connected()
            
            # Ensure collection exists by inserting and removing a dummy document
            if self.count_documents() == 0:
                print("Creating collection...")
                dummy_id = self.collection.insert_one({"_dummy": True}).inserted_id
                self.collection.delete_one({"_id": dummy_id})
                print("Collection created.")
            
            index_definition = {
                "name": self.vector_index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": config.embedding_dimension,
                        "similarity": "cosine"
                    }]
                }
            }
            
            # Check if index exists
            existing_indexes = list(self.collection.list_search_indexes())
            index_exists = any(
                idx.get("name") == self.vector_index_name 
                for idx in existing_indexes
            )
            
            if not index_exists:
                print(f"Creating vector search index: {self.vector_index_name}")
                self.collection.create_search_index(index_definition)
                print("Vector search index created. Waiting for it to become active...")
                
                # Wait for index to be ready
                time.sleep(VectorStoreConfig.INDEX_WAIT_TIME)
                
                # Check index status
                for attempt in range(VectorStoreConfig.MAX_INDEX_ATTEMPTS):
                    indexes = list(self.collection.list_search_indexes())
                    for idx in indexes:
                        if idx.get("name") == self.vector_index_name:
                            status = idx.get("status")
                            if status == "READY":
                                print("Vector search index is ready!")
                                return
                            print(f"Index status: {status}. Waiting...")
                    
                    time.sleep(VectorStoreConfig.INDEX_CHECK_INTERVAL)
                
                raise StorageError("Vector index did not become ready in time")
            else:
                print(f"Vector search index '{self.vector_index_name}' already exists")
                
        except Exception as e:
            raise StorageError(
                f"Failed to create vector search index: {e}\n"
                "Note: Vector search requires MongoDB Atlas M10 or higher tier"
            )
            
    def insert_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]) -> List[str]:
        """Insert code chunks with embeddings.
        
        Args:
            chunks_with_embeddings: List of chunk dictionaries with embeddings
            
        Returns:
            List of inserted document IDs
            
        Raises:
            StorageError: If insertion fails
            ValueError: If chunks are missing embeddings
        """
        if not chunks_with_embeddings:
            return []
            
        try:
            # Validate embeddings
            for chunk in chunks_with_embeddings:
                if 'embedding' not in chunk:
                    raise ValueError("Each chunk must have an 'embedding' field")
                    
            return self.insert_many(chunks_with_embeddings)
            
        except Exception as e:
            raise StorageError(f"Failed to insert chunks: {e}")
        
    def search_similar(
        self,
        query_embedding: List[float],
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar code chunks using vector search.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter: Additional filter criteria
            
        Returns:
            List of similar code chunks with scores
            
        Raises:
            StorageError: If search fails
        """
        try:
            self.ensure_connected()
            
            limit = limit or config.max_search_results
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * VectorStoreConfig.NUM_CANDIDATES_MULTIPLIER,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "search_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            if filter:
                pipeline.append({"$match": filter})
                
            pipeline.append({
                "$project": {
                    "embedding": 0  # Exclude embeddings from results
                }
            })
            
            return list(self.collection.aggregate(pipeline))
            
        except OperationFailure as e:
            print(f"Vector search failed: {e}")
            print("Falling back to regular search...")
            return self._fallback_search(limit, filter)
            
        except Exception as e:
            raise StorageError(f"Search failed: {e}")
            
    def _fallback_search(
        self,
        limit: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback search when vector search is not available.
        
        Args:
            limit: Maximum number of results
            filter: Optional filter criteria
            
        Returns:
            List of code chunks
            
        Raises:
            StorageError: If fallback search fails
        """
        try:
            filter = filter or {}
            results = self.find_many(filter, limit=limit)
            
            # Remove embeddings and add default score
            for result in results:
                result.pop('embedding', None)
                result['search_score'] = 0.5
                
            return results
            
        except Exception as e:
            raise StorageError(f"Fallback search failed: {e}")
        
    def search_by_text(
        self,
        text_query: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for code chunks by text.
        
        Args:
            text_query: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching code chunks
            
        Raises:
            StorageError: If text search fails
        """
        try:
            self.ensure_connected()
            
            limit = limit or config.max_search_results
            
            filter = {
                "$or": [
                    {"name": {"$regex": text_query, "$options": "i"}},
                    {"content": {"$regex": text_query, "$options": "i"}},
                    {"docstring": {"$regex": text_query, "$options": "i"}}
                ]
            }
            
            results = self.find_many(filter, limit=limit)
            
            for result in results:
                result.pop('embedding', None)
                result['search_score'] = 0.5
                
            return results
            
        except Exception as e:
            raise StorageError(f"Text search failed: {e}")
        
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            The chunk document or None
            
        Raises:
            StorageError: If retrieval fails
        """
        try:
            result = self.find_one({"_id": ObjectId(chunk_id)})
            if result:
                result.pop('embedding', None)
            return result
            
        except Exception as e:
            raise StorageError(f"Failed to get chunk by ID: {e}")
        
    def get_chunks_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunks from the file
            
        Raises:
            StorageError: If retrieval fails
        """
        try:
            results = self.find_many({"file_path": file_path})
            
            for result in results:
                result.pop('embedding', None)
                
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to get chunks by file: {e}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
            
        Raises:
            StorageError: If stats collection fails
        """
        try:
            self.ensure_connected()
            
            stats = {
                "total_chunks": self.count_documents(),
                "chunks_by_type": {},
                "files_indexed": 0,
                "index_status": "unknown"
            }
            
            # Count chunks by type
            for chunk_type in ChunkType:
                count = self.count_documents({"chunk_type": chunk_type.name})
                stats["chunks_by_type"][chunk_type.name] = count
                
            # Count unique files
            pipeline = [
                {"$group": {"_id": "$file_path"}},
                {"$count": "total"}
            ]
            result = list(self.collection.aggregate(pipeline))
            if result:
                stats["files_indexed"] = result[0]["total"]
                
            # Check index status
            indexes = list(self.collection.list_search_indexes())
            for idx in indexes:
                if idx.get("name") == self.vector_index_name:
                    stats["index_status"] = idx.get("status", "unknown")
                    break
                    
            return stats
            
        except Exception as e:
            raise StorageError(f"Failed to get statistics: {e}")
        
    def clear_all_chunks(self) -> int:
        """Delete all chunks from the store.
        
        Returns:
            Number of deleted chunks
            
        Raises:
            StorageError: If deletion fails
        """
        try:
            count = self.count_documents()
            self.drop_collection()
            print(f"Cleared {count} chunks from vector store")
            return count
            
        except Exception as e:
            raise StorageError(f"Failed to clear chunks: {e}")
