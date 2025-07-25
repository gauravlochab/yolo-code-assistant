"""MongoDB Atlas vector store for storing and retrieving code chunks."""

from typing import List, Dict, Any, Optional
from pymongo.errors import OperationFailure
import time

from ..config import config
from .mongodb_client import MongoDBClient


class MongoDBVectorStore(MongoDBClient):
    """MongoDB Atlas vector store for code chunks with vector search capabilities."""
    
    def __init__(self, connection_string: str = None):
        """Initialize vector store.
        
        Args:
            connection_string: Optional MongoDB connection string override
        """
        super().__init__(connection_string)
        self.vector_index_name = "vector_search_index"
        
    def ensure_vector_index(self) -> None:
        """Create vector search index for embeddings."""
        self.ensure_connected()
        
        # Define the vector search index
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
        
        try:
            # Check if index already exists
            existing_indexes = list(self.collection.list_search_indexes())
            index_exists = any(idx.get("name") == self.vector_index_name for idx in existing_indexes)
            
            if not index_exists:
                print(f"Creating vector search index: {self.vector_index_name}")
                self.collection.create_search_index(index_definition)
                print("Vector search index created. Waiting for it to become active...")
                
                # Wait for index to be ready
                time.sleep(10)  # Initial wait
                
                # Check index status
                max_attempts = 30
                for attempt in range(max_attempts):
                    indexes = list(self.collection.list_search_indexes())
                    for idx in indexes:
                        if idx.get("name") == self.vector_index_name:
                            status = idx.get("status")
                            if status == "READY":
                                print("Vector search index is ready!")
                                return
                            else:
                                print(f"Index status: {status}. Waiting...")
                    
                    time.sleep(5)
                
                print("Warning: Vector index may not be fully ready yet.")
            else:
                print(f"Vector search index '{self.vector_index_name}' already exists")
                
        except Exception as e:
            print(f"Error creating vector search index: {e}")
            print("Note: Vector search may require a paid MongoDB Atlas tier (M10 or higher)")
            
    def insert_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]) -> List[str]:
        """Insert code chunks with embeddings.
        
        Args:
            chunks_with_embeddings: List of chunk dictionaries with embeddings
            
        Returns:
            List of inserted document IDs
        """
        if not chunks_with_embeddings:
            return []
            
        # Ensure embeddings are present
        for chunk in chunks_with_embeddings:
            if 'embedding' not in chunk:
                raise ValueError("Each chunk must have an 'embedding' field")
                
        return self.insert_many(chunks_with_embeddings)
        
    def search_similar(self, query_embedding: List[float], 
                      limit: int = None, 
                      filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar code chunks using vector search.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter: Additional filter criteria
            
        Returns:
            List of similar code chunks with scores
        """
        self.ensure_connected()
        
        limit = limit or config.max_search_results
        
        # Construct the aggregation pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,  # Over-fetch for better results
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "search_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add filter if provided
        if filter:
            pipeline.append({"$match": filter})
            
        # Project to exclude embedding from results (save bandwidth)
        pipeline.append({
            "$project": {
                "embedding": 0
            }
        })
        
        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except OperationFailure as e:
            print(f"Vector search failed: {e}")
            print("Falling back to regular search...")
            # Fallback to regular search if vector search fails
            return self._fallback_search(limit, filter)
            
    def _fallback_search(self, limit: int, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback search when vector search is not available.
        
        Args:
            limit: Maximum number of results
            filter: Optional filter criteria
            
        Returns:
            List of code chunks
        """
        filter = filter or {}
        results = self.find_many(filter, limit=limit)
        
        # Remove embeddings from results
        for result in results:
            result.pop('embedding', None)
            result['search_score'] = 0.5  # Default score for fallback
            
        return results
        
    def search_by_text(self, text_query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for code chunks by text (without embeddings).
        
        Args:
            text_query: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching code chunks
        """
        self.ensure_connected()
        
        limit = limit or config.max_search_results
        
        # Create text search filter
        filter = {
            "$or": [
                {"name": {"$regex": text_query, "$options": "i"}},
                {"content": {"$regex": text_query, "$options": "i"}},
                {"docstring": {"$regex": text_query, "$options": "i"}}
            ]
        }
        
        results = self.find_many(filter, limit=limit)
        
        # Remove embeddings and add score
        for result in results:
            result.pop('embedding', None)
            result['search_score'] = 0.5  # Default score for text search
            
        return results
        
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            The chunk document or None
        """
        from bson import ObjectId
        
        try:
            result = self.find_one({"_id": ObjectId(chunk_id)})
            if result:
                result.pop('embedding', None)  # Remove embedding
            return result
        except Exception as e:
            print(f"Error getting chunk by ID: {e}")
            return None
            
    def get_chunks_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunks from the file
        """
        filter = {"file_path": file_path}
        results = self.find_many(filter)
        
        # Remove embeddings
        for result in results:
            result.pop('embedding', None)
            
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        self.ensure_connected()
        
        stats = {
            "total_chunks": self.count_documents(),
            "chunks_by_type": {},
            "files_indexed": 0,
            "index_status": "unknown"
        }
        
        # Count chunks by type
        for chunk_type in ['function', 'class', 'method']:
            count = self.count_documents({"chunk_type": chunk_type})
            stats["chunks_by_type"][chunk_type] = count
            
        # Count unique files
        pipeline = [
            {"$group": {"_id": "$file_path"}},
            {"$count": "total"}
        ]
        result = list(self.collection.aggregate(pipeline))
        if result:
            stats["files_indexed"] = result[0]["total"]
            
        # Check index status
        try:
            indexes = list(self.collection.list_search_indexes())
            for idx in indexes:
                if idx.get("name") == self.vector_index_name:
                    stats["index_status"] = idx.get("status", "unknown")
                    break
        except:
            pass
            
        return stats
        
    def clear_all_chunks(self) -> int:
        """Delete all chunks from the store.
        
        Returns:
            Number of deleted chunks
        """
        count = self.count_documents()
        self.drop_collection()
        print(f"Cleared {count} chunks from vector store")
        return count
