"""Code search functionality for retrieving relevant chunks."""

from typing import List, Dict, Any, Optional
from ..config import config
from ..indexer import CodeEmbedder
from ..storage import MongoDBVectorStore


class CodeSearcher:
    """Handles code search operations using embeddings and vector similarity."""
    
    def __init__(self, vector_store: MongoDBVectorStore = None, embedder: CodeEmbedder = None):
        """Initialize the code searcher.
        
        Args:
            vector_store: Optional vector store instance
            embedder: Optional embedder instance
        """
        self.vector_store = vector_store or MongoDBVectorStore()
        self.embedder = embedder or CodeEmbedder()
        
    def search(self, query: str, limit: int = None, 
               filter: Dict[str, Any] = None,
               use_embeddings: bool = True) -> List[Dict[str, Any]]:
        """Search for code chunks matching the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filter: Additional filter criteria
            use_embeddings: Whether to use vector search
            
        Returns:
            List of matching code chunks with scores
        """
        limit = limit or config.max_search_results
        
        if use_embeddings:
            # Generate embedding for the query
            query_embedding = self.embedder.embed_text(query)
            
            # Perform vector search
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                filter=filter
            )
        else:
            # Fallback to text search
            results = self.vector_store.search_by_text(query, limit=limit)
            
        return results
        
    def search_by_function_name(self, function_name: str, 
                               limit: int = None) -> List[Dict[str, Any]]:
        """Search for functions by name.
        
        Args:
            function_name: Name of the function to search for
            limit: Maximum number of results
            
        Returns:
            List of matching function chunks
        """
        filter = {
            "chunk_type": {"$in": ["function", "method"]},
            "name": {"$regex": function_name, "$options": "i"}
        }
        
        return self.vector_store.find_many(filter, limit=limit)
        
    def search_by_class_name(self, class_name: str, 
                            include_methods: bool = True) -> List[Dict[str, Any]]:
        """Search for classes by name.
        
        Args:
            class_name: Name of the class to search for
            include_methods: Whether to include class methods
            
        Returns:
            List of matching class and optionally method chunks
        """
        if include_methods:
            filter = {
                "$or": [
                    {"chunk_type": "class", "name": {"$regex": class_name, "$options": "i"}},
                    {"chunk_type": "method", "parent_class": {"$regex": class_name, "$options": "i"}}
                ]
            }
        else:
            filter = {
                "chunk_type": "class",
                "name": {"$regex": class_name, "$options": "i"}
            }
            
        return self.vector_store.find_many(filter)
        
    def search_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunks from the file
        """
        return self.vector_store.get_chunks_by_file(file_path)
        
    def search_with_context(self, query: str, 
                           context_size: int = 2) -> List[Dict[str, Any]]:
        """Search with additional context from surrounding code.
        
        Args:
            query: Search query
            context_size: Number of surrounding chunks to include
            
        Returns:
            Search results with additional context
        """
        # First get the main results
        results = self.search(query)
        
        # For each result, try to get surrounding chunks
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()
            
            # Get chunks from the same file
            file_chunks = self.search_by_file_path(result['file_path'])
            
            # Sort by line number
            file_chunks.sort(key=lambda x: x.get('start_line', 0))
            
            # Find the current chunk
            current_idx = None
            for idx, chunk in enumerate(file_chunks):
                if chunk.get('_id') == result.get('_id'):
                    current_idx = idx
                    break
                    
            if current_idx is not None:
                # Get surrounding chunks
                start_idx = max(0, current_idx - context_size)
                end_idx = min(len(file_chunks), current_idx + context_size + 1)
                
                context_chunks = []
                for i in range(start_idx, end_idx):
                    if i != current_idx:
                        context_chunks.append({
                            'name': file_chunks[i].get('name'),
                            'chunk_type': file_chunks[i].get('chunk_type'),
                            'start_line': file_chunks[i].get('start_line'),
                            'end_line': file_chunks[i].get('end_line')
                        })
                        
                enhanced_result['context_chunks'] = context_chunks
                
            enhanced_results.append(enhanced_result)
            
        return enhanced_results
        
    def get_related_chunks(self, chunk_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get chunks related to a specific chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum number of related chunks
            
        Returns:
            List of related chunks
        """
        # Get the reference chunk
        ref_chunk = self.vector_store.get_chunk_by_id(chunk_id)
        if not ref_chunk:
            return []
            
        # Search for similar chunks
        query = f"{ref_chunk.get('name', '')} {ref_chunk.get('docstring', '')}"
        results = self.search(query, limit=limit + 1)  # +1 to exclude self
        
        # Filter out the reference chunk itself
        related = [r for r in results if str(r.get('_id')) != chunk_id]
        
        return related[:limit]
