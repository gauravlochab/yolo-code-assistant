"""MongoDB Atlas client for database operations."""

from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.server_api import ServerApi

from ..config import config


class MongoDBClient:
    """MongoDB Atlas client for managing database connections and operations."""
    
    def __init__(self, connection_string: str = None):
        """Initialize MongoDB client.
        
        Args:
            connection_string: Optional MongoDB connection string override
        """
        self.connection_string = connection_string or config.mongodb_uri
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        
    def connect(self) -> None:
        """Establish connection to MongoDB Atlas."""
        try:
            print("Connecting to MongoDB Atlas...")
            # Use ServerApi version 1 for stable API
            self.client = MongoClient(self.connection_string, server_api=ServerApi('1'))
            
            # Test the connection
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            
            # Get database and collection
            self.db = self.client[config.database_name]
            self.collection = self.db[config.collection_name]
            
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error connecting to MongoDB: {e}")
            raise
            
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")
            
    def ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self.client:
            self.connect()
            
    def insert_one(self, document: Dict[str, Any]) -> str:
        """Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        self.ensure_connected()
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
        
    def insert_many(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            List of inserted document IDs
        """
        self.ensure_connected()
        if not documents:
            return []
            
        result = self.collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
        
    def find_one(self, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Found document or None
        """
        self.ensure_connected()
        return self.collection.find_one(filter)
        
    def find_many(self, filter: Dict[str, Any], limit: int = None) -> List[Dict[str, Any]]:
        """Find multiple documents.
        
        Args:
            filter: Query filter
            limit: Maximum number of documents to return
            
        Returns:
            List of found documents
        """
        self.ensure_connected()
        cursor = self.collection.find(filter)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
        
    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update a single document.
        
        Args:
            filter: Query filter
            update: Update operations
            
        Returns:
            Number of modified documents
        """
        self.ensure_connected()
        result = self.collection.update_one(filter, update)
        return result.modified_count
        
    def delete_one(self, filter: Dict[str, Any]) -> int:
        """Delete a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Number of deleted documents
        """
        self.ensure_connected()
        result = self.collection.delete_one(filter)
        return result.deleted_count
        
    def delete_many(self, filter: Dict[str, Any]) -> int:
        """Delete multiple documents.
        
        Args:
            filter: Query filter
            
        Returns:
            Number of deleted documents
        """
        self.ensure_connected()
        result = self.collection.delete_many(filter)
        return result.deleted_count
        
    def count_documents(self, filter: Dict[str, Any] = None) -> int:
        """Count documents matching filter.
        
        Args:
            filter: Optional query filter
            
        Returns:
            Number of matching documents
        """
        self.ensure_connected()
        filter = filter or {}
        return self.collection.count_documents(filter)
        
    def create_index(self, keys: Dict[str, Any], **kwargs) -> str:
        """Create an index on the collection.
        
        Args:
            keys: Index specification
            **kwargs: Additional index options
            
        Returns:
            Name of created index
        """
        self.ensure_connected()
        return self.collection.create_index(keys, **kwargs)
        
    def drop_collection(self) -> None:
        """Drop the entire collection."""
        self.ensure_connected()
        self.collection.drop()
        print(f"Dropped collection: {config.collection_name}")
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Collection statistics
        """
        self.ensure_connected()
        return self.db.command("collStats", config.collection_name)
