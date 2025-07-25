"""MongoDB Atlas client for database operations with strong typing."""

from typing import Optional, Dict, Any, List, Protocol
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.server_api import ServerApi

from ..config import config
from ..types import YOLOAssistantError


class DatabaseError(YOLOAssistantError):
    """Raised when database operations fail."""


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""


class OperationError(DatabaseError):
    """Raised when database operations fail."""


class DatabaseConfig:
    """Configuration for database operations."""
    
    SERVER_API_VERSION: str = '1'
    PING_COMMAND: str = 'ping'


class DatabaseClient(Protocol):
    """Protocol for database client implementations."""
    
    def connect(self) -> None:
        """Establish database connection."""
        ...
        
    def disconnect(self) -> None:
        """Close database connection."""
        ...
        
    def ensure_connected(self) -> None:
        """Ensure client is connected."""
        ...


class MongoDBClient:
    """MongoDB Atlas client with strong typing and error handling."""
    
    def __init__(self, connection_string: Optional[str] = None) -> None:
        """Initialize MongoDB client.
        
        Args:
            connection_string: Optional MongoDB connection string override
            
        Raises:
            ConnectionError: If connection string is invalid
        """
        try:
            self.connection_string = connection_string or config.mongodb_uri
            self.client: Optional[MongoClient] = None
            self.db: Optional[Database] = None
            self.collection: Optional[Collection] = None
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize MongoDB client: {e}")
        
    def connect(self) -> None:
        """Establish connection to MongoDB Atlas.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            print("Connecting to MongoDB Atlas...")
            self.client = MongoClient(
                self.connection_string,
                server_api=ServerApi(DatabaseConfig.SERVER_API_VERSION)
            )
            
            # Test connection
            self.client.admin.command(DatabaseConfig.PING_COMMAND)
            print("Successfully connected to MongoDB Atlas!")
            
            # Get database and collection
            self.db = self.client[config.database_name]
            self.collection = self.db[config.collection_name]
            
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to MongoDB: {e}")
            
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                print("Disconnected from MongoDB")
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")
            
    def ensure_connected(self) -> None:
        """Ensure client is connected.
        
        Raises:
            ConnectionError: If connection fails
        """
        if not self.client:
            self.connect()
            
    def insert_one(self, document: Dict[str, Any]) -> str:
        """Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            Inserted document ID
            
        Raises:
            OperationError: If insertion fails
        """
        try:
            self.ensure_connected()
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            raise OperationError(f"Failed to insert document: {e}")
        
    def insert_many(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            List of inserted document IDs
            
        Raises:
            OperationError: If insertion fails
        """
        if not documents:
            return []
            
        try:
            self.ensure_connected()
            result = self.collection.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            raise OperationError(f"Failed to insert documents: {e}")
        
    def find_one(self, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Found document or None
            
        Raises:
            OperationError: If query fails
        """
        try:
            self.ensure_connected()
            return self.collection.find_one(filter)
            
        except Exception as e:
            raise OperationError(f"Failed to find document: {e}")
        
    def find_many(
        self,
        filter: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find multiple documents.
        
        Args:
            filter: Query filter
            limit: Maximum number of documents to return
            
        Returns:
            List of found documents
            
        Raises:
            OperationError: If query fails
        """
        try:
            self.ensure_connected()
            cursor = self.collection.find(filter)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
            
        except Exception as e:
            raise OperationError(f"Failed to find documents: {e}")
        
    def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """Update a single document.
        
        Args:
            filter: Query filter
            update: Update operations
            
        Returns:
            Number of modified documents
            
        Raises:
            OperationError: If update fails
        """
        try:
            self.ensure_connected()
            result = self.collection.update_one(filter, update)
            return result.modified_count
            
        except Exception as e:
            raise OperationError(f"Failed to update document: {e}")
        
    def delete_one(self, filter: Dict[str, Any]) -> int:
        """Delete a single document.
        
        Args:
            filter: Query filter
            
        Returns:
            Number of deleted documents
            
        Raises:
            OperationError: If deletion fails
        """
        try:
            self.ensure_connected()
            result = self.collection.delete_one(filter)
            return result.deleted_count
            
        except Exception as e:
            raise OperationError(f"Failed to delete document: {e}")
        
    def delete_many(self, filter: Dict[str, Any]) -> int:
        """Delete multiple documents.
        
        Args:
            filter: Query filter
            
        Returns:
            Number of deleted documents
            
        Raises:
            OperationError: If deletion fails
        """
        try:
            self.ensure_connected()
            result = self.collection.delete_many(filter)
            return result.deleted_count
            
        except Exception as e:
            raise OperationError(f"Failed to delete documents: {e}")
        
    def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching filter.
        
        Args:
            filter: Optional query filter
            
        Returns:
            Number of matching documents
            
        Raises:
            OperationError: If count fails
        """
        try:
            self.ensure_connected()
            filter = filter or {}
            return self.collection.count_documents(filter)
            
        except Exception as e:
            raise OperationError(f"Failed to count documents: {e}")
        
    def create_index(self, keys: Dict[str, Any], **kwargs: Any) -> str:
        """Create an index on the collection.
        
        Args:
            keys: Index specification
            **kwargs: Additional index options
            
        Returns:
            Name of created index
            
        Raises:
            OperationError: If index creation fails
        """
        try:
            self.ensure_connected()
            return self.collection.create_index(keys, **kwargs)
            
        except Exception as e:
            raise OperationError(f"Failed to create index: {e}")
        
    def drop_collection(self) -> None:
        """Drop the entire collection.
        
        Raises:
            OperationError: If collection drop fails
        """
        try:
            self.ensure_connected()
            self.collection.drop()
            print(f"Dropped collection: {config.collection_name}")
            
        except Exception as e:
            raise OperationError(f"Failed to drop collection: {e}")
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Collection statistics
            
        Raises:
            OperationError: If stats retrieval fails
        """
        try:
            self.ensure_connected()
            return self.db.command("collStats", config.collection_name)
            
        except Exception as e:
            raise OperationError(f"Failed to get collection stats: {e}")
