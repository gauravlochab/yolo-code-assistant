"""Configuration management for YOLO Code Assistant."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for YOLO Code Assistant."""

    def __init__(self):
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.ultralytics_repo_dir = self.data_dir / "ultralytics_repo"
        
        # MongoDB Configuration
        self.mongodb_uri = os.getenv("MONGODB_URI", "")
        self.database_name = "yolo_assistant"
        self.collection_name = "code_chunks"
        
        # OpenRouter Configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.llm_model = "mistralai/mistral-7b-instruct:free"
        
        # Embedding Configuration
        self.embedding_model_name = "jinaai/jina-embeddings-v2-base-code"
        self.embedding_dimension = 768
        
        # Indexing Configuration
        self.target_dirs = [
            "ultralytics/models/",
            "ultralytics/engine/",
            "ultralytics/data/"
        ]
        self.embedding_batch_size = 50  # Process embeddings in smaller batches to avoid memory issues
        
        # Retrieval Configuration
        self.max_search_results = 5
        self.similarity_threshold = 0.5
        
        # UI Configuration
        self.gradio_share = False
        self.gradio_server_name = "0.0.0.0"
        self.gradio_server_port = 7860
        
    def validate(self) -> bool:
        """Validate required configuration."""
        errors = []
        
        if not self.mongodb_uri:
            errors.append("MONGODB_URI environment variable is not set")
            
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY environment variable is not set")
            
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True


# Global config instance
config = Config()
