"""Main entry point for YOLO Code Assistant."""

import argparse
import sys
from pathlib import Path
import time
from git import Repo
from tqdm import tqdm

from src.config import config
from src.indexer import CodeChunker, CodeEmbedder
from src.storage import MongoDBVectorStore
from src.ui import YOLOAssistantUI


def clone_ultralytics_repo():
    """Clone the Ultralytics repository if it doesn't exist."""
    repo_path = config.ultralytics_repo_dir
    
    if repo_path.exists() and (repo_path / ".git").exists():
        print(f"Ultralytics repository already exists at {repo_path}")
        return
    
    print("Cloning Ultralytics repository...")
    repo_url = "https://github.com/ultralytics/ultralytics.git"
    
    try:
        Repo.clone_from(repo_url, repo_path)
        print(f"Successfully cloned repository to {repo_path}")
    except Exception as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)


def index_codebase():
    """Index the Ultralytics codebase."""
    print("\n=== Starting Codebase Indexing ===")
    
    # Ensure repository is cloned
    clone_ultralytics_repo()
    
    # Initialize components
    print("\nInitializing components...")
    chunker = CodeChunker()
    embedder = CodeEmbedder()
    vector_store = MongoDBVectorStore()
    
    # Connect to MongoDB and ensure index
    print("\nConnecting to MongoDB...")
    vector_store.connect()
    vector_store.ensure_vector_index()
    
    # Clear existing chunks (optional)
    print("\nClearing existing chunks...")
    vector_store.clear_all_chunks()
    
    # Process each target directory
    all_chunks = []
    
    for target_dir in config.target_dirs:
        dir_path = config.ultralytics_repo_dir / target_dir
        
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist, skipping...")
            continue
            
        print(f"\nProcessing directory: {target_dir}")
        
        # Chunk the directory
        chunks = chunker.chunk_directory(dir_path, recursive=True)
        print(f"  Found {len(chunks)} code chunks")
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks found: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks found to index!")
        return
    
    # Process chunks in batches to avoid memory issues
    batch_size = config.embedding_batch_size
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    total_stored = 0
    
    print(f"\nGenerating embeddings and storing chunks in {total_batches} batches...")
    print(f"Batch size: {batch_size} chunks per batch")
    
    for i in range(0, len(all_chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch_chunks = all_chunks[i:i + batch_size]
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
        
        # Generate embeddings for this batch
        documents = embedder.prepare_chunks_for_storage(batch_chunks)
        
        # Store this batch in MongoDB
        chunk_ids = vector_store.insert_chunks(documents)
        total_stored += len(chunk_ids)
        
        print(f"  Stored {len(chunk_ids)} chunks (Total: {total_stored}/{len(all_chunks)})")
    
    # Get statistics
    stats = vector_store.get_statistics()
    print("\n=== Indexing Complete ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Files indexed: {stats['files_indexed']}")
    print("Chunks by type:")
    for chunk_type, count in stats['chunks_by_type'].items():
        print(f"  - {chunk_type}: {count}")


def serve_app():
    """Launch the Gradio web interface."""
    print("\n=== Starting YOLO Code Assistant ===")
    
    # Validate configuration
    if not config.validate():
        print("\nPlease set the required environment variables in your .env file:")
        print("MONGODB_URI=your_mongodb_atlas_connection_string")
        print("OPENROUTER_API_KEY=your_openrouter_api_key")
        sys.exit(1)
    
    # Create and launch UI
    ui = YOLOAssistantUI()
    
    print("\nStarting Gradio interface...")
    print(f"Server will be available at: http://localhost:{config.gradio_server_port}")
    
    ui.launch()


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO Code Assistant - A RAG-based code assistant"
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the Ultralytics codebase"
    )
    
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the web interface"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check system status and configuration"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --index    # Index the codebase")
        print("  python main.py --serve    # Launch the web interface")
        print("  python main.py --check    # Check system status")
        sys.exit(0)
    
    # Execute requested actions
    if args.check:
        check_system_status()
    
    if args.index:
        index_codebase()
    
    if args.serve:
        serve_app()


def check_system_status():
    """Check system configuration and status."""
    print("\n=== System Configuration Check ===")
    
    print("\nEnvironment Variables:")
    print(f"  MONGODB_URI: {'✓ Set' if config.mongodb_uri else '✗ Not set'}")
    print(f"  OPENROUTER_API_KEY: {'✓ Set' if config.openrouter_api_key else '✗ Not set'}")
    
    print("\nConfiguration:")
    print(f"  Database: {config.database_name}")
    print(f"  Collection: {config.collection_name}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Embedding Model: {config.embedding_model_name}")
    print(f"  Target Directories: {', '.join(config.target_dirs)}")
    
    print("\nTesting connections...")
    
    # Test MongoDB
    try:
        vector_store = MongoDBVectorStore()
        vector_store.connect()
        stats = vector_store.get_statistics()
        print(f"  ✓ MongoDB connected")
        print(f"    - Total chunks: {stats['total_chunks']}")
        print(f"    - Index status: {stats['index_status']}")
    except Exception as e:
        print(f"  ✗ MongoDB error: {e}")
    
    # Test OpenRouter
    try:
        from src.generation import OpenRouterClient
        client = OpenRouterClient()
        if client.check_api_status():
            print(f"  ✓ OpenRouter API connected")
        else:
            print(f"  ✗ OpenRouter API failed")
    except Exception as e:
        print(f"  ✗ OpenRouter error: {e}")
    
    # Test embedding model
    try:
        embedder = CodeEmbedder()
        test_emb = embedder.embed_text("test")
        print(f"  ✓ Embedding model loaded")
        print(f"    - Dimension: {len(test_emb)}")
    except Exception as e:
        print(f"  ✗ Embedding model error: {e}")
    
    print("\n=== Check Complete ===")


if __name__ == "__main__":
    main()
