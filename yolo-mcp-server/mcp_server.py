"""MCP Server for YOLO Code Assistant - Code Generation Tools."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from fastmcp import FastMCP
from dotenv import load_dotenv

# Add parent directory to path to import from yolo-code-assistant
# In Docker, the structure will be different
if os.environ.get("DOCKER_ENV"):
    parent_dir = Path("/app/yolo-code-assistant")
else:
    parent_dir = Path(__file__).parent.parent / "yolo-code-assistant"
sys.path.insert(0, str(parent_dir))

# Import necessary components from the main project
from src.generation.openrouter_client import OpenRouterClient
from src.retrieval.search import CodeSearcher
from src.storage.vector_store import MongoDBVectorStore
from src.indexer.embedder import CodeEmbedder
from src.config import config

# Load environment variables
load_dotenv()

# Initialize the MCP server
mcp = FastMCP("YOLO Code Assistant")

# Initialize components (lazy loading)
_components = {
    "vector_store": None,
    "embedder": None,
    "searcher": None,
    "llm_client": None
}


def get_components():
    """Initialize and return components on first use."""
    if _components["vector_store"] is None:
        _components["vector_store"] = MongoDBVectorStore()
        _components["vector_store"].connect()
        
    if _components["embedder"] is None:
        _components["embedder"] = CodeEmbedder()
        
    if _components["searcher"] is None:
        _components["searcher"] = CodeSearcher(
            vector_store=_components["vector_store"],
            embedder=_components["embedder"]
        )
        
    if _components["llm_client"] is None:
        _components["llm_client"] = OpenRouterClient()
        
    return _components


@mcp.tool
def generate_yolo_code(task_description: str) -> str:
    """Generate complete Python code for a YOLO task.
    
    Args:
        task_description: Description of what the code should do (e.g., "detect people in a video")
        
    Returns:
        Complete, executable Python code for the task
    """
    try:
        components = get_components()
        
        # Search for relevant code examples
        search_results = components["searcher"].search(
            query=task_description,
            limit=5
        )
        
        # Generate code using the LLM
        system_prompt = """You are an expert YOLO developer. Generate clean, complete, and working Python code.
Include all necessary imports, proper error handling, and helpful comments.
Focus on practical, executable code that follows YOLO best practices."""

        prompt = f"""Task: {task_description}

Based on the YOLO codebase, generate a complete Python script that accomplishes this task.
The code should be production-ready with proper imports, error handling, and clear comments."""

        response = components["llm_client"].generate_code_example(
            task_description=prompt,
            context_chunks=search_results
        )
        
        return response
        
    except Exception as e:
        return f"Error generating code: {str(e)}\n\nPlease ensure MongoDB and OpenRouter are properly configured."


@mcp.tool
def ask_yolo_question(question: str) -> str:
    """Answer technical questions about YOLO with code examples.
    
    Args:
        question: Technical question about YOLO (e.g., "How do I train YOLO on custom dataset?")
        
    Returns:
        Detailed answer with code snippets if relevant
    """
    try:
        components = get_components()
        
        # Search for relevant documentation and code
        search_results = components["searcher"].search(
            query=question,
            limit=5
        )
        
        # Generate answer
        response = components["llm_client"].generate_response(
            query=question,
            context_chunks=search_results,
            max_tokens=1500
        )
        
        return response
        
    except Exception as e:
        return f"Error answering question: {str(e)}\n\nPlease ensure MongoDB and OpenRouter are properly configured."


@mcp.tool
def improve_yolo_code(code: str, improvement_request: str = "General improvements") -> str:
    """Improve existing YOLO code with best practices and optimizations.
    
    Args:
        code: The existing YOLO code to improve
        improvement_request: Specific improvements requested (optional)
        
    Returns:
        Improved version of the code with explanations
    """
    try:
        components = get_components()
        
        # Search for best practices and similar code patterns
        search_query = f"YOLO best practices optimization {improvement_request}"
        search_results = components["searcher"].search(
            query=search_query,
            limit=3
        )
        
        # Create improvement prompt
        system_prompt = """You are a YOLO optimization expert. Improve the provided code by:
1. Following YOLO best practices
2. Optimizing performance
3. Adding proper error handling
4. Improving code readability
5. Adding helpful comments
Explain each significant change you make."""

        prompt = f"""Original Code:
```python
{code}
```

Improvement Request: {improvement_request}

Please provide an improved version of this YOLO code with explanations for the changes."""

        # Generate improved code
        response = components["llm_client"].generate_response(
            query=prompt,
            context_chunks=search_results,
            max_tokens=2000,
            temperature=0.5
        )
        
        return response
        
    except Exception as e:
        return f"Error improving code: {str(e)}\n\nPlease ensure MongoDB and OpenRouter are properly configured."


@mcp.tool
def explain_yolo_concept(concept: str) -> str:
    """Explain YOLO concepts with practical code examples.
    
    Args:
        concept: YOLO concept to explain (e.g., "NMS", "anchor boxes", "loss function")
        
    Returns:
        Clear explanation with implementation examples
    """
    try:
        components = get_components()
        
        # Search for relevant code and documentation
        search_results = components["searcher"].search(
            query=f"YOLO {concept}",
            limit=4
        )
        
        # Generate explanation
        system_prompt = """You are a YOLO educator. Explain concepts clearly with:
1. A simple, intuitive explanation
2. Technical details when necessary
3. Practical code examples from the YOLO codebase
4. Common use cases and best practices"""

        prompt = f"""Explain the YOLO concept: {concept}

Provide a clear explanation suitable for developers, including practical code examples."""

        response = components["llm_client"].generate_response(
            query=prompt,
            context_chunks=search_results,
            max_tokens=1500
        )
        
        return response
        
    except Exception as e:
        return f"Error explaining concept: {str(e)}\n\nPlease ensure MongoDB and OpenRouter are properly configured."


# Main entry point
if __name__ == "__main__":
    # Validate configuration before starting
    if not os.getenv("MONGODB_URI"):
        print("Error: MONGODB_URI environment variable not set")
        print("Please create a .env file with your MongoDB connection string")
        sys.exit(1)
        
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please create a .env file with your OpenRouter API key")
        sys.exit(1)
    
    print("Starting YOLO Code Assistant MCP Server...")
    print("Available tools:")
    print("  - generate_yolo_code: Generate complete YOLO code for a task")
    print("  - ask_yolo_question: Answer technical YOLO questions")
    print("  - improve_yolo_code: Improve existing YOLO code")
    print("  - explain_yolo_concept: Explain YOLO concepts with examples")
    
    mcp.run()
