"""Gradio interface for YOLO Code Assistant."""

import gradio as gr
from typing import List, Tuple, Optional
import time

from ..config import config
from ..storage import MongoDBVectorStore
from ..retrieval import CodeSearcher, ResultRanker
from ..generation import OpenRouterClient
from ..indexer import CodeEmbedder


class YOLOAssistantUI:
    """Gradio interface for the YOLO Code Assistant."""
    
    def __init__(self):
        """Initialize the UI components."""
        # Initialize components
        self.vector_store = MongoDBVectorStore()
        self.embedder = CodeEmbedder()
        self.searcher = CodeSearcher(self.vector_store, self.embedder)
        self.ranker = ResultRanker()
        self.llm_client = OpenRouterClient()
        
        # Example questions
        self.example_questions = [
            "How do I train a YOLO model with custom data?",
            "What are the different YOLO model architectures available?",
            "How does the loss function work in YOLO training?",
            "How to export a YOLO model to ONNX format?",
            "What data augmentation techniques are used in YOLO?",
            "How to perform inference with a trained YOLO model?",
            "What are the hyperparameters for YOLO training?",
            "How to evaluate YOLO model performance?"
        ]
        
    def chat_function(self, message: str, history: List[List[str]]) -> str:
        """Handle chat interactions.
        
        Args:
            message: User's message
            history: Chat history
            
        Returns:
            Assistant's response
        """
        try:
            # Search for relevant code chunks
            print(f"Searching for: {message}")
            search_results = self.searcher.search(message, limit=5)
            
            # Rank and filter results
            ranked_results = self.ranker.rank_results(search_results, message)
            ranked_results = self.ranker.filter_by_threshold(ranked_results, threshold=0.3)
            
            # Enhance results with summaries
            enhanced_results = self.ranker.enhance_results_with_summary(ranked_results)
            
            # Get top results
            top_results = self.ranker.get_best_matches(enhanced_results, n=3)
            
            if not top_results:
                # Fallback to text search if vector search returns nothing
                print("No vector search results, trying text search...")
                search_results = self.searcher.search(message, use_embeddings=False, limit=5)
                ranked_results = self.ranker.rank_results(search_results, message)
                enhanced_results = self.ranker.enhance_results_with_summary(ranked_results)
                top_results = self.ranker.get_best_matches(enhanced_results, n=3)
            
            # Generate response using LLM
            print(f"Found {len(top_results)} relevant code chunks")
            response = self.llm_client.generate_response(message, top_results)
            
            # Add source references
            if top_results:
                sources = "\n\n**Sources:**\n"
                for i, result in enumerate(top_results, 1):
                    sources += f"{i}. {result.get('summary', 'Unknown source')}\n"
                response += sources
                
            return response
            
        except Exception as e:
            print(f"Error in chat function: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}\n\nPlease make sure the MongoDB connection is properly configured and the codebase has been indexed."
    
    def create_interface(self) -> gr.Interface:
        """Create and configure the Gradio interface.
        
        Returns:
            Gradio interface object
        """
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
        }
        .chat-message {
            padding: 1em;
            margin: 0.5em 0;
            border-radius: 8px;
        }
        """
        
        # Create the chat interface
        interface = gr.ChatInterface(
            fn=self.chat_function,
            title="🚀 Ultralytics YOLO Code Assistant",
            description="Ask questions about the YOLO codebase implementation and usage. I can help with training, inference, model architectures, and more!",
            examples=self.example_questions,
            theme=gr.themes.Soft(),
            css=custom_css,
            analytics_enabled=False,
            submit_btn="Ask",
        )
        
        return interface
    
    def launch(self, share: bool = None, 
               server_name: str = None,
               server_port: int = None) -> None:
        """Launch the Gradio interface.
        
        Args:
            share: Whether to create a public share link
            server_name: Server name/IP
            server_port: Server port
        """
        # Use config values if not provided
        share = share if share is not None else config.gradio_share
        server_name = server_name or config.gradio_server_name
        server_port = server_port or config.gradio_server_port
        
        # Check if components are ready
        print("Checking system status...")
        self._check_system_status()
        
        # Create and launch interface
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            favicon_path=None
        )
        
    def _check_system_status(self) -> None:
        """Check if all components are properly configured."""
        print("\n=== System Status Check ===")
        
        # Check MongoDB connection
        try:
            self.vector_store.connect()
            stats = self.vector_store.get_statistics()
            print(f"✓ MongoDB connected")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Files indexed: {stats['files_indexed']}")
            print(f"  - Index status: {stats['index_status']}")
        except Exception as e:
            print(f"✗ MongoDB connection failed: {e}")
            print("  Please check your MONGODB_URI in .env file")
            
        # Check OpenRouter API
        try:
            if self.llm_client.check_api_status():
                print(f"✓ OpenRouter API connected (Model: {config.llm_model})")
            else:
                print("✗ OpenRouter API check failed")
        except Exception as e:
            print(f"✗ OpenRouter API error: {e}")
            print("  Please check your OPENROUTER_API_KEY in .env file")
            
        # Check embedding model
        try:
            test_embedding = self.embedder.embed_text("test")
            print(f"✓ Embedding model loaded ({config.embedding_model_name})")
            print(f"  - Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"✗ Embedding model error: {e}")
            
        print("========================\n")


def main():
    """Main function to run the Gradio app."""
    # Validate configuration
    if not config.validate():
        print("\nPlease set the required environment variables in your .env file:")
        print("MONGODB_URI=your_mongodb_atlas_connection_string")
        print("OPENROUTER_API_KEY=your_openrouter_api_key")
        return
        
    # Create and launch the UI
    ui = YOLOAssistantUI()
    ui.launch()


if __name__ == "__main__":
    main()
