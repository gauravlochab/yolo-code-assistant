"""OpenRouter client for LLM response generation."""

from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

from ..config import config


class OpenRouterClient:
    """Client for interacting with OpenRouter API for LLM responses."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenRouter client.
        
        Args:
            api_key: Optional API key override
            model: Optional model override
        """
        self.api_key = api_key or config.openrouter_api_key
        self.model = model or config.llm_model
        
        # Using mistral-7b-instruct:free - highest quality completely free model on OpenRouter as of Jan 2024
        # Provides good instruction following and technical Q&A capabilities while maintaining zero cost
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/yolo-code-assistant",
                "X-Title": "YOLO Code Assistant"
            }
        )
        
    def generate_response(self, query: str, 
                         context_chunks: List[Dict[str, Any]],
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> str:
        """Generate response using retrieved code context.
        
        Args:
            query: User's question
            context_chunks: List of relevant code chunks
            max_tokens: Maximum tokens in response
            temperature: Model temperature
            
        Returns:
            Generated response text
        """
        # Format the context
        context_text = self._format_context(context_chunks)
        
        # Create the system message
        system_message = """You are a helpful AI assistant specialized in the Ultralytics YOLO codebase.
You have deep knowledge of computer vision, object detection, and the YOLO architecture.
Use the provided code context to give accurate, detailed answers about YOLO implementation and usage.
When referencing code, mention the specific file and function/class names.
If the provided context doesn't contain enough information to fully answer the question, say so."""

        # Create the user message with context
        user_message = f"""Context (relevant code from YOLO codebase):
{context_text}

Question: {query}

Please provide a helpful and accurate answer based on the code context above."""

        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
            
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format code chunks into a readable context string.
        
        Args:
            context_chunks: List of code chunks
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant code context found."
            
        formatted_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            # Create header for each chunk
            chunk_type = chunk.get('chunk_type', 'code')
            name = chunk.get('name', 'Unknown')
            file_path = chunk.get('file_path', 'Unknown file')
            start_line = chunk.get('start_line', '?')
            end_line = chunk.get('end_line', '?')
            
            header = f"\n{'='*60}\n"
            header += f"[{i}] {chunk_type.upper()}: {name}\n"
            header += f"File: {file_path} (Lines {start_line}-{end_line})\n"
            
            if chunk.get('parent_class'):
                header += f"Parent Class: {chunk['parent_class']}\n"
                
            if chunk.get('docstring'):
                header += f"Docstring: {chunk['docstring'][:200]}{'...' if len(chunk.get('docstring', '')) > 200 else ''}\n"
                
            header += f"{'='*60}\n"
            
            # Add the code content
            content = chunk.get('content', '')
            
            formatted_parts.append(header + content)
            
        return "\n".join(formatted_parts)
        
    def generate_code_example(self, task_description: str,
                             context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a code example based on the task description.
        
        Args:
            task_description: Description of what the code should do
            context_chunks: Relevant code chunks for reference
            
        Returns:
            Generated code example
        """
        context_text = self._format_context(context_chunks)
        
        system_message = """You are an expert Python developer specializing in YOLO and computer vision.
Generate clean, working code examples that follow the patterns shown in the context.
Include appropriate imports and comments explaining the code."""

        user_message = f"""Context (YOLO codebase examples):
{context_text}

Task: {task_description}

Generate a complete, working code example that accomplishes this task using YOLO.
Follow the coding patterns and conventions shown in the context."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,  # Lower temperature for code generation
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating code example: {e}")
            return f"Error generating code example: {str(e)}"
            
    def summarize_code(self, code_chunks: List[Dict[str, Any]]) -> str:
        """Generate a summary of the provided code chunks.
        
        Args:
            code_chunks: List of code chunks to summarize
            
        Returns:
            Summary text
        """
        context_text = self._format_context(code_chunks)
        
        system_message = """You are a technical documentation expert.
Summarize the provided code, explaining what it does, how it works, and its key features.
Focus on the main functionality and important implementation details."""

        user_message = f"""Please summarize the following code:

{context_text}

Provide a clear, concise summary that explains the purpose and functionality of this code."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
            
    def check_api_status(self) -> bool:
        """Check if the API is accessible and working.
        
        Returns:
            True if API is working, False otherwise
        """
        try:
            # Try a minimal API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"API check failed: {e}")
            return False
