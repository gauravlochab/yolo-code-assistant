"""Code chunking module for creating searchable code chunks."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from .code_parser import TreeSitterParser, CodeElement


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata for storage and retrieval."""
    
    content: str
    chunk_type: str  # 'function', 'class', 'method'
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    imports: List[str] = None
    decorators: List[str] = None
    
    # Additional fields for search and context
    full_context: str = ""  # Includes imports and surrounding context
    search_text: str = ""   # Optimized text for search
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return asdict(self)
        
    def get_display_name(self) -> str:
        """Get a human-readable display name for the chunk."""
        if self.chunk_type == 'method' and self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name


class CodeChunker:
    """Converts parsed code elements into searchable chunks."""
    
    def __init__(self):
        """Initialize the code chunker."""
        self.parser = TreeSitterParser()
        
    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """Convert a Python file into code chunks.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of CodeChunk objects
        """
        # Parse the file
        elements = self.parser.parse_file(file_path)
        
        # Convert elements to chunks
        chunks = []
        for element in elements:
            chunk = self._element_to_chunk(element)
            chunks.append(chunk)
            
        return chunks
        
    def chunk_directory(self, directory_path: Path, recursive: bool = True) -> List[CodeChunk]:
        """Chunk all Python files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of CodeChunk objects from all files
        """
        chunks = []
        
        # Find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(directory_path.glob(pattern))
        
        for file_path in python_files:
            # Skip __pycache__ and other generated files
            if '__pycache__' in str(file_path) or '.pyc' in str(file_path):
                continue
                
            try:
                file_chunks = self.chunk_file(file_path)
                chunks.extend(file_chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
        return chunks
        
    def _element_to_chunk(self, element: CodeElement) -> CodeChunk:
        """Convert a CodeElement to a CodeChunk with enhanced metadata.
        
        Args:
            element: CodeElement from the parser
            
        Returns:
            CodeChunk with additional search-optimized fields
        """
        # Create full context with imports
        full_context_parts = []
        
        # Add imports at the beginning
        if element.imports:
            full_context_parts.extend(element.imports)
            full_context_parts.append("")  # Empty line after imports
            
        # Add the actual code
        full_context_parts.append(element.content)
        full_context = "\n".join(full_context_parts)
        
        # Create search text (combine name, docstring, and key parts of code)
        search_text_parts = []
        
        # Add the name
        if element.parent_class:
            search_text_parts.append(f"{element.parent_class}.{element.name}")
        else:
            search_text_parts.append(element.name)
            
        # Add docstring if available
        if element.docstring:
            search_text_parts.append(element.docstring)
            
        # Add the full code content
        search_text_parts.append(element.content)
        
        # Add file path for context
        search_text_parts.append(f"File: {element.file_path}")
        
        search_text = "\n".join(search_text_parts)
        
        # Create the chunk
        chunk = CodeChunk(
            content=element.content,
            chunk_type=element.element_type,
            name=element.name,
            file_path=element.file_path,
            start_line=element.start_line,
            end_line=element.end_line,
            docstring=element.docstring,
            parent_class=element.parent_class,
            imports=element.imports,
            decorators=element.decorators,
            full_context=full_context,
            search_text=search_text
        )
        
        return chunk
        
    def create_summary_chunk(self, chunks: List[CodeChunk], file_path: str) -> Optional[CodeChunk]:
        """Create a summary chunk for a file containing all function/class signatures.
        
        Args:
            chunks: List of chunks from a file
            file_path: Path to the file
            
        Returns:
            Optional summary chunk
        """
        if not chunks:
            return None
            
        # Group chunks by type
        functions = []
        classes = {}
        
        for chunk in chunks:
            if chunk.chunk_type == 'function':
                sig = self._extract_signature(chunk.content)
                if sig:
                    functions.append(f"def {chunk.name}{sig}")
                    
            elif chunk.chunk_type == 'class':
                sig = self._extract_signature(chunk.content)
                if sig:
                    classes[chunk.name] = {
                        'signature': f"class {chunk.name}{sig}",
                        'methods': []
                    }
                    
            elif chunk.chunk_type == 'method' and chunk.parent_class:
                if chunk.parent_class in classes:
                    sig = self._extract_signature(chunk.content)
                    if sig:
                        classes[chunk.parent_class]['methods'].append(
                            f"    def {chunk.name}{sig}"
                        )
                        
        # Build summary content
        summary_parts = [f"# File Summary: {file_path}", ""]
        
        if functions:
            summary_parts.append("## Functions")
            summary_parts.extend(functions)
            summary_parts.append("")
            
        if classes:
            summary_parts.append("## Classes")
            for class_name, class_info in classes.items():
                summary_parts.append(class_info['signature'])
                if class_info['methods']:
                    summary_parts.extend(class_info['methods'])
                summary_parts.append("")
                
        if len(summary_parts) > 2:  # More than just the header
            summary_content = "\n".join(summary_parts)
            
            return CodeChunk(
                content=summary_content,
                chunk_type='summary',
                name=f"Summary of {Path(file_path).name}",
                file_path=file_path,
                start_line=0,
                end_line=0,
                full_context=summary_content,
                search_text=summary_content
            )
            
        return None
        
    def _extract_signature(self, code: str) -> Optional[str]:
        """Extract function/method signature from code.
        
        Args:
            code: The code content
            
        Returns:
            Optional signature string (e.g., "(self, x, y)")
        """
        lines = code.strip().split('\n')
        if not lines:
            return None
            
        # Find the def or class line
        for line in lines:
            line = line.strip()
            if line.startswith(('def ', 'class ')):
                # Extract the part in parentheses
                start = line.find('(')
                if start != -1:
                    # Find matching closing parenthesis
                    count = 1
                    i = start + 1
                    while i < len(line) and count > 0:
                        if line[i] == '(':
                            count += 1
                        elif line[i] == ')':
                            count -= 1
                        i += 1
                    if count == 0:
                        return line[start:i]
                elif line.endswith(':'):
                    # Class without explicit inheritance
                    return ""
                    
        return None
