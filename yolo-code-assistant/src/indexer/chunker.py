"""Code chunking module for creating searchable code chunks."""

from pathlib import Path
from typing import List, Optional, Dict

from .code_parser import TreeSitterParser
from ..types import (
    CodeChunk,
    CodeLocation,
    ChunkType,
    ChunkingError,
)


class CodeChunker:
    """Converts source code into searchable chunks with strong typing."""
    
    def __init__(self) -> None:
        """Initialize the code chunker."""
        self.parser = TreeSitterParser()
        
    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """Convert a Python file into code chunks.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of CodeChunk objects
            
        Raises:
            ChunkingError: If file processing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parser.chunk_code(content, str(file_path))
        except Exception as e:
            raise ChunkingError(f"Failed to chunk file {file_path}: {e}")
        
    def chunk_directory(
        self,
        directory_path: Path,
        recursive: bool = True
    ) -> List[CodeChunk]:
        """Chunk all Python files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of CodeChunk objects from all files
            
        Raises:
            ChunkingError: If directory processing fails
        """
        try:
            chunks: List[CodeChunk] = []
            pattern = "**/*.py" if recursive else "*.py"
            
            for file_path in directory_path.glob(pattern):
                # Skip __pycache__ and other generated files
                if '__pycache__' in str(file_path) or file_path.suffix == '.pyc':
                    continue
                    
                try:
                    file_chunks = self.chunk_file(file_path)
                    chunks.extend(file_chunks)
                    
                    # Create summary chunk
                    if summary := self.create_summary_chunk(file_chunks, str(file_path)):
                        chunks.append(summary)
                        
                except ChunkingError as e:
                    # Log error but continue processing other files
                    print(f"Warning: {e}")
                    continue
                    
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to process directory {directory_path}: {e}")
        
    def create_summary_chunk(
        self,
        chunks: List[CodeChunk],
        file_path: str
    ) -> Optional[CodeChunk]:
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
        functions: List[str] = []
        classes: Dict[str, Dict[str, List[str]]] = {}
        
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.FUNCTION:
                sig = self._extract_signature(chunk.content)
                if sig:
                    functions.append(f"def {chunk.name}{sig}")
                    
            elif chunk.chunk_type == ChunkType.CLASS:
                sig = self._extract_signature(chunk.content)
                if sig:
                    classes[chunk.name] = {
                        'signature': f"class {chunk.name}{sig}",
                        'methods': []
                    }
                    
            elif chunk.chunk_type == ChunkType.METHOD and chunk.parent_name:
                if chunk.parent_name in classes:
                    sig = self._extract_signature(chunk.content)
                    if sig:
                        classes[chunk.parent_name]['methods'].append(
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
            
            location = CodeLocation(
                file_path=file_path,
                start_line=0,
                end_line=0
            )
            
            return CodeChunk(
                content=summary_content,
                location=location,
                chunk_type=ChunkType.MODULE,
                name=f"Summary of {Path(file_path).name}",
                docstring=None,
                metadata={
                    'summary': True,
                    'functions': len(functions),
                    'classes': len(classes)
                }
            )
            
        return None
        
    def _extract_signature(self, code: str) -> Optional[str]:
        """Extract function/method signature from code.
        
        Args:
            code: The code content
            
        Returns:
            Optional signature string (e.g., "(self, x, y)")
        """
        try:
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
            
        except Exception:
            return None  # Fail gracefully for malformed code
