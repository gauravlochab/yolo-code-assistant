"""Tree-sitter based Python code parser with strong typing."""

from pathlib import Path
from typing import List, Optional, Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from ..types import (
    CodeChunk,
    CodeLocation,
    ChunkType,
    Chunker,
    ChunkingError,
)


class TreeSitterParser(Chunker):
    """Parser for Python code using Tree-sitter with strong typing."""
    
    def __init__(self) -> None:
        """Initialize the Tree-sitter parser."""
        try:
            self.language = Language(tspython.language())
            self.parser = Parser()
            self.parser.set_language(self.language)
        except Exception as e:
            raise ChunkingError(f"Failed to initialize Tree-sitter parser: {e}")

    def chunk_code(self, code: str, file_path: str) -> List[CodeChunk]:
        """Parse Python code and extract semantic chunks.
        
        Args:
            code: Python source code to parse
            file_path: Path to the source file
            
        Returns:
            List of code chunks with metadata
            
        Raises:
            ChunkingError: If parsing or extraction fails
        """
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            
            imports = list(self._extract_imports(root_node, code))
            chunks = []
            
            # Extract top-level functions
            chunks.extend(self._extract_functions(
                node=root_node,
                content=code,
                file_path=file_path,
                imports=imports
            ))
            
            # Extract classes and their methods
            chunks.extend(self._extract_classes(
                node=root_node,
                content=code,
                file_path=file_path,
                imports=imports
            ))
            
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to parse {file_path}: {e}")

    def _extract_imports(self, node: Node, content: str) -> Iterator[str]:
        """Extract import statements from the AST.
        
        Args:
            node: AST node to process
            content: Source code content
            
        Yields:
            Import statements as strings
        """
        for child in node.children:
            if child.type in {'import_statement', 'import_from_statement'}:
                yield content[child.start_byte:child.end_byte].strip()
            yield from self._extract_imports(child, content)

    def _extract_functions(
        self,
        node: Node,
        content: str,
        file_path: str,
        imports: List[str],
        parent_class: Optional[str] = None
    ) -> List[CodeChunk]:
        """Extract function definitions from the AST.
        
        Args:
            node: AST node to process
            content: Source code content
            file_path: Path to source file
            imports: List of import statements
            parent_class: Name of parent class if method
            
        Returns:
            List of function/method chunks
        """
        chunks = []
        
        for child in node.children:
            if child.type == 'function_definition':
                # Skip if it's a method inside a class (handled separately)
                if parent_class is None and self._is_inside_class(child):
                    continue
                    
                name_node = child.child_by_field_name('name')
                if not name_node:
                    continue
                    
                name = content[name_node.start_byte:name_node.end_byte]
                chunk_content = content[child.start_byte:child.end_byte]
                docstring = self._extract_docstring(child, content)
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    start_col=child.start_point[1],
                    end_col=child.end_point[1]
                )
                
                chunk = CodeChunk(
                    content=chunk_content,
                    location=location,
                    chunk_type=ChunkType.METHOD if parent_class else ChunkType.FUNCTION,
                    name=name,
                    docstring=docstring,
                    parent_name=parent_class,
                    imports=imports.copy() if not parent_class else []
                )
                chunks.append(chunk)
                
            chunks.extend(self._extract_functions(
                child, content, file_path, imports, parent_class
            ))
            
        return chunks

    def _extract_classes(
        self,
        node: Node,
        content: str,
        file_path: str,
        imports: List[str]
    ) -> List[CodeChunk]:
        """Extract class definitions and their methods from the AST.
        
        Args:
            node: AST node to process
            content: Source code content
            file_path: Path to source file
            imports: List of import statements
            
        Returns:
            List of class and method chunks
        """
        chunks = []
        
        for child in node.children:
            if child.type == 'class_definition':
                name_node = child.child_by_field_name('name')
                if not name_node:
                    continue
                    
                class_name = content[name_node.start_byte:name_node.end_byte]
                class_content = content[child.start_byte:child.end_byte]
                docstring = self._extract_docstring(child, content)
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    start_col=child.start_point[1],
                    end_col=child.end_point[1]
                )
                
                class_chunk = CodeChunk(
                    content=class_content,
                    location=location,
                    chunk_type=ChunkType.CLASS,
                    name=class_name,
                    docstring=docstring,
                    imports=imports.copy()
                )
                chunks.append(class_chunk)
                
                # Extract methods
                chunks.extend(self._extract_functions(
                    child, content, file_path, imports, class_name
                ))
                
            chunks.extend(self._extract_classes(child, content, file_path, imports))
            
        return chunks

    def _extract_docstring(self, node: Node, content: str) -> Optional[str]:
        """Extract docstring from a function or class node.
        
        Args:
            node: AST node to process
            content: Source code content
            
        Returns:
            Docstring if found, None otherwise
        """
        body = node.child_by_field_name('body')
        if not body or not body.children:
            return None
            
        first_stmt = body.children[0]
        if first_stmt.type == 'expression_statement':
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == 'string':
                docstring = content[expr.start_byte:expr.end_byte]
                return docstring.strip('"""').strip("'''").strip('"').strip("'").strip()
                
        return None

    def _is_inside_class(self, node: Node) -> bool:
        """Check if a node is inside a class definition.
        
        Args:
            node: AST node to check
            
        Returns:
            True if node is inside a class, False otherwise
        """
        parent = node.parent
        while parent:
            if parent.type == 'class_definition':
                return True
            parent = parent.parent
        return False
