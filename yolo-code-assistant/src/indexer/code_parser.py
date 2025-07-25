"""Tree-sitter based Python code parser."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node


@dataclass
class CodeElement:
    """Represents a code element (function, class, method) with metadata."""
    
    content: str
    element_type: str  # 'function', 'class', 'method'
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    imports: List[str] = None
    decorators: List[str] = None


class TreeSitterParser:
    """Parser for Python code using Tree-sitter."""
    
    def __init__(self):
        """Initialize the Tree-sitter parser."""
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        
    def parse_file(self, file_path: Path) -> List[CodeElement]:
        """Parse a Python file and extract code elements.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of CodeElement objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
            
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node
        
        elements = []
        imports = self._extract_imports(root_node, content)
        
        # Extract top-level functions
        functions = self._extract_functions(root_node, content, str(file_path), imports)
        elements.extend(functions)
        
        # Extract classes and their methods
        classes = self._extract_classes(root_node, content, str(file_path), imports)
        elements.extend(classes)
        
        return elements
        
    def _extract_imports(self, node: Node, content: str) -> List[str]:
        """Extract import statements from the AST."""
        imports = []
        
        def visit(node: Node):
            if node.type in ['import_statement', 'import_from_statement']:
                import_text = content[node.start_byte:node.end_byte]
                imports.append(import_text.strip())
            
            for child in node.children:
                visit(child)
                
        visit(node)
        return imports
        
    def _extract_functions(self, node: Node, content: str, file_path: str, 
                          imports: List[str], parent_class: Optional[str] = None) -> List[CodeElement]:
        """Extract function definitions from the AST."""
        functions = []
        
        def visit(node: Node):
            if node.type == 'function_definition':
                # Get function name
                name_node = node.child_by_field_name('name')
                if not name_node:
                    return
                    
                name = content[name_node.start_byte:name_node.end_byte]
                
                # Skip if it's a method inside a class (handled separately)
                if parent_class is None and self._is_inside_class(node):
                    return
                
                # Get the full function content
                func_content = content[node.start_byte:node.end_byte]
                
                # Extract docstring
                docstring = self._extract_docstring(node, content)
                
                # Extract decorators
                decorators = self._extract_decorators(node, content)
                
                # Get line numbers
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                
                element = CodeElement(
                    content=func_content,
                    element_type='method' if parent_class else 'function',
                    name=name,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                    parent_class=parent_class,
                    imports=imports.copy() if parent_class is None else [],
                    decorators=decorators
                )
                functions.append(element)
                
            for child in node.children:
                visit(child)
                
        visit(node)
        return functions
        
    def _extract_classes(self, node: Node, content: str, file_path: str, 
                        imports: List[str]) -> List[CodeElement]:
        """Extract class definitions and their methods from the AST."""
        classes = []
        
        def visit(node: Node):
            if node.type == 'class_definition':
                # Get class name
                name_node = node.child_by_field_name('name')
                if not name_node:
                    return
                    
                class_name = content[name_node.start_byte:name_node.end_byte]
                
                # Get the full class content
                class_content = content[node.start_byte:node.end_byte]
                
                # Extract docstring
                docstring = self._extract_docstring(node, content)
                
                # Extract decorators
                decorators = self._extract_decorators(node, content)
                
                # Get line numbers
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                
                # Create class element
                class_element = CodeElement(
                    content=class_content,
                    element_type='class',
                    name=class_name,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                    imports=imports.copy(),
                    decorators=decorators
                )
                classes.append(class_element)
                
                # Extract methods from the class
                methods = self._extract_functions(node, content, file_path, imports, class_name)
                classes.extend(methods)
                
            for child in node.children:
                visit(child)
                
        visit(node)
        return classes
        
    def _extract_docstring(self, node: Node, content: str) -> Optional[str]:
        """Extract docstring from a function or class node."""
        body = node.child_by_field_name('body')
        if not body or not body.children:
            return None
            
        first_stmt = body.children[0]
        if first_stmt.type == 'expression_statement':
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == 'string':
                docstring = content[expr.start_byte:expr.end_byte]
                # Remove quotes
                docstring = docstring.strip().strip('"""').strip("'''").strip('"').strip("'")
                return docstring
                
        return None
        
    def _extract_decorators(self, node: Node, content: str) -> List[str]:
        """Extract decorators from a function or class node."""
        decorators = []
        
        # Look for decorator nodes that are siblings before the definition
        parent = node.parent
        if parent:
            for i, child in enumerate(parent.children):
                if child == node:
                    # Look at previous siblings
                    for j in range(i-1, -1, -1):
                        prev = parent.children[j]
                        if prev.type == 'decorator':
                            decorator_text = content[prev.start_byte:prev.end_byte]
                            decorators.append(decorator_text.strip())
                        else:
                            break
                    break
                    
        return list(reversed(decorators))
        
    def _is_inside_class(self, node: Node) -> bool:
        """Check if a node is inside a class definition."""
        parent = node.parent
        while parent:
            if parent.type == 'class_definition':
                return True
            parent = parent.parent
        return False
