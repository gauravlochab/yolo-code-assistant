"""Utility functions for testing generated YOLO code."""

import ast
import sys
import subprocess
import tempfile
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch
import importlib.util


class CodeExecutionResult:
    """Result of code execution test."""
    
    def __init__(self, success: bool, output: str = "", error: str = "", 
                 execution_time: float = 0.0, syntax_valid: bool = True):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.syntax_valid = syntax_valid


class YOLOMocks:
    """Mock objects for YOLO dependencies."""
    
    @staticmethod
    def create_yolo_mock():
        """Create a mock YOLO model."""
        mock_yolo = Mock()
        mock_yolo.predict.return_value = [Mock(boxes=Mock(data=[]), names={0: 'person'})]
        mock_yolo.train.return_value = Mock(results=Mock())
        mock_yolo.export.return_value = "model.onnx"
        return mock_yolo
    
    @staticmethod
    def create_cv2_mock():
        """Create a mock OpenCV module."""
        mock_cv2 = Mock()
        mock_cv2.imread.return_value = Mock(shape=(640, 640, 3))
        mock_cv2.VideoCapture.return_value = Mock()
        mock_cv2.imwrite.return_value = True
        return mock_cv2
    
    @staticmethod
    def create_torch_mock():
        """Create a mock PyTorch module."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        return mock_torch


def is_valid_python_syntax(code: str) -> Tuple[bool, str]:
    """Check if code has valid Python syntax.
    
    Args:
        code: Python code string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def extract_imports(code: str) -> List[str]:
    """Extract import statements from code.
    
    Args:
        code: Python code string
        
    Returns:
        List of import statements
    """
    imports = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
    except:
        # Fallback to regex if AST parsing fails
        import_pattern = r'^(?:from\s+\S+\s+)?import\s+.+$'
        imports = re.findall(import_pattern, code, re.MULTILINE)
    
    return imports


def extract_functions(code: str) -> List[str]:
    """Extract function names from code.
    
    Args:
        code: Python code string
        
    Returns:
        List of function names
    """
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
    except:
        pass
    return functions


def validate_imports(imports: List[str]) -> Dict[str, bool]:
    """Validate that imports can be resolved.
    
    Args:
        imports: List of import statements
        
    Returns:
        Dictionary mapping import to validity
    """
    results = {}
    
    for import_stmt in imports:
        try:
            # Extract module name from import statement
            if import_stmt.startswith("from "):
                # from module import item
                parts = import_stmt.split()
                module_name = parts[1]
            else:
                # import module
                parts = import_stmt.split()
                module_name = parts[1].split('.')[0]
            
            # Try to find the module
            spec = importlib.util.find_spec(module_name)
            results[import_stmt] = spec is not None
            
        except Exception:
            results[import_stmt] = False
    
    return results


def create_mock_environment() -> Dict[str, Any]:
    """Create a mock environment for YOLO code execution.
    
    Returns:
        Dictionary of mock objects
    """
    mocks = YOLOMocks()
    
    mock_env = {
        'YOLO': mocks.create_yolo_mock,
        'cv2': mocks.create_cv2_mock(),
        'torch': mocks.create_torch_mock(),
        'numpy': Mock(),
        'PIL': Mock(),
        'matplotlib': Mock(),
    }
    
    # Add common file/path mocks
    mock_env['Path'] = Mock(return_value=Mock(exists=Mock(return_value=True)))
    mock_env['os'] = Mock(path=Mock(exists=Mock(return_value=True)))
    
    return mock_env


def execute_code_safely(code: str, timeout: int = 30, 
                       use_mocks: bool = True) -> CodeExecutionResult:
    """Execute Python code safely in an isolated environment.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        use_mocks: Whether to use mock objects for external dependencies
        
    Returns:
        CodeExecutionResult with execution details
    """
    # First check syntax
    syntax_valid, syntax_error = is_valid_python_syntax(code)
    if not syntax_valid:
        return CodeExecutionResult(
            success=False, 
            error=syntax_error, 
            syntax_valid=False
        )
    
    # Prepare code with mocks if requested
    if use_mocks:
        mock_code = _prepare_code_with_mocks(code)
    else:
        mock_code = code
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(mock_code)
        temp_file = f.name
    
    try:
        start_time = time.time()
        
        # Execute in subprocess for isolation
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(temp_file)
        )
        
        execution_time = time.time() - start_time
        
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        
        return CodeExecutionResult(
            success=success,
            output=output,
            error=error,
            execution_time=execution_time,
            syntax_valid=True
        )
        
    except subprocess.TimeoutExpired:
        return CodeExecutionResult(
            success=False,
            error=f"Code execution timed out after {timeout} seconds",
            execution_time=timeout,
            syntax_valid=True
        )
    except Exception as e:
        return CodeExecutionResult(
            success=False,
            error=f"Execution error: {str(e)}",
            syntax_valid=True
        )
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


def _prepare_code_with_mocks(code: str) -> str:
    """Prepare code with mock imports and objects.
    
    Args:
        code: Original Python code
        
    Returns:
        Modified code with mocks
    """
    mock_imports = """
# Mock imports for testing
from unittest.mock import Mock
import sys

# Create mock modules
class MockYOLO:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        return [Mock(boxes=Mock(data=[]), names={0: 'person', 1: 'car'})]
    def train(self, *args, **kwargs):
        return Mock(results=Mock())
    def export(self, *args, **kwargs):
        return "model.onnx"

class MockCV2:
    @staticmethod
    def imread(*args, **kwargs):
        return Mock(shape=(640, 640, 3))
    @staticmethod
    def imwrite(*args, **kwargs):
        return True
    @staticmethod
    def VideoCapture(*args, **kwargs):
        cap = Mock()
        cap.read.return_value = (True, Mock(shape=(640, 640, 3)))
        cap.release.return_value = None
        return cap

class MockTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False
    @staticmethod
    def device(device_name):
        return device_name

# Mock common modules
sys.modules['ultralytics'] = Mock(YOLO=MockYOLO)
sys.modules['cv2'] = MockCV2()
sys.modules['torch'] = MockTorch()
sys.modules['numpy'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['matplotlib'] = Mock()

# Override YOLO import
YOLO = MockYOLO
cv2 = MockCV2()
torch = MockTorch()

"""
    
    return mock_imports + "\n\n" + code


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """Analyze the quality of generated code.
    
    Args:
        code: Python code to analyze
        
    Returns:
        Dictionary with quality metrics
    """
    analysis = {
        'has_imports': len(extract_imports(code)) > 0,
        'has_functions': len(extract_functions(code)) > 0,
        'has_error_handling': 'try:' in code or 'except' in code,
        'has_comments': '#' in code or '"""' in code or "'''" in code,
        'line_count': len(code.split('\n')),
        'has_yolo_usage': 'YOLO' in code,
        'has_main_guard': 'if __name__' in code,
    }
    
    # Check for common YOLO patterns
    yolo_patterns = [
        'model.predict',
        'model.train',
        'model.export',
        '.detect',
        '.track',
        'results.show',
        'results.save'
    ]
    
    analysis['yolo_patterns_found'] = [
        pattern for pattern in yolo_patterns if pattern in code
    ]
    
    return analysis


def validate_yolo_code_structure(code: str) -> Dict[str, bool]:
    """Validate that code follows YOLO best practices.
    
    Args:
        code: Python code to validate
        
    Returns:
        Dictionary of validation results
    """
    validations = {
        'has_yolo_import': any('YOLO' in imp for imp in extract_imports(code)),
        'creates_model': 'YOLO(' in code,
        'handles_results': any(pattern in code for pattern in ['results', 'prediction', 'output']),
        'proper_file_handling': any(pattern in code for pattern in ['imread', 'VideoCapture', 'Path']),
        'has_error_handling': 'try:' in code and 'except' in code,
    }
    
    return validations
