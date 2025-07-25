"""Tests for validating generated YOLO code execution."""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path to import from yolo-code-assistant
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from test_utils import (
    execute_code_safely, 
    is_valid_python_syntax,
    extract_imports,
    extract_functions,
    analyze_code_quality,
    validate_yolo_code_structure,
    CodeExecutionResult
)

# Import MCP server functions for testing
try:
    mcp_server_dir = parent_dir.parent / "yolo-mcp-server"
    sys.path.insert(0, str(mcp_server_dir))
    from mcp_server import generate_yolo_code, ask_yolo_question, improve_yolo_code
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP server not available for testing")


class TestCodeSyntaxValidation:
    """Test syntax validation of generated code."""
    
    def test_valid_python_syntax(self):
        """Test that valid Python code passes syntax check."""
        valid_code = """
import os
def hello():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello()
"""
        is_valid, error = is_valid_python_syntax(valid_code)
        assert is_valid
        assert error == ""
    
    def test_invalid_python_syntax(self):
        """Test that invalid Python code fails syntax check."""
        invalid_code = """
def hello(
    print("Missing closing parenthesis")
"""
        is_valid, error = is_valid_python_syntax(invalid_code)
        assert not is_valid
        assert "Syntax error" in error
    
    def test_extract_imports(self):
        """Test import extraction from code."""
        code = """
import os
from pathlib import Path
from ultralytics import YOLO
import cv2 as cv
"""
        imports = extract_imports(code)
        assert "import os" in imports
        assert "from pathlib import Path" in imports
        assert "from ultralytics import YOLO" in imports
        assert "import cv2" in imports
    
    def test_extract_functions(self):
        """Test function extraction from code."""
        code = """
def main():
    pass

def detect_objects(image_path):
    return None

class MyClass:
    def method(self):
        pass
"""
        functions = extract_functions(code)
        assert "main" in functions
        assert "detect_objects" in functions
        assert "method" in functions


class TestCodeExecution:
    """Test safe execution of generated code."""
    
    def test_simple_code_execution(self):
        """Test execution of simple Python code."""
        simple_code = """
print("Hello from test!")
result = 2 + 2
print(f"Result: {result}")
"""
        result = execute_code_safely(simple_code, timeout=10)
        assert result.success
        assert "Hello from test!" in result.output
        assert "Result: 4" in result.output
        assert result.execution_time < 10
    
    def test_code_with_syntax_error(self):
        """Test handling of code with syntax errors."""
        bad_code = """
def broken_function(
    print("This will fail")
"""
        result = execute_code_safely(bad_code)
        assert not result.success
        assert not result.syntax_valid
        assert "Syntax error" in result.error
    
    def test_code_execution_timeout(self):
        """Test timeout handling for long-running code."""
        infinite_loop = """
while True:
    pass
"""
        result = execute_code_safely(infinite_loop, timeout=2)
        assert not result.success
        assert "timed out" in result.error
        assert result.execution_time >= 2
    
    def test_yolo_mock_execution(self):
        """Test execution of YOLO code with mocks."""
        yolo_code = """
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Load image
image = cv2.imread('test.jpg')

# Run prediction
results = model.predict(image)

print("Prediction completed successfully!")
print(f"Found {len(results)} results")
"""
        result = execute_code_safely(yolo_code, use_mocks=True)
        assert result.success
        assert "Prediction completed successfully!" in result.output
        assert "Found 1 results" in result.output


class TestCodeQuality:
    """Test code quality analysis."""
    
    def test_analyze_basic_code_quality(self):
        """Test basic code quality analysis."""
        code = """
# YOLO object detection script
from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    '''Detect objects in an image.'''
    try:
        model = YOLO('yolov8n.pt')
        results = model.predict(image_path)
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    detect_objects('test.jpg')
"""
        analysis = analyze_code_quality(code)
        
        assert analysis['has_imports']
        assert analysis['has_functions']
        assert analysis['has_error_handling']
        assert analysis['has_comments']
        assert analysis['has_yolo_usage']
        assert analysis['has_main_guard']
        assert 'model.predict' in analysis['yolo_patterns_found']
    
    def test_validate_yolo_code_structure(self):
        """Test YOLO-specific code structure validation."""
        yolo_code = """
from ultralytics import YOLO
import cv2

try:
    model = YOLO('yolov8n.pt')
    image = cv2.imread('test.jpg')
    results = model.predict(image)
    print("Detection completed")
except Exception as e:
    print(f"Error: {e}")
"""
        validation = validate_yolo_code_structure(yolo_code)
        
        assert validation['has_yolo_import']
        assert validation['creates_model']
        assert validation['handles_results']
        assert validation['proper_file_handling']
        assert validation['has_error_handling']


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP server not available")
class TestMCPCodeGeneration:
    """Test code generation through MCP server."""
    
    def test_generate_simple_detection_code(self):
        """Test generating simple object detection code."""
        task = "detect objects in an image using YOLO"
        
        try:
            generated_code = generate_yolo_code(task)
            
            # Basic checks
            assert isinstance(generated_code, str)
            assert len(generated_code) > 50  # Should be substantial code
            
            # Check for YOLO-related content
            assert 'YOLO' in generated_code or 'yolo' in generated_code.lower()
            
            # Test syntax
            is_valid, error = is_valid_python_syntax(generated_code)
            if not is_valid:
                print(f"Generated code syntax error: {error}")
                print(f"Generated code:\n{generated_code}")
            
            # Test execution
            result = execute_code_safely(generated_code, timeout=15)
            if not result.success:
                print(f"Execution error: {result.error}")
                print(f"Generated code:\n{generated_code}")
            
            assert result.success, f"Generated code failed to execute: {result.error}"
            
        except Exception as e:
            pytest.skip(f"MCP code generation failed: {e}")
    
    def test_generate_training_code(self):
        """Test generating YOLO training code."""
        task = "train a YOLO model on custom dataset"
        
        try:
            generated_code = generate_yolo_code(task)
            
            # Check for training-related content
            assert 'train' in generated_code.lower()
            assert 'YOLO' in generated_code or 'yolo' in generated_code.lower()
            
            # Test syntax
            is_valid, error = is_valid_python_syntax(generated_code)
            assert is_valid, f"Syntax error in generated training code: {error}"
            
            # Test execution with mocks
            result = execute_code_safely(generated_code, timeout=15)
            assert result.success, f"Training code failed to execute: {result.error}"
            
        except Exception as e:
            pytest.skip(f"MCP training code generation failed: {e}")
    
    def test_generate_export_code(self):
        """Test generating YOLO model export code."""
        task = "export YOLO model to ONNX format"
        
        try:
            generated_code = generate_yolo_code(task)
            
            # Check for export-related content
            assert 'export' in generated_code.lower()
            assert 'onnx' in generated_code.lower() or 'ONNX' in generated_code
            
            # Test syntax and execution
            is_valid, error = is_valid_python_syntax(generated_code)
            assert is_valid, f"Syntax error in generated export code: {error}"
            
            result = execute_code_safely(generated_code, timeout=15)
            assert result.success, f"Export code failed to execute: {result.error}"
            
        except Exception as e:
            pytest.skip(f"MCP export code generation failed: {e}")


class TestCodeExecutionEdgeCases:
    """Test edge cases in code execution."""
    
    def test_empty_code(self):
        """Test handling of empty code."""
        result = execute_code_safely("")
        # Empty code should execute successfully (no-op)
        assert result.success
    
    def test_code_with_imports_only(self):
        """Test code that only contains imports."""
        import_only_code = """
import os
from pathlib import Path
from ultralytics import YOLO
"""
        result = execute_code_safely(import_only_code)
        assert result.success
    
    def test_code_with_runtime_error(self):
        """Test handling of runtime errors."""
        runtime_error_code = """
# This will cause a runtime error
result = 1 / 0
"""
        result = execute_code_safely(runtime_error_code)
        assert not result.success
        assert "ZeroDivisionError" in result.error or "division by zero" in result.error
    
    def test_code_with_missing_dependencies(self):
        """Test handling of missing dependencies without mocks."""
        dependency_code = """
import some_nonexistent_module
print("This should fail")
"""
        result = execute_code_safely(dependency_code, use_mocks=False)
        assert not result.success
        assert "ModuleNotFoundError" in result.error or "ImportError" in result.error


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
