"""Pytest configuration and shared fixtures for YOLO code testing."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from test_utils import YOLOMocks, create_mock_environment


@pytest.fixture
def yolo_mocks():
    """Provide YOLO mock objects for testing."""
    return YOLOMocks()


@pytest.fixture
def mock_environment():
    """Provide a complete mock environment for YOLO code execution."""
    return create_mock_environment()


@pytest.fixture
def sample_yolo_detection_code():
    """Provide sample YOLO detection code for testing."""
    return """
from ultralytics import YOLO
import cv2

def detect_objects(image_path, model_path='yolov8n.pt'):
    '''Detect objects in an image using YOLO.'''
    try:
        # Load model
        model = YOLO(model_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run prediction
        results = model.predict(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()
                    })
        
        return detections
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return []

if __name__ == "__main__":
    results = detect_objects('test.jpg')
    print(f"Found {len(results)} objects")
"""


@pytest.fixture
def sample_yolo_training_code():
    """Provide sample YOLO training code for testing."""
    return """
from ultralytics import YOLO
import yaml

def train_yolo_model(data_config, model_size='n', epochs=100):
    '''Train a YOLO model on custom dataset.'''
    try:
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train the model
        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='custom_yolo'
        )
        
        print(f"Training completed. Best mAP: {results.results_dict.get('metrics/mAP50', 'N/A')}")
        return results
        
    except Exception as e:
        print(f"Training error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    data_config = {
        'train': 'path/to/train/images',
        'val': 'path/to/val/images',
        'nc': 80,
        'names': ['person', 'bicycle', 'car']  # example classes
    }
    
    train_yolo_model(data_config)
"""


@pytest.fixture
def sample_yolo_export_code():
    """Provide sample YOLO export code for testing."""
    return """
from ultralytics import YOLO

def export_yolo_model(model_path, export_format='onnx'):
    '''Export YOLO model to different formats.'''
    try:
        # Load model
        model = YOLO(model_path)
        
        # Export model
        exported_path = model.export(format=export_format)
        
        print(f"Model exported successfully to: {exported_path}")
        return exported_path
        
    except Exception as e:
        print(f"Export error: {e}")
        return None

if __name__ == "__main__":
    export_yolo_model('yolov8n.pt', 'onnx')
"""


@pytest.fixture(scope="session")
def mcp_server_available():
    """Check if MCP server is available for testing."""
    try:
        mcp_server_dir = parent_dir.parent / "yolo-mcp-server"
        sys.path.insert(0, str(mcp_server_dir))
        from mcp_server import generate_yolo_code
        return True
    except ImportError:
        return False


@pytest.fixture
def temp_test_files(tmp_path):
    """Create temporary test files for YOLO testing."""
    # Create a fake image file
    test_image = tmp_path / "test.jpg"
    test_image.write_text("fake image data")
    
    # Create a fake model file
    test_model = tmp_path / "yolov8n.pt"
    test_model.write_text("fake model data")
    
    # Create a fake dataset config
    data_config = tmp_path / "data.yaml"
    data_config.write_text("""
train: path/to/train
val: path/to/val
nc: 2
names: ['person', 'car']
""")
    
    return {
        'image': str(test_image),
        'model': str(test_model),
        'data_config': str(data_config)
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "mcp: marks tests that require MCP server"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark MCP tests
        if "mcp" in item.nodeid.lower() or "TestMCP" in str(item.cls):
            item.add_marker(pytest.mark.mcp)
        
        # Mark slow tests
        if "timeout" in item.name or "long" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "TestIntegration" in str(item.cls):
            item.add_marker(pytest.mark.integration)


# Custom assertion helpers
def assert_valid_yolo_code(code):
    """Assert that code is valid YOLO code."""
    from test_utils import is_valid_python_syntax, validate_yolo_code_structure
    
    # Check syntax
    is_valid, error = is_valid_python_syntax(code)
    assert is_valid, f"Invalid Python syntax: {error}"
    
    # Check YOLO structure
    validation = validate_yolo_code_structure(code)
    assert validation['has_yolo_import'], "Code should import YOLO"
    assert validation['creates_model'], "Code should create a YOLO model"


def assert_code_executes_successfully(code, timeout=15):
    """Assert that code executes successfully with mocks."""
    from test_utils import execute_code_safely
    
    result = execute_code_safely(code, timeout=timeout, use_mocks=True)
    assert result.success, f"Code execution failed: {result.error}"
    assert result.syntax_valid, "Code should have valid syntax"


# Add custom assertions to pytest namespace
pytest.assert_valid_yolo_code = assert_valid_yolo_code
pytest.assert_code_executes_successfully = assert_code_executes_successfully
