"""Tests for YOLO-specific code generation scenarios."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from test_utils import (
    execute_code_safely,
    is_valid_python_syntax,
    analyze_code_quality,
    validate_yolo_code_structure
)


class TestYOLOCodePatterns:
    """Test common YOLO code patterns and structures."""
    
    def test_sample_detection_code(self, sample_yolo_detection_code):
        """Test sample YOLO detection code execution."""
        code = sample_yolo_detection_code
        
        # Validate syntax
        is_valid, error = is_valid_python_syntax(code)
        assert is_valid, f"Syntax error: {error}"
        
        # Validate YOLO structure
        validation = validate_yolo_code_structure(code)
        assert validation['has_yolo_import']
        assert validation['creates_model']
        assert validation['handles_results']
        assert validation['has_error_handling']
        
        # Test execution
        result = execute_code_safely(code, timeout=10)
        assert result.success, f"Execution failed: {result.error}"
        assert "Found 0 objects" in result.output
    
    def test_sample_training_code(self, sample_yolo_training_code):
        """Test sample YOLO training code execution."""
        code = sample_yolo_training_code
        
        # Validate syntax and structure
        is_valid, error = is_valid_python_syntax(code)
        assert is_valid, f"Syntax error: {error}"
        
        validation = validate_yolo_code_structure(code)
        assert validation['has_yolo_import']
        assert validation['creates_model']
        
        # Check for training-specific patterns
        assert 'train' in code.lower()
        assert 'epochs' in code.lower()
        
        # Test execution
        result = execute_code_safely(code, timeout=10)
        assert result.success, f"Training code execution failed: {result.error}"
    
    def test_sample_export_code(self, sample_yolo_export_code):
        """Test sample YOLO export code execution."""
        code = sample_yolo_export_code
        
        # Validate syntax and structure
        is_valid, error = is_valid_python_syntax(code)
        assert is_valid, f"Syntax error: {error}"
        
        validation = validate_yolo_code_structure(code)
        assert validation['has_yolo_import']
        assert validation['creates_model']
        
        # Check for export-specific patterns
        assert 'export' in code.lower()
        assert 'onnx' in code.lower()
        
        # Test execution
        result = execute_code_safely(code, timeout=10)
        assert result.success, f"Export code execution failed: {result.error}"


class TestYOLOCodeQuality:
    """Test quality metrics for YOLO code."""
    
    def test_detection_code_quality(self, sample_yolo_detection_code):
        """Test quality metrics for detection code."""
        analysis = analyze_code_quality(sample_yolo_detection_code)
        
        assert analysis['has_imports']
        assert analysis['has_functions']
        assert analysis['has_error_handling']
        assert analysis['has_comments']
        assert analysis['has_yolo_usage']
        assert analysis['has_main_guard']
        
        # Check for YOLO-specific patterns
        expected_patterns = ['model.predict']
        for pattern in expected_patterns:
            assert pattern in analysis['yolo_patterns_found']
    
    def test_training_code_quality(self, sample_yolo_training_code):
        """Test quality metrics for training code."""
        analysis = analyze_code_quality(sample_yolo_training_code)
        
        assert analysis['has_imports']
        assert analysis['has_functions']
        assert analysis['has_error_handling']
        assert analysis['has_yolo_usage']
        
        # Training code should be substantial
        assert analysis['line_count'] > 20
    
    def test_export_code_quality(self, sample_yolo_export_code):
        """Test quality metrics for export code."""
        analysis = analyze_code_quality(sample_yolo_export_code)
        
        assert analysis['has_imports']
        assert analysis['has_functions']
        assert analysis['has_error_handling']
        assert analysis['has_yolo_usage']


class TestYOLOCodeVariations:
    """Test different variations of YOLO code."""
    
    def test_video_detection_code(self):
        """Test YOLO video detection code."""
        video_code = """
from ultralytics import YOLO
import cv2

def detect_in_video(video_path, model_path='yolov8n.pt'):
    '''Detect objects in a video using YOLO.'''
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame)
            frame_count += 1
            
            if frame_count > 10:  # Limit for testing
                break
        
        cap.release()
        print(f"Processed {frame_count} frames")
        return frame_count
        
    except Exception as e:
        print(f"Video detection error: {e}")
        return 0

if __name__ == "__main__":
    detect_in_video('test_video.mp4')
"""
        
        # Test syntax and execution
        is_valid, error = is_valid_python_syntax(video_code)
        assert is_valid, f"Syntax error: {error}"
        
        result = execute_code_safely(video_code, timeout=10)
        assert result.success, f"Video detection code failed: {result.error}"
        assert "Processed" in result.output
    
    def test_batch_detection_code(self):
        """Test YOLO batch detection code."""
        batch_code = """
from ultralytics import YOLO
import os
from pathlib import Path

def batch_detect(image_dir, model_path='yolov8n.pt'):
    '''Detect objects in multiple images.'''
    try:
        model = YOLO(model_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        processed = 0
        for ext in image_extensions:
            # Simulate finding images
            for i in range(3):  # Simulate 3 images per extension
                fake_path = f"image_{i}{ext}"
                results = model.predict(fake_path)
                processed += 1
        
        print(f"Processed {processed} images")
        return processed
        
    except Exception as e:
        print(f"Batch detection error: {e}")
        return 0

if __name__ == "__main__":
    batch_detect('images/')
"""
        
        # Test syntax and execution
        is_valid, error = is_valid_python_syntax(batch_code)
        assert is_valid, f"Syntax error: {error}"
        
        result = execute_code_safely(batch_code, timeout=10)
        assert result.success, f"Batch detection code failed: {result.error}"
        assert "Processed 12 images" in result.output
    
    def test_custom_model_code(self):
        """Test YOLO custom model loading code."""
        custom_model_code = """
from ultralytics import YOLO
import torch

def load_custom_model(model_path, device='cpu'):
    '''Load a custom YOLO model.'''
    try:
        # Set device
        device = torch.device(device)
        
        # Load model
        model = YOLO(model_path)
        model.to(device)
        
        # Get model info
        model_info = {
            'device': str(device),
            'model_path': model_path,
            'loaded': True
        }
        
        print(f"Model loaded on {device}")
        print(f"Model path: {model_path}")
        
        return model_info
        
    except Exception as e:
        print(f"Model loading error: {e}")
        return {'loaded': False, 'error': str(e)}

if __name__ == "__main__":
    info = load_custom_model('custom_model.pt')
    print(f"Model loaded: {info['loaded']}")
"""
        
        # Test syntax and execution
        is_valid, error = is_valid_python_syntax(custom_model_code)
        assert is_valid, f"Syntax error: {error}"
        
        result = execute_code_safely(custom_model_code, timeout=10)
        assert result.success, f"Custom model code failed: {result.error}"
        assert "Model loaded on cpu" in result.output
        assert "Model loaded: True" in result.output


class TestYOLOErrorHandling:
    """Test error handling in YOLO code."""
    
    def test_missing_file_handling(self):
        """Test handling of missing files in YOLO code."""
        error_handling_code = """
from ultralytics import YOLO
import cv2
import os

def robust_detection(image_path, model_path='yolov8n.pt'):
    '''Detect objects with proper error handling.'''
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, using default")
            model_path = 'yolov8n.pt'
        
        model = YOLO(model_path)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        results = model.predict(image)
        print("Detection completed successfully")
        return results
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    result = robust_detection('nonexistent.jpg')
    print(f"Result: {result}")
"""
        
        # Test syntax and execution
        is_valid, error = is_valid_python_syntax(error_handling_code)
        assert is_valid, f"Syntax error: {error}"
        
        result = execute_code_safely(error_handling_code, timeout=10)
        assert result.success, f"Error handling code failed: {result.error}"
        # Should handle missing file gracefully
        assert "Error: Image file" in result.output or "Detection completed" in result.output


class TestYOLOCodeComplexity:
    """Test different complexity levels of YOLO code."""
    
    def test_simple_yolo_code(self):
        """Test simple YOLO code execution."""
        simple_code = """
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict('test.jpg')
print("Simple detection completed")
"""
        
        result = execute_code_safely(simple_code, timeout=5)
        assert result.success
        assert "Simple detection completed" in result.output
    
    def test_intermediate_yolo_code(self):
        """Test intermediate complexity YOLO code."""
        intermediate_code = """
from ultralytics import YOLO
import cv2

def detect_and_save(input_path, output_path, conf_threshold=0.5):
    model = YOLO('yolov8n.pt')
    
    # Load and process image
    image = cv2.imread(input_path)
    results = model.predict(image, conf=conf_threshold)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detections.append({
                    'confidence': float(box.conf),
                    'class': int(box.cls)
                })
    
    print(f"Found {len(detections)} detections")
    return detections

detect_and_save('input.jpg', 'output.jpg')
"""
        
        result = execute_code_safely(intermediate_code, timeout=10)
        assert result.success
        assert "Found 0 detections" in result.output
    
    def test_complex_yolo_code(self):
        """Test complex YOLO code with multiple features."""
        complex_code = """
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.detection_count = 0
    
    def detect_objects(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            results = self.model.predict(image, conf=self.conf_threshold)
            detections = self._process_results(results)
            self.detection_count += len(detections)
            
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _process_results(self, results):
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({
                        'confidence': float(box.conf),
                        'class': int(box.cls),
                        'bbox': box.xyxy.tolist() if hasattr(box, 'xyxy') else []
                    })
        return detections
    
    def get_stats(self):
        return {'total_detections': self.detection_count}

# Usage
detector = YOLODetector()
results = detector.detect_objects('test.jpg')
stats = detector.get_stats()
print(f"Detection stats: {stats}")
"""
        
        result = execute_code_safely(complex_code, timeout=15)
        assert result.success
        assert "Detection stats:" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
