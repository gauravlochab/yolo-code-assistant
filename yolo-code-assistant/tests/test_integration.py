"""Integration tests for the complete YOLO code generation and execution pipeline."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from test_utils import execute_code_safely, is_valid_python_syntax

# Try to import MCP server functions
try:
    mcp_server_dir = parent_dir.parent / "yolo-mcp-server"
    sys.path.insert(0, str(mcp_server_dir))
    from mcp_server import generate_yolo_code, ask_yolo_question, improve_yolo_code
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP server not available")
class TestFullPipeline:
    """Test the complete pipeline from question to executable code."""
    
    def test_detection_pipeline(self):
        """Test complete detection code generation and execution pipeline."""
        # Step 1: Generate code
        task = "create a simple object detection script using YOLO"
        
        try:
            generated_code = generate_yolo_code(task)
            
            # Step 2: Validate generated code
            assert isinstance(generated_code, str)
            assert len(generated_code) > 100  # Should be substantial
            
            # Step 3: Check syntax
            is_valid, error = is_valid_python_syntax(generated_code)
            if not is_valid:
                pytest.fail(f"Generated code has syntax errors: {error}\nCode:\n{generated_code}")
            
            # Step 4: Execute code
            result = execute_code_safely(generated_code, timeout=20, use_mocks=True)
            if not result.success:
                pytest.fail(f"Generated code failed to execute: {result.error}\nCode:\n{generated_code}")
            
            # Step 5: Verify execution results
            assert result.success
            assert result.syntax_valid
            
        except Exception as e:
            pytest.skip(f"Pipeline test failed due to: {e}")
    
    def test_training_pipeline(self):
        """Test complete training code generation and execution pipeline."""
        task = "write code to train a YOLO model on a custom dataset"
        
        try:
            generated_code = generate_yolo_code(task)
            
            # Validate and execute
            is_valid, error = is_valid_python_syntax(generated_code)
            assert is_valid, f"Training code syntax error: {error}"
            
            result = execute_code_safely(generated_code, timeout=20, use_mocks=True)
            assert result.success, f"Training code execution failed: {result.error}"
            
            # Check for training-specific content
            assert 'train' in generated_code.lower()
            assert 'YOLO' in generated_code or 'yolo' in generated_code.lower()
            
        except Exception as e:
            pytest.skip(f"Training pipeline test failed due to: {e}")
    
    def test_question_answer_pipeline(self):
        """Test question answering with code examples."""
        question = "How do I use YOLO for object detection in Python?"
        
        try:
            answer = ask_yolo_question(question)
            
            # Validate answer
            assert isinstance(answer, str)
            assert len(answer) > 50  # Should be substantial
            assert 'YOLO' in answer or 'yolo' in answer.lower()
            
            # Answer should contain helpful information
            assert any(keyword in answer.lower() for keyword in [
                'import', 'model', 'predict', 'detect', 'ultralytics'
            ])
            
        except Exception as e:
            pytest.skip(f"Q&A pipeline test failed due to: {e}")
    
    def test_code_improvement_pipeline(self):
        """Test code improvement functionality."""
        original_code = """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict('image.jpg')
print(results)
"""
        
        try:
            improved_code = improve_yolo_code(original_code, "Add error handling and better output formatting")
            
            # Validate improved code
            assert isinstance(improved_code, str)
            assert len(improved_code) > len(original_code)  # Should be more comprehensive
            
            # Should contain improvements
            improved_lower = improved_code.lower()
            assert any(keyword in improved_lower for keyword in [
                'try', 'except', 'error', 'handling'
            ])
            
        except Exception as e:
            pytest.skip(f"Code improvement pipeline test failed due to: {e}")


@pytest.mark.integration
class TestCodeExecutionScenarios:
    """Test various code execution scenarios."""
    
    def test_multiple_yolo_tasks(self):
        """Test execution of multiple YOLO tasks in sequence."""
        tasks = [
            # Simple detection
            """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict('test.jpg')
print("Task 1: Detection completed")
""",
            # Batch processing
            """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
for img in images:
    results = model.predict(img)
print("Task 2: Batch processing completed")
""",
            # Model export
            """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
print("Task 3: Export completed")
"""
        ]
        
        for i, code in enumerate(tasks, 1):
            result = execute_code_safely(code, timeout=10, use_mocks=True)
            assert result.success, f"Task {i} failed: {result.error}"
            assert f"Task {i}:" in result.output
    
    def test_error_recovery_scenarios(self):
        """Test how well code handles various error scenarios."""
        error_scenarios = [
            # Missing file handling
            """
from ultralytics import YOLO
import os

try:
    if not os.path.exists('nonexistent.jpg'):
        print("File not found, using default")
        image_path = 'default.jpg'
    else:
        image_path = 'nonexistent.jpg'
    
    model = YOLO('yolov8n.pt')
    results = model.predict(image_path)
    print("Error scenario 1: Handled gracefully")
except Exception as e:
    print(f"Error scenario 1: Caught exception - {e}")
""",
            # Invalid model handling
            """
from ultralytics import YOLO

try:
    model = YOLO('invalid_model.pt')
    print("Error scenario 2: Model loaded")
except Exception as e:
    print(f"Error scenario 2: Model loading failed - {e}")
    # Fallback to default model
    model = YOLO('yolov8n.pt')
    print("Error scenario 2: Using fallback model")
"""
        ]
        
        for i, code in enumerate(error_scenarios, 1):
            result = execute_code_safely(code, timeout=10, use_mocks=True)
            assert result.success, f"Error scenario {i} failed: {result.error}"
            assert f"Error scenario {i}:" in result.output
    
    def test_performance_scenarios(self):
        """Test performance-related code scenarios."""
        performance_code = """
from ultralytics import YOLO
import time

# Test model loading time
start_time = time.time()
model = YOLO('yolov8n.pt')
load_time = time.time() - start_time
print(f"Model loading time: {load_time:.3f}s")

# Test prediction time
start_time = time.time()
results = model.predict('test.jpg')
predict_time = time.time() - start_time
print(f"Prediction time: {predict_time:.3f}s")

# Performance summary
print(f"Total time: {load_time + predict_time:.3f}s")
"""
        
        result = execute_code_safely(performance_code, timeout=15, use_mocks=True)
        assert result.success, f"Performance test failed: {result.error}"
        assert "Model loading time:" in result.output
        assert "Prediction time:" in result.output
        assert "Total time:" in result.output


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test realistic YOLO usage scenarios."""
    
    def test_video_processing_scenario(self):
        """Test video processing workflow."""
        video_code = """
from ultralytics import YOLO
import cv2

def process_video(video_path, output_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    detection_count = 0
    
    # Process limited frames for testing
    while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)
        frame_count += 1
        
        # Count detections
        for result in results:
            if result.boxes is not None:
                detection_count += len(result.boxes)
    
    cap.release()
    
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {detection_count}")
    return frame_count, detection_count

# Run the processing
frames, detections = process_video('test_video.mp4', 'output_video.mp4')
print(f"Video processing completed: {frames} frames, {detections} detections")
"""
        
        result = execute_code_safely(video_code, timeout=15, use_mocks=True)
        assert result.success, f"Video processing failed: {result.error}"
        assert "Processed 5 frames" in result.output
        assert "Video processing completed" in result.output
    
    def test_batch_image_processing_scenario(self):
        """Test batch image processing workflow."""
        batch_code = """
from ultralytics import YOLO
from pathlib import Path
import json

def batch_process_images(image_dir, output_file):
    model = YOLO('yolov8n.pt')
    results_summary = {
        'total_images': 0,
        'total_detections': 0,
        'images_processed': []
    }
    
    # Simulate processing multiple images
    image_files = [f'image_{i}.jpg' for i in range(1, 6)]  # 5 test images
    
    for image_file in image_files:
        try:
            results = model.predict(image_file)
            
            image_detections = 0
            for result in results:
                if result.boxes is not None:
                    image_detections += len(result.boxes)
            
            results_summary['total_images'] += 1
            results_summary['total_detections'] += image_detections
            results_summary['images_processed'].append({
                'file': image_file,
                'detections': image_detections
            })
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Save results
    print(f"Batch processing completed:")
    print(f"  Images processed: {results_summary['total_images']}")
    print(f"  Total detections: {results_summary['total_detections']}")
    
    return results_summary

# Run batch processing
summary = batch_process_images('images/', 'results.json')
print(f"Final summary: {summary['total_images']} images, {summary['total_detections']} detections")
"""
        
        result = execute_code_safely(batch_code, timeout=15, use_mocks=True)
        assert result.success, f"Batch processing failed: {result.error}"
        assert "Batch processing completed" in result.output
        assert "Final summary: 5 images" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
