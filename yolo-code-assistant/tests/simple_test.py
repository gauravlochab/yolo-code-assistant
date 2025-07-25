"""Simple test script to demonstrate YOLO code testing without pytest."""

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


def test_basic_functionality():
    """Test basic functionality of the test utilities."""
    print("🧪 Testing Basic Functionality")
    print("-" * 50)
    
    # Test 1: Syntax validation
    print("1. Testing syntax validation...")
    valid, error = is_valid_python_syntax('print("Hello World")')
    assert valid, f"Should be valid syntax: {error}"
    print("   ✅ Valid syntax test passed")
    
    invalid, error = is_valid_python_syntax('def broken_function(\n    print("missing parenthesis")')
    assert not invalid, "Should be invalid syntax"
    print("   ✅ Invalid syntax test passed")
    
    # Test 2: Code execution
    print("2. Testing code execution...")
    result = execute_code_safely('print("Test execution successful")', timeout=5)
    assert result.success, f"Execution should succeed: {result.error}"
    assert "Test execution successful" in result.output
    print("   ✅ Code execution test passed")
    
    print("✅ Basic functionality tests completed!\n")


def test_yolo_mock_execution():
    """Test YOLO code execution with mocks."""
    print("🤖 Testing YOLO Mock Execution")
    print("-" * 50)
    
    yolo_code = '''
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("test.jpg")

# Run prediction
results = model.predict(image)

print("YOLO prediction completed successfully!")
print(f"Found {len(results)} results")
'''
    
    print("1. Testing YOLO code with mocks...")
    result = execute_code_safely(yolo_code, use_mocks=True, timeout=10)
    assert result.success, f"YOLO code should execute: {result.error}"
    assert "YOLO prediction completed successfully!" in result.output
    assert "Found 1 results" in result.output
    print("   ✅ YOLO mock execution test passed")
    
    print("✅ YOLO mock tests completed!\n")


def test_code_quality_analysis():
    """Test code quality analysis features."""
    print("📊 Testing Code Quality Analysis")
    print("-" * 50)
    
    sample_code = '''
# YOLO object detection script
from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    """Detect objects in an image."""
    try:
        model = YOLO('yolov8n.pt')
        results = model.predict(image_path)
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    detect_objects('test.jpg')
'''
    
    print("1. Testing code quality analysis...")
    analysis = analyze_code_quality(sample_code)
    assert analysis['has_imports'], "Should detect imports"
    assert analysis['has_functions'], "Should detect functions"
    assert analysis['has_error_handling'], "Should detect error handling"
    assert analysis['has_yolo_usage'], "Should detect YOLO usage"
    print("   ✅ Code quality analysis test passed")
    
    print("2. Testing YOLO structure validation...")
    validation = validate_yolo_code_structure(sample_code)
    assert validation['has_yolo_import'], "Should detect YOLO import"
    assert validation['creates_model'], "Should detect model creation"
    assert validation['has_error_handling'], "Should detect error handling"
    print("   ✅ YOLO structure validation test passed")
    
    print("✅ Code quality tests completed!\n")


def test_error_handling():
    """Test error handling in code execution."""
    print("⚠️  Testing Error Handling")
    print("-" * 50)
    
    # Test syntax error handling
    print("1. Testing syntax error handling...")
    bad_code = 'def broken_function(\n    print("missing parenthesis")'
    result = execute_code_safely(bad_code)
    assert not result.success, "Should fail for syntax errors"
    assert not result.syntax_valid, "Should detect syntax error"
    print("   ✅ Syntax error handling test passed")
    
    # Test runtime error handling
    print("2. Testing runtime error handling...")
    runtime_error_code = 'result = 1 / 0'
    result = execute_code_safely(runtime_error_code)
    assert not result.success, "Should fail for runtime errors"
    assert result.syntax_valid, "Syntax should be valid"
    print("   ✅ Runtime error handling test passed")
    
    # Test timeout handling
    print("3. Testing timeout handling...")
    infinite_loop = 'while True:\n    pass'
    result = execute_code_safely(infinite_loop, timeout=2)
    assert not result.success, "Should timeout"
    assert "timed out" in result.error, "Should indicate timeout"
    print("   ✅ Timeout handling test passed")
    
    print("✅ Error handling tests completed!\n")


def test_complex_yolo_scenarios():
    """Test more complex YOLO code scenarios."""
    print("🎯 Testing Complex YOLO Scenarios")
    print("-" * 50)
    
    # Test video processing code
    print("1. Testing video processing code...")
    video_code = '''
from ultralytics import YOLO
import cv2

def process_video(video_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while frame_count < 3:  # Limit for testing
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        frame_count += 1
    
    cap.release()
    print(f"Processed {frame_count} frames")
    return frame_count

frames = process_video('test_video.mp4')
print(f"Video processing completed: {frames} frames")
'''
    
    result = execute_code_safely(video_code, use_mocks=True, timeout=10)
    assert result.success, f"Video processing should work: {result.error}"
    assert "Processed 3 frames" in result.output
    print("   ✅ Video processing test passed")
    
    # Test batch processing code
    print("2. Testing batch processing code...")
    batch_code = '''
from ultralytics import YOLO

def batch_process(image_list):
    model = YOLO('yolov8n.pt')
    results = []
    
    for image_path in image_list:
        result = model.predict(image_path)
        results.append(result)
    
    print(f"Processed {len(results)} images")
    return results

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = batch_process(images)
print(f"Batch processing completed: {len(batch_results)} results")
'''
    
    result = execute_code_safely(batch_code, use_mocks=True, timeout=10)
    assert result.success, f"Batch processing should work: {result.error}"
    assert "Processed 3 images" in result.output
    print("   ✅ Batch processing test passed")
    
    print("✅ Complex YOLO scenario tests completed!\n")


def main():
    """Run all tests."""
    print("🚀 Starting YOLO Code Testing Suite")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_yolo_mock_execution()
        test_code_quality_analysis()
        test_error_handling()
        test_complex_yolo_scenarios()
        
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("✅ The YOLO code testing framework is working correctly!")
        print("✅ Generated code can be validated for syntax and execution")
        print("✅ YOLO-specific patterns are properly detected")
        print("✅ Error handling and edge cases are covered")
        print("✅ Complex YOLO scenarios can be tested")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
