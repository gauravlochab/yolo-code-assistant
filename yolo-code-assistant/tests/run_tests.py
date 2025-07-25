"""Test runner script for YOLO code generation tests."""

import sys
import subprocess
from pathlib import Path

def run_tests(test_type="all", verbose=True):
    """Run the test suite.
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "mcp")
        verbose: Whether to run in verbose mode
    """
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add test selection based on type
    if test_type == "unit":
        cmd.extend([
            "test_code_execution.py::TestCodeSyntaxValidation",
            "test_code_execution.py::TestCodeExecution", 
            "test_code_execution.py::TestCodeQuality",
            "test_yolo_code_generation.py"
        ])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "mcp":
        cmd.extend(["-m", "mcp"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "all":
        cmd.append(".")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Change to tests directory
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(tests_dir)
        
        print(f"Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        # Run the tests
        result = subprocess.run(cmd, capture_output=False)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return result.returncode == 0
        
    finally:
        os.chdir(original_cwd)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run YOLO code generation tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "mcp", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run in quiet mode (less verbose output)"
    )
    
    args = parser.parse_args()
    
    success = run_tests(test_type=args.type, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
