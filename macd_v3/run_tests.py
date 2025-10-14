#!/usr/bin/env python3
"""
Test runner for MACD v3.
Run this script to execute all tests.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests using pytest."""
    print("🧪 Running MACD v3 Tests")
    print("=" * 50)
    
    # Change to the macd_v3 directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--cov=macd", 
        "--cov-report=term-missing",
        "--cov-report=html",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-cov")
        return 1

def main():
    """Main test runner."""
    return run_tests()

if __name__ == "__main__":
    sys.exit(main())
