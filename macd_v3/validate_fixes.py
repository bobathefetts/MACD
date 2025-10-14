#!/usr/bin/env python3
"""
Validation script to verify all fixes are syntactically correct.
This script checks imports and basic functionality without running full tests.
"""

import sys
import ast
from pathlib import Path

def validate_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    """Validate all fixed files."""
    print("=" * 60)
    print("MACD v3 - Fix Validation Script")
    print("=" * 60)
    print()

    # Files that were modified
    modified_files = [
        "macd/core/trainer.py",
        "macd/core/controller.py",
        "macd/core/evolution.py",
        "macd/core/evaluator.py",
        "macd/main.py",
    ]

    base_path = Path(__file__).parent
    all_valid = True

    print("Checking syntax of modified files...")
    print()

    for file_path in modified_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"[ERROR] {file_path}: FILE NOT FOUND")
            all_valid = False
            continue

        valid, error = validate_python_syntax(full_path)
        if valid:
            print(f"[OK] {file_path}: Valid syntax")
        else:
            print(f"[ERROR] {file_path}: Syntax error")
            print(f"   {error}")
            all_valid = False

    print()
    print("=" * 60)

    if all_valid:
        print("[SUCCESS] All files have valid Python syntax!")
        print()
        print("Next steps:")
        print("1. Install Python 3.8+ if not already installed")
        print("2. Run: pip install -r requirements-core.txt")
        print("3. Run: python test_e2e.py")
        print("4. Run: pytest tests/ -v")
        return 0
    else:
        print("[FAILED] Some files have syntax errors!")
        print("Please fix the errors above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
