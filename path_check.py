import sys
import os
from pathlib import Path
import importlib.util
import pandas as pd

def check_python_paths():
    """
    Check and display all Python paths and module accessibility.
    """
    print("\n=== Python Path Information ===")
    print(f"Current Working Directory: {os.getcwd()}")
    
    # List all paths in sys.path
    print("\nPython Path (sys.path):")
    for idx, path in enumerate(sys.path, 1):
        exists = os.path.exists(path)
        print(f"{idx}. {path}")
        print(f"   Exists: {'✓' if exists else '✗'}")
        
def check_module_location(module_name):
    """
    Check if a module can be found and where it's located.
    """
    print(f"\n=== Checking Module: {module_name} ===")
    
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"Module Location: {spec.origin}")
        print("Module can be imported ✓")
    else:
        print(f"Module '{module_name}' not found ✗")
        
def check_project_structure():
    """
    Display the project directory structure.
    """
    print("\n=== Project Structure ===")
    def display_tree(directory, prefix=""):
        paths = sorted(Path(directory).iterdir())
        for idx, path in enumerate(paths):
            is_last = idx == len(paths) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")
            if path.is_dir() and not path.name.startswith('.'):
                display_tree(path, prefix + ('    ' if is_last else '│   '))
                
    display_tree(os.getcwd())
    
def check_utils_accessibility():
    """
    Specifically check utils module accessibility.
    """
    print("\n=== Utils Module Accessibility ===")
    
    # Check different possible utils locations
    possible_paths = [
        Path.cwd() / 'utils',
        Path.cwd().parent / 'utils',
        Path.cwd().parent.parent / 'utils'
    ]
    
    print("Checking possible utils locations:")
    for path in possible_paths:
        exists = path.exists()
        is_package = (path / '__init__.py').exists() if exists else False
        has_vis = (path / 'visualization_utils.py').exists() if exists else False
        
        print(f"\nPath: {path}")
        print(f"Directory exists: {'✓' if exists else '✗'}")
        if exists:
            print(f"Has __init__.py: {'✓' if is_package else '✗'}")
            print(f"Has visualization_utils.py: {'✓' if has_vis else '✗'}")
            
            if is_package and has_vis:
                print("All required files present!")
                
def test_utils_import():
    """
    Test importing the utils module.
    """
    print("\n=== Testing Utils Import ===")
    try:
        import utils
        print("Successfully imported utils package ✓")
        
        try:
            from utils.visualization_utils import DataVisualizer
            print("Successfully imported DataVisualizer ✓")
        except ImportError as e:
            print(f"Could not import DataVisualizer: {e} ✗")
            
    except ImportError as e:
        print(f"Could not import utils package: {e} ✗")

def main():
    """
    Run all checks.
    """
    print("=== Python Environment Checker ===")
    print(f"Python Version: {sys.version}")
    
    # Run all checks
    check_python_paths()
    check_project_structure()
    check_utils_accessibility()
    test_utils_import()
    
    # Additional module checks
    check_module_location('utils')
    check_module_location('pandas')  # Check a known installed package
    
    print("\n=== Check Complete ===")

if __name__ == "__main__":
    main()
