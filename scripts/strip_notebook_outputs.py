#!/usr/bin/env python3
"""
Script to remove all outputs from Jupyter notebooks.

Usage:
    python strip_notebook_outputs.py <notebook_path>
    python strip_notebook_outputs.py <notebook_path1> <notebook_path2> ...
"""

import sys
import json
from pathlib import Path


def strip_notebook_outputs(notebook_path):
    """
    Remove all outputs from a Jupyter notebook file.
    
    Args:
        notebook_path: Path to the .ipynb file
        
    Returns:
        bool: True if successful, False otherwise
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Error: File not found: {notebook_path}")
        return False
    
    if notebook_path.suffix != '.ipynb':
        print(f"Error: Not a Jupyter notebook: {notebook_path}")
        return False
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Clear outputs from all code cells
        cells_cleared = 0
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
                cells_cleared += 1
        
        # Write the notebook back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write('\n')
        
        print(f"✓ Stripped outputs from {notebook_path} ({cells_cleared} cells)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in {notebook_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python strip_notebook_outputs.py <notebook_path> [notebook_path2] ...")
        print("\nExample:")
        print("  python strip_notebook_outputs.py 03_post_training/dinov2_svm_rf_workflow.ipynb")
        sys.exit(1)
    
    notebook_paths = sys.argv[1:]
    results = [strip_notebook_outputs(path) for path in notebook_paths]
    
    if all(results):
        print(f"\n✓ Successfully processed {len(results)} notebook(s)")
        sys.exit(0)
    else:
        print(f"\n✗ Failed to process {sum(not r for r in results)} notebook(s)")
        sys.exit(1)


if __name__ == '__main__':
    main()
