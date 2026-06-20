#!/usr/bin/env python
"""Test DeepChem import after scipy fix."""
import sys

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except Exception as e:
    print(f"✗ SciPy error: {e}")
    
try:
    import deepchem as dc
    print(f"✓ DeepChem version: {dc.__version__}")
except Exception as e:
    print(f"✗ DeepChem error: {e}")
    sys.exit(1)

print("\n✓ All imports successful - Phase 2 can proceed")
