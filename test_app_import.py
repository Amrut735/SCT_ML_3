#!/usr/bin/env python3
"""
Test script to check app import and load_trained_models
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    print("=== TESTING APP IMPORT ===")
    import app
    
    print("✓ App imported successfully")
    print(f"Model results before: {list(app.model_results.keys())}")
    
    # Clear and reload
    app.model_results.clear()
    app._initialized = False
    
    print("\n=== CALLING LOAD_TRAINED_MODELS ===")
    app.load_trained_models()
    
    print(f"✓ Models loaded. Keys: {list(app.model_results.keys())}")
    for k, v in app.model_results.items():
        print(f"  {k}: {list(v.keys())}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 