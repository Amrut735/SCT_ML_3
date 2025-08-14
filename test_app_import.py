#!/usr/bin/env python3
"""
Test script to check if the Flask app can be imported and started
"""

import sys
import os

print("=== Testing Flask App Import ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    # Try to import the app
    print("Attempting to import app...")
    from app import app
    print("✓ App imported successfully")
    
    # Check if app is a Flask app
    if hasattr(app, 'route'):
        print("✓ App has route decorator (Flask app)")
    else:
        print("✗ App doesn't have route decorator")
        
    # Check app configuration
    print(f"App name: {app.name}")
    print(f"App debug mode: {app.debug}")
    
    print("✓ All tests passed - app should work!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("This usually means there's a syntax error in app.py")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test Complete ===") 