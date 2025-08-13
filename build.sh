#!/bin/bash
# Force Python 3.11 for TensorFlow compatibility
echo "Setting up Python 3.11 environment..."
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt
echo "Build completed successfully!" 