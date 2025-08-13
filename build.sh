#!/bin/bash
# Force Python 3.10 for TensorFlow compatibility
echo "Setting up Python 3.10 environment..."
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt
echo "Build completed successfully!" 