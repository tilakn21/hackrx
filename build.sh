#!/bin/bash
# Render build script for robust deployment

set -e  # Exit on error

echo "Starting build process..."

# Update pip and setuptools
echo "Updating pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Set environment variables for compilation
export CFLAGS="-O2"
export CXXFLAGS="-O2"

# Install dependencies with fallback strategy
echo "Installing dependencies..."

# Try to install with binary wheels first
if ! python -m pip install --no-cache-dir --only-binary=all -r requirements.txt; then
    echo "Binary installation failed, trying with compilation..."
    python -m pip install --no-cache-dir -r requirements.txt
fi

echo "Build completed successfully!"
