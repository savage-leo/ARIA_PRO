#!/usr/bin/env bash
set -e

echo "Building ARIA_PRO C++ Core Components..."

# Check if we're in the right directory
if [ ! -d "cpp_core" ]; then
    echo "Error: cpp_core directory not found. Run this from ARIA_PRO root."
    exit 1
fi

cd cpp_core

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building C++ components..."
cmake --build . --config Release

echo "C++ build complete!"
echo "The aria_core module should now be available in the backend directory."

cd ../..
