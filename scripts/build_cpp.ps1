# ARIA_PRO C++ Core Build Script for Windows
# Run this from the ARIA_PRO root directory

Write-Host "Building ARIA_PRO C++ Core Components..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "cpp_core")) {
    Write-Host "Error: cpp_core directory not found. Run this from ARIA_PRO root." -ForegroundColor Red
    exit 1
}

# Change to cpp_core directory
Set-Location cpp_core

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Name "build" | Out-Null
}
Set-Location build

# Configure with CMake
Write-Host "Configuring CMake..." -ForegroundColor Yellow
cmake .. -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "Building C++ components..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "C++ build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "C++ build complete!" -ForegroundColor Green
Write-Host "The aria_core module should now be available in the backend directory." -ForegroundColor Cyan

# Return to original directory
Set-Location ../..
