cmake_minimum_required(VERSION 3.16)
project(ARIA_CORE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and PyBind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/market_data_processor.cpp
    src/smc_engine.cpp
    python_bindings.cpp
)

# Create the Python module
pybind11_add_module(aria_core ${SOURCES})

# Set output directory
set_target_properties(aria_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../backend
)

# Compiler flags for optimization
if(MSVC)
    target_compile_options(aria_core PRIVATE /O2 /DNDEBUG)
else()
    target_compile_options(aria_core PRIVATE -O3 -DNDEBUG -march=native)
endif()

# Link libraries
target_link_libraries(aria_core PRIVATE pybind11::module)
