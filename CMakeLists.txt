cmake_minimum_required(VERSION 3.18)
project(LambdaCDMRaytracing VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optional dependencies with graceful fallback
option(USE_CUDA "Enable CUDA support" ON)
option(USE_MPI "Enable MPI support" ON)
option(USE_HDF5 "Enable HDF5 support" ON)
option(USE_TENSORRT "Enable TensorRT support" ON)

# Required dependencies
find_package(PkgConfig REQUIRED)

# CUDA support
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        set(HAVE_CUDA TRUE)
        enable_language(CUDA)
        
        # Set CUDA architecture (auto-detect if not specified)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
        endif()
        
        # Enable CUDA separable compilation for better performance
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(WARNING "CUDA not found - GPU acceleration disabled")
        set(HAVE_CUDA FALSE)
        set(USE_CUDA OFF)
    endif()
endif()

# MPI support
if(USE_MPI)
    find_package(MPI QUIET)
    if(MPI_FOUND)
        message(STATUS "MPI found")
        set(HAVE_MPI TRUE)
    else()
        message(WARNING "MPI not found - cluster support disabled")
        set(HAVE_MPI FALSE)
        set(USE_MPI OFF)
    endif()
endif()

# HDF5 support
if(USE_HDF5)
    find_package(HDF5 QUIET COMPONENTS CXX)
    if(HDF5_FOUND)
        message(STATUS "HDF5 found")
        set(HAVE_HDF5 TRUE)
    else()
        message(WARNING "HDF5 not found - advanced I/O disabled")
        set(HAVE_HDF5 FALSE)
        set(USE_HDF5 OFF)
    endif()
endif()

# TensorRT support (optional)
if(USE_TENSORRT)
    pkg_check_modules(TensorRT QUIET tensorrt)
    if(TensorRT_FOUND)
        message(STATUS "TensorRT found")
        set(HAVE_TENSORRT TRUE)
    else()
        message(WARNING "TensorRT not found - TensorRT optimization disabled")
        set(HAVE_TENSORRT FALSE)
        set(USE_TENSORRT OFF)
    endif()
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
if(HAVE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --use_fast_math")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(HAVE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
endif()

if(HAVE_HDF5)
    include_directories(${HDF5_INCLUDE_DIRS})
endif()

if(HAVE_TENSORRT)
    include_directories(${TensorRT_INCLUDE_DIRS})
endif()

# Generate config header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
    @ONLY
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Add subdirectories
add_subdirectory(src)

# Examples (optional)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

# Testing (optional)
option(ENABLE_TESTING "Enable testing" OFF)
if(ENABLE_TESTING)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        add_subdirectory(tests)
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Lambda-CDM Raytracing Configuration Summary:")
message(STATUS "==========================================")
message(STATUS "CUDA support:     ${HAVE_CUDA}")
message(STATUS "MPI support:      ${HAVE_MPI}")
message(STATUS "HDF5 support:     ${HAVE_HDF5}")
message(STATUS "TensorRT support: ${HAVE_TENSORRT}")
message(STATUS "Testing enabled:  ${ENABLE_TESTING}")
message(STATUS "")