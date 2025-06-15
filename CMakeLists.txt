cmake_minimum_required(VERSION 3.18)
project(LambdaCDMRaytracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(PkgConfig REQUIRED)

pkg_check_modules(TensorRT REQUIRED tensorrt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)