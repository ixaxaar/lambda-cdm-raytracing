set(SOURCES
    physics/lambda_cdm.cu
    tensorrt/nbody_engine.cpp
    cuda/kernels.cu
    mpi/cluster_comm.cpp
    raytracing/renderer.cu
    io/hdf5_writer.cpp
    utils/profiler.cpp
)

add_library(lambda_cdm_core ${SOURCES})

target_link_libraries(lambda_cdm_core
    CUDA::cudart
    CUDA::cufft
    ${MPI_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${TensorRT_LIBRARIES}
    nvinfer
    nvonnxparser
)

target_compile_definitions(lambda_cdm_core PRIVATE
    USE_TENSORRT
    USE_MPI
    USE_HDF5
)

add_executable(lambda_cdm_sim main.cpp)
target_link_libraries(lambda_cdm_sim lambda_cdm_core)