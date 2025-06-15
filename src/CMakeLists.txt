# Core sources - always included
set(CORE_SOURCES
    core/simulation_context.cpp
    core/configuration_manager.cpp
    core/component_registry.cpp
    core/simulation_engine.cpp
    forces/force_computer_factory.cpp
    forces/tree_force_computer.cpp
    physics/cosmology_model.cpp
    physics/initial_conditions.cpp
)

# Conditional sources based on available features
set(SOURCES ${CORE_SOURCES})

if(HAVE_CUDA)
    # Only add CUDA sources if they exist
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/physics/lambda_cdm_impl.cu")
        list(APPEND SOURCES physics/lambda_cdm_impl.cu)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/physics/lambda_cdm_kernels.cu")
        list(APPEND SOURCES physics/lambda_cdm_kernels.cu)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/forces/tree_force_computer.cu")
        list(APPEND SOURCES forces/tree_force_computer.cu)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/forces/barnes_hut_tree.cu")
        list(APPEND SOURCES forces/barnes_hut_tree.cu)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/resource_manager.cu")
        list(APPEND SOURCES core/resource_manager.cu)
    endif()
endif()

if(HAVE_MPI)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mpi/cluster_comm.cpp")
        list(APPEND SOURCES mpi/cluster_comm.cpp)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mpi/domain_decomposition.cpp")
        list(APPEND SOURCES mpi/domain_decomposition.cpp)
    endif()
endif()

if(HAVE_HDF5)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/io/hdf5_writer.cpp")
        list(APPEND SOURCES io/hdf5_writer.cpp)
    endif()
endif()

if(HAVE_TENSORRT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/nbody_engine.cpp")
        list(APPEND SOURCES tensorrt/nbody_engine.cpp)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/nbody_plugins.cu")
        list(APPEND SOURCES tensorrt/nbody_plugins.cu)
    endif()
endif()

# Create the core library
add_library(lambda_cdm_core ${SOURCES})

# Set library properties
target_include_directories(lambda_cdm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

# Link libraries conditionally
set(LINK_LIBRARIES)

if(HAVE_CUDA)
    list(APPEND LINK_LIBRARIES
        CUDA::cudart
        CUDA::cufft
    )
endif()

if(HAVE_MPI)
    list(APPEND LINK_LIBRARIES
        ${MPI_LIBRARIES}
    )
endif()

if(HAVE_HDF5)
    list(APPEND LINK_LIBRARIES
        ${HDF5_LIBRARIES}
    )
endif()

if(HAVE_TENSORRT)
    list(APPEND LINK_LIBRARIES
        ${TensorRT_LIBRARIES}
    )
endif()

target_link_libraries(lambda_cdm_core ${LINK_LIBRARIES})

# Set compile definitions
if(HAVE_CUDA)
    target_compile_definitions(lambda_cdm_core PUBLIC HAVE_CUDA)
endif()

if(HAVE_MPI)
    target_compile_definitions(lambda_cdm_core PUBLIC HAVE_MPI)
endif()

if(HAVE_HDF5)
    target_compile_definitions(lambda_cdm_core PUBLIC HAVE_HDF5)
endif()

if(HAVE_TENSORRT)
    target_compile_definitions(lambda_cdm_core PUBLIC HAVE_TENSORRT)
endif()

# Create minimal example executable (if main.cpp exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(lambda_cdm_sim main.cpp)
    target_link_libraries(lambda_cdm_sim lambda_cdm_core)
endif()