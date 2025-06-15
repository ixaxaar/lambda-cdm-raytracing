# Examples CMakeLists.txt

# Basic simulation example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/basic_simulation.cpp")
    add_executable(basic_simulation basic_simulation.cpp)
    target_link_libraries(basic_simulation lambda_cdm_core)
    
    # Copy configuration files to build directory
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/configs/basic_lambda_cdm.json"
        "${CMAKE_CURRENT_BINARY_DIR}/basic_lambda_cdm.json"
        COPYONLY
    )
endif()

# CUDA N-body test example
if(HAVE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_nbody_test.cpp")
    add_executable(cuda_nbody_test cuda_nbody_test.cpp)
    target_link_libraries(cuda_nbody_test lambda_cdm_core)
endif()

# Barnes-Hut tree test example
if(HAVE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/barnes_hut_test.cu")
    add_executable(barnes_hut_test barnes_hut_test.cu)
    target_link_libraries(barnes_hut_test lambda_cdm_core)
endif()

# Cosmology test example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cosmology_test.cpp")
    add_executable(cosmology_test cosmology_test.cpp)
    target_link_libraries(cosmology_test lambda_cdm_core)
endif()

# Zel'dovich approximation test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zeldovich_test.cpp")
    add_executable(zeldovich_test zeldovich_test.cpp)
    target_link_libraries(zeldovich_test lambda_cdm_core)
endif()

# 2LPT test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/2lpt_test.cpp")
    add_executable(2lpt_test 2lpt_test.cpp)
    target_link_libraries(2lpt_test lambda_cdm_core)
endif()

# Advanced examples (when implementations are complete)
# add_executable(cluster_simulation cluster_simulation.cpp)
# add_executable(tensorrt_optimization tensorrt_optimization.cpp)
# add_executable(cosmology_benchmark cosmology_benchmark.cpp)