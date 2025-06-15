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

# Advanced examples (when implementations are complete)
# add_executable(cluster_simulation cluster_simulation.cpp)
# add_executable(tensorrt_optimization tensorrt_optimization.cpp)
# add_executable(cosmology_benchmark cosmology_benchmark.cpp)