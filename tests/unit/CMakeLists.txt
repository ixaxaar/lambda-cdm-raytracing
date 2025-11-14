# Unit tests - test individual components in isolation

# Test: Cosmology Model
add_executable(test_cosmology_model test_cosmology_model.cpp)
target_link_libraries(test_cosmology_model ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_cosmology_model)

# Test: Binary Writer
add_executable(test_binary_writer test_binary_writer.cpp)
target_link_libraries(test_binary_writer ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_binary_writer)

# Test: Checkpoint Manager
add_executable(test_checkpoint_manager test_checkpoint_manager.cpp)
target_link_libraries(test_checkpoint_manager ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_checkpoint_manager)

# Test: Simulation Engine
add_executable(test_simulation_engine test_simulation_engine.cpp)
target_link_libraries(test_simulation_engine ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_simulation_engine)

# Test: Component Registry
add_executable(test_component_registry test_component_registry.cpp)
target_link_libraries(test_component_registry ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_component_registry)

# Test: Configuration Manager
add_executable(test_configuration test_configuration.cpp)
target_link_libraries(test_configuration ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_configuration)
