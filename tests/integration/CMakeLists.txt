# Integration tests - test components working together

# Test: Full simulation pipeline
add_executable(test_simulation_pipeline test_simulation_pipeline.cpp)
target_link_libraries(test_simulation_pipeline ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_simulation_pipeline)

# Test: I/O integration
add_executable(test_io_integration test_io_integration.cpp)
target_link_libraries(test_io_integration ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_io_integration)

# Test: Checkpoint and restart
add_executable(test_checkpoint_restart test_checkpoint_restart.cpp)
target_link_libraries(test_checkpoint_restart ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_checkpoint_restart)
