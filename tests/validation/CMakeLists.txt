# Validation tests - compare against analytical solutions

# Test: Friedmann equations
add_executable(test_friedmann_equations test_friedmann_equations.cpp)
target_link_libraries(test_friedmann_equations ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_friedmann_equations)

# Test: Two-body problem
add_executable(test_two_body_problem test_two_body_problem.cpp)
target_link_libraries(test_two_body_problem ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_two_body_problem)

# Test: Free expansion (Hubble flow)
add_executable(test_hubble_flow test_hubble_flow.cpp)
target_link_libraries(test_hubble_flow ${TEST_LINK_LIBRARIES})
gtest_discover_tests(test_hubble_flow)
