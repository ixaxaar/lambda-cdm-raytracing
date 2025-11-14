# Test infrastructure for Lambda-CDM Raytracing
cmake_minimum_required(VERSION 3.18)

# Download and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
include(GoogleTest)

# Common test settings
set(TEST_LINK_LIBRARIES
    lambda_cdm_core
    GTest::gtest_main
    GTest::gmock
)

# Unit Tests
add_subdirectory(unit)

# Integration Tests
add_subdirectory(integration)

# Validation Tests (against analytical solutions)
add_subdirectory(validation)

# Print test summary
message(STATUS "")
message(STATUS "Test Configuration:")
message(STATUS "==================")
message(STATUS "GoogleTest version: 1.14.0")
message(STATUS "Unit tests:         Enabled")
message(STATUS "Integration tests:  Enabled")
message(STATUS "Validation tests:   Enabled")
message(STATUS "")
