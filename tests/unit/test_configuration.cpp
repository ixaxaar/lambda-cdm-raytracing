#include <gtest/gtest.h>
#include "core/configuration_manager.hpp"
#include <filesystem>
#include <fstream>

using namespace core;

class ConfigurationManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = std::make_unique<ConfigurationManager>();
        test_config_file = "test_config.json";
    }

    void TearDown() override {
        if (std::filesystem::exists(test_config_file)) {
            std::filesystem::remove(test_config_file);
        }
    }

    void create_test_config_file() {
        std::ofstream file(test_config_file);
        file << R"({
            "simulation": {
                "num_particles": 1000,
                "box_size": 100.0,
                "time_step": 0.01
            },
            "cosmology": {
                "omega_m": 0.3,
                "omega_lambda": 0.7,
                "h": 0.7
            }
        })";
        file.close();
    }

    std::unique_ptr<ConfigurationManager> config;
    std::string test_config_file;
};

// Test: Default construction
TEST_F(ConfigurationManagerTest, DefaultConstruction) {
    EXPECT_NE(config, nullptr);
}

// Test: Set and get value
TEST_F(ConfigurationManagerTest, SetAndGetValue) {
    config->set("test.value", 42);

    int value = config->get<int>("test.value", 0);
    EXPECT_EQ(value, 42);
}

// Test: Get with default
TEST_F(ConfigurationManagerTest, GetWithDefault) {
    int value = config->get<int>("nonexistent.key", 99);
    EXPECT_EQ(value, 99);
}

// Test: Has key
TEST_F(ConfigurationManagerTest, HasKey) {
    config->set("existing.key", 123);

    EXPECT_TRUE(config->has("existing.key"));
    EXPECT_FALSE(config->has("nonexistent.key"));
}

// Test: Different data types
TEST_F(ConfigurationManagerTest, DifferentDataTypes) {
    config->set("int_val", 42);
    config->set("double_val", 3.14159);
    config->set("string_val", std::string("hello"));
    config->set("bool_val", true);

    EXPECT_EQ(config->get<int>("int_val", 0), 42);
    EXPECT_NEAR(config->get<double>("double_val", 0.0), 3.14159, 1e-6);
    EXPECT_EQ(config->get<std::string>("string_val", ""), "hello");
    EXPECT_TRUE(config->get<bool>("bool_val", false));
}

// Test: Nested keys
TEST_F(ConfigurationManagerTest, NestedKeys) {
    config->set("level1.level2.level3", 777);

    int value = config->get<int>("level1.level2.level3", 0);
    EXPECT_EQ(value, 777);
}

// Test: Override value
TEST_F(ConfigurationManagerTest, OverrideValue) {
    config->set("key", 100);
    EXPECT_EQ(config->get<int>("key", 0), 100);

    config->set("key", 200);
    EXPECT_EQ(config->get<int>("key", 0), 200);
}

// Test: Multiple configurations
TEST_F(ConfigurationManagerTest, MultipleConfigurations) {
    config->set("config1.param", 111);
    config->set("config2.param", 222);
    config->set("config3.param", 333);

    EXPECT_EQ(config->get<int>("config1.param", 0), 111);
    EXPECT_EQ(config->get<int>("config2.param", 0), 222);
    EXPECT_EQ(config->get<int>("config3.param", 0), 333);
}

// Test: Type safety
TEST_F(ConfigurationManagerTest, TypeSafety) {
    config->set("value", 42);

    // Getting wrong type should return default
    double as_double = config->get<double>("value", 99.9);
    // The get method should handle type conversion or return default
}
