#include <gtest/gtest.h>
#include "core/component_registry.hpp"
#include "core/simulation_context.hpp"

using namespace core;

// Mock component for testing
class MockComponent : public IComponent {
private:
    std::string name_;
    bool initialized_;

public:
    MockComponent(const std::string& name) : name_(name), initialized_(false) {}

    bool initialize(const SimulationContext& context) override {
        (void)context;
        initialized_ = true;
        return true;
    }

    void finalize() override {
        initialized_ = false;
    }

    std::string get_type() const override { return "MockComponent"; }
    std::string get_name() const override { return name_; }
    std::string get_version() const override { return "1.0"; }

    bool is_initialized() const { return initialized_; }
};

class ComponentRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = std::make_unique<ComponentRegistry>();
    }

    std::unique_ptr<ComponentRegistry> registry;
};

// Test: Register component
TEST_F(ComponentRegistryTest, RegisterComponent) {
    auto component = std::make_shared<MockComponent>("test_component");
    bool success = registry->register_component("test", component);

    EXPECT_TRUE(success);
    EXPECT_TRUE(registry->has_component("test"));
}

// Test: Get component
TEST_F(ComponentRegistryTest, GetComponent) {
    auto component = std::make_shared<MockComponent>("test_component");
    registry->register_component("test", component);

    auto retrieved = registry->get_component("test");
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->get_name(), "test_component");
}

// Test: Unregister component
TEST_F(ComponentRegistryTest, UnregisterComponent) {
    auto component = std::make_shared<MockComponent>("test_component");
    registry->register_component("test", component);

    EXPECT_TRUE(registry->has_component("test"));

    bool success = registry->unregister_component("test");
    EXPECT_TRUE(success);
    EXPECT_FALSE(registry->has_component("test"));
}

// Test: Duplicate registration
TEST_F(ComponentRegistryTest, DuplicateRegistration) {
    auto component1 = std::make_shared<MockComponent>("component1");
    auto component2 = std::make_shared<MockComponent>("component2");

    bool success1 = registry->register_component("test", component1);
    bool success2 = registry->register_component("test", component2);

    EXPECT_TRUE(success1);
    EXPECT_FALSE(success2);  // Should fail
}

// Test: Get nonexistent component
TEST_F(ComponentRegistryTest, GetNonexistentComponent) {
    auto component = registry->get_component("nonexistent");
    EXPECT_EQ(component, nullptr);
}

// Test: Initialize all components
TEST_F(ComponentRegistryTest, InitializeAllComponents) {
    auto comp1 = std::make_shared<MockComponent>("comp1");
    auto comp2 = std::make_shared<MockComponent>("comp2");

    registry->register_component("c1", comp1);
    registry->register_component("c2", comp2);

    SimulationContext context;
    bool success = registry->initialize_all_components(context);

    EXPECT_TRUE(success);
    EXPECT_TRUE(comp1->is_initialized());
    EXPECT_TRUE(comp2->is_initialized());
}

// Test: Finalize all components
TEST_F(ComponentRegistryTest, FinalizeAllComponents) {
    auto comp1 = std::make_shared<MockComponent>("comp1");
    auto comp2 = std::make_shared<MockComponent>("comp2");

    registry->register_component("c1", comp1);
    registry->register_component("c2", comp2);

    SimulationContext context;
    registry->initialize_all_components(context);

    registry->finalize_all_components();

    EXPECT_FALSE(comp1->is_initialized());
    EXPECT_FALSE(comp2->is_initialized());
}

// Test: Get all component names
TEST_F(ComponentRegistryTest, GetAllComponentNames) {
    registry->register_component("c1", std::make_shared<MockComponent>("comp1"));
    registry->register_component("c2", std::make_shared<MockComponent>("comp2"));
    registry->register_component("c3", std::make_shared<MockComponent>("comp3"));

    auto names = registry->get_all_component_names();
    EXPECT_EQ(names.size(), 3);
}

// Test: Component info
TEST_F(ComponentRegistryTest, GetComponentInfo) {
    auto component = std::make_shared<MockComponent>("test_component");
    registry->register_component("test", component);

    std::string info = registry->get_component_info("test");
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("test_component"), std::string::npos);
}

// Test: Print registry status
TEST_F(ComponentRegistryTest, PrintRegistryStatus) {
    registry->register_component("c1", std::make_shared<MockComponent>("comp1"));

    // Should not crash
    testing::internal::CaptureStdout();
    registry->print_registry_status();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(output.empty());
}
