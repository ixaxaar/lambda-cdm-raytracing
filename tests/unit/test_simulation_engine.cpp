#include <gtest/gtest.h>
#include "core/simulation_engine.hpp"
#include <cmath>

using namespace core;

class SimulationEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<SimulationEngine>();
        context = std::make_unique<SimulationContext>();

        // Basic configuration
        context->set_num_particles(100);
    }

    std::unique_ptr<SimulationEngine> engine;
    std::unique_ptr<SimulationContext> context;
};

// Test: Initial state
TEST_F(SimulationEngineTest, InitialState) {
    EXPECT_EQ(engine->get_state(), SimulationState::UNINITIALIZED);
}

// Test: Initialization
TEST_F(SimulationEngineTest, Initialization) {
    bool success = engine->initialize(std::move(context));
    EXPECT_TRUE(success);
    EXPECT_EQ(engine->get_state(), SimulationState::INITIALIZED);
}

// Test: Configuration setters
TEST_F(SimulationEngineTest, ConfigurationSetters) {
    engine->set_time_step(0.01);
    engine->set_max_time(10.0);
    engine->set_max_steps(1000);
    engine->set_output_frequency(10);
    engine->set_output_directory("test_output");

    // No direct getters, but we can check initialization succeeds
    engine->initialize(std::move(context));
    EXPECT_EQ(engine->get_state(), SimulationState::INITIALIZED);
}

// Test: Energy calculations
TEST_F(SimulationEngineTest, EnergyCalculations) {
    engine->initialize(std::move(context));

    // Get pointers to particle data
    const float* positions = engine->get_positions();
    const float* velocities = engine->get_velocities();
    const float* masses = engine->get_masses();

    // Initially all zero, so energies should be zero
    double kinetic = engine->compute_kinetic_energy();
    double potential = engine->compute_potential_energy();
    double total = engine->compute_total_energy();

    EXPECT_GE(kinetic, 0.0);
    EXPECT_LE(potential, 0.0);  // Gravitational potential is negative
    EXPECT_NEAR(total, kinetic + potential, 1e-6);
}

// Test: Center of mass with uniform distribution
TEST_F(SimulationEngineTest, CenterOfMass) {
    engine->initialize(std::move(context));

    float3 com = engine->compute_center_of_mass();

    // For zero-initialized particles, COM should be at origin
    EXPECT_NEAR(com.x, 0.0f, 1e-5f);
    EXPECT_NEAR(com.y, 0.0f, 1e-5f);
    EXPECT_NEAR(com.z, 0.0f, 1e-5f);
}

// Test: Angular momentum
TEST_F(SimulationEngineTest, AngularMomentum) {
    engine->initialize(std::move(context));

    float3 L = engine->compute_angular_momentum();

    // For static particles, angular momentum should be zero
    EXPECT_NEAR(L.x, 0.0f, 1e-5f);
    EXPECT_NEAR(L.y, 0.0f, 1e-5f);
    EXPECT_NEAR(L.z, 0.0f, 1e-5f);
}

// Test: State transitions
TEST_F(SimulationEngineTest, StateTransitions) {
    engine->initialize(std::move(context));

    EXPECT_EQ(engine->get_state(), SimulationState::INITIALIZED);

    // Can't directly test RUNNING without actually running
    // but we can test other transitions

    engine->reset();
    EXPECT_EQ(engine->get_state(), SimulationState::UNINITIALIZED);
}

// Test: Reset functionality
TEST_F(SimulationEngineTest, Reset) {
    engine->initialize(std::move(context));
    EXPECT_EQ(engine->get_state(), SimulationState::INITIALIZED);

    bool reset_success = engine->reset();
    EXPECT_TRUE(reset_success);
    EXPECT_EQ(engine->get_state(), SimulationState::UNINITIALIZED);
}

// Test: Statistics structure
TEST_F(SimulationEngineTest, Statistics) {
    engine->initialize(std::move(context));

    const auto& stats = engine->get_statistics();

    EXPECT_EQ(stats.current_step, 0);
    EXPECT_EQ(stats.total_steps, 0);
    EXPECT_NEAR(stats.current_time, 0.0, 1e-6);
}

// Test: Data access
TEST_F(SimulationEngineTest, DataAccess) {
    engine->initialize(std::move(context));

    const float* positions = engine->get_positions();
    const float* velocities = engine->get_velocities();
    const float* masses = engine->get_masses();
    const float* forces = engine->get_forces();

    EXPECT_NE(positions, nullptr);
    EXPECT_NE(velocities, nullptr);
    EXPECT_NE(masses, nullptr);
    EXPECT_NE(forces, nullptr);
}

// Test: Builder pattern
TEST_F(SimulationEngineTest, BuilderPattern) {
    SimulationBuilder builder;

    auto built_engine = builder
        .with_num_particles(50)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .with_max_time(5.0)
        .with_output_directory("test_output")
        .build();

    EXPECT_NE(built_engine, nullptr);
    EXPECT_EQ(built_engine->get_state(), SimulationState::INITIALIZED);
}

// Test: Multiple initialization
TEST_F(SimulationEngineTest, MultipleInitialization) {
    // First initialization
    auto ctx1 = std::make_unique<SimulationContext>();
    ctx1->set_num_particles(50);
    bool success1 = engine->initialize(std::move(ctx1));
    EXPECT_TRUE(success1);

    // Reset and reinitialize
    engine->reset();

    auto ctx2 = std::make_unique<SimulationContext>();
    ctx2->set_num_particles(100);
    bool success2 = engine->initialize(std::move(ctx2));
    EXPECT_TRUE(success2);
}

// Test: Performance summary
TEST_F(SimulationEngineTest, PerformanceSummary) {
    engine->initialize(std::move(context));

    // Should not crash
    testing::internal::CaptureStdout();
    engine->print_performance_summary();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(output.empty());
}
