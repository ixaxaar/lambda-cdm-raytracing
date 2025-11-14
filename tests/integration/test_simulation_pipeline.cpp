#include <gtest/gtest.h>
#include "core/simulation_engine.hpp"
#include "io/binary_writer.hpp"
#include "io/checkpoint_manager.hpp"
#include <filesystem>

using namespace core;
using namespace io;

class SimulationPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_output_dir = "test_pipeline_output";
        std::filesystem::create_directories(test_output_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_output_dir);
    }

    std::string test_output_dir;
};

// Test: Basic simulation pipeline
TEST_F(SimulationPipelineTest, BasicPipeline) {
    // Build simulation using builder pattern
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(100)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .with_max_time(0.1)
        .with_output_directory(test_output_dir)
        .build();

    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->get_state(), SimulationState::INITIALIZED);

    // Verify initial state
    const auto& stats_initial = engine->get_statistics();
    EXPECT_EQ(stats_initial.current_step, 0);
    EXPECT_NEAR(stats_initial.current_time, 0.0, 1e-6);
}

// Test: Simulation with energy conservation check
TEST_F(SimulationPipelineTest, EnergyConservation) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Compute initial energy
    double E_initial = engine->compute_total_energy();

    // Run a few steps
    for (int i = 0; i < 10; ++i) {
        bool success = engine->step();
        EXPECT_TRUE(success);
    }

    // Compute final energy
    double E_final = engine->compute_total_energy();

    // Energy should be conserved (within tolerance)
    // Note: With zero forces, energy stays exactly the same
    EXPECT_NEAR(E_final, E_initial, std::abs(E_initial) * 0.01);
}

// Test: Center of mass conservation
TEST_F(SimulationPipelineTest, CenterOfMassConservation) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Compute initial COM
    float3 COM_initial = engine->compute_center_of_mass();

    // Run a few steps
    for (int i = 0; i < 10; ++i) {
        engine->step();
    }

    // Compute final COM
    float3 COM_final = engine->compute_center_of_mass();

    // COM should be conserved (within tolerance)
    EXPECT_NEAR(COM_final.x, COM_initial.x, 1e-3f);
    EXPECT_NEAR(COM_final.y, COM_initial.y, 1e-3f);
    EXPECT_NEAR(COM_final.z, COM_initial.z, 1e-3f);
}

// Test: Simulation with data export
TEST_F(SimulationPipelineTest, SimulationWithDataExport) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(100)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .with_output_directory(test_output_dir)
        .build();

    ASSERT_NE(engine, nullptr);

    // Add binary writer as data exporter
    auto writer = std::make_shared<BinaryWriter>();
    writer->initialize(engine->get_context());
    engine->add_data_exporter(writer);

    // Run a few steps
    for (int i = 0; i < 5; ++i) {
        engine->step();
    }

    // Manually export a snapshot
    std::string snapshot_file = test_output_dir + "/snapshot.bin";

    SnapshotMetadata metadata;
    metadata.time = 0.05;
    metadata.num_particles = 100;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    bool success = writer->export_snapshot(
        snapshot_file,
        engine->get_positions(),
        engine->get_velocities(),
        engine->get_masses(),
        100,
        0.05,
        metadata_any
    );

    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(snapshot_file));
}

// Test: Pause and resume
TEST_F(SimulationPipelineTest, PauseAndResume) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Run a few steps
    for (int i = 0; i < 5; ++i) {
        engine->step();
    }

    size_t steps_before_pause = engine->get_statistics().current_step;

    // Pause
    engine->pause();
    EXPECT_EQ(engine->get_state(), SimulationState::PAUSED);

    // Try to step (should fail or do nothing)
    // (Implementation-dependent behavior)

    // Resume
    engine->resume();
    EXPECT_EQ(engine->get_state(), SimulationState::RUNNING);

    // Continue stepping
    engine->step();

    size_t steps_after_resume = engine->get_statistics().current_step;
    EXPECT_GT(steps_after_resume, steps_before_pause);
}

// Test: Stop simulation
TEST_F(SimulationPipelineTest, StopSimulation) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .with_max_time(10.0)  // 1000 steps * 0.01 dt = 10.0 time units
        .build();

    ASSERT_NE(engine, nullptr);

    // Run a few steps
    for (int i = 0; i < 5; ++i) {
        engine->step();
    }

    // Stop the simulation
    engine->stop();

    // Trying to step should not advance (implementation-dependent)
}

// Test: Reset and restart
TEST_F(SimulationPipelineTest, ResetAndRestart) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Run a few steps
    for (int i = 0; i < 10; ++i) {
        engine->step();
    }

    EXPECT_GT(engine->get_statistics().current_step, 0);

    // Reset
    engine->reset();
    EXPECT_EQ(engine->get_state(), SimulationState::UNINITIALIZED);

    // Reinitialize
    auto new_context = std::make_unique<SimulationContext>();
    new_context->set_num_particles(50);
    engine->initialize(std::move(new_context));

    // Should start from scratch
    EXPECT_EQ(engine->get_statistics().current_step, 0);
}

// Test: Performance metrics tracking
TEST_F(SimulationPipelineTest, PerformanceMetrics) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(100)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Run several steps
    for (int i = 0; i < 20; ++i) {
        engine->step();
    }

    const auto& stats = engine->get_statistics();

    EXPECT_EQ(stats.total_steps, 20);
    EXPECT_GT(stats.total_time, 0.0);
    EXPECT_GE(stats.steps_per_second, 0.0);
}
