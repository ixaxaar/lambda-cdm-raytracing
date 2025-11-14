#include <gtest/gtest.h>
#include "core/simulation_engine.hpp"
#include "io/checkpoint_manager.hpp"
#include "io/binary_writer.hpp"
#include <filesystem>
#include <cmath>
#include <cstring>

using namespace core;
using namespace io;

class CheckpointRestartTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = "test_checkpoint_restart";
        checkpoint_dir = test_dir + "/checkpoints";

        std::filesystem::create_directories(checkpoint_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }

    std::string test_dir;
    std::string checkpoint_dir;
};

// Test: Basic checkpoint and restart
TEST_F(CheckpointRestartTest, BasicCheckpointRestart) {
    // Create and run simulation
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(100)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Run for a few steps
    for (int i = 0; i < 10; ++i) {
        engine->step();
    }

    // Get state before checkpoint
    const auto& stats_before = engine->get_statistics();
    size_t step_before = stats_before.current_step;
    double time_before = stats_before.current_time;

    // Create checkpoint
    CheckpointManager checkpoint_mgr(checkpoint_dir);
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(engine->get_context());
    checkpoint_mgr.set_exporter(std::move(writer));
    checkpoint_mgr.enable(true);

    SnapshotMetadata metadata;
    metadata.time = time_before;
    metadata.step = step_before;
    metadata.num_particles = 100;

    std::string checkpoint_name = "restart_point";
    bool checkpoint_success = checkpoint_mgr.create_checkpoint(
        checkpoint_name,
        engine->get_positions(),
        engine->get_velocities(),
        engine->get_masses(),
        100,
        time_before,
        metadata
    );

    EXPECT_TRUE(checkpoint_success);

    // Continue simulation
    for (int i = 0; i < 10; ++i) {
        engine->step();
    }

    size_t step_after_continue = engine->get_statistics().current_step;
    EXPECT_GT(step_after_continue, step_before);

    // Now restart from checkpoint
    auto restored_pos = std::make_unique<float[]>(100 * 3);
    auto restored_vel = std::make_unique<float[]>(100 * 3);
    auto restored_mass = std::make_unique<float[]>(100);
    size_t restored_num = 0;
    double restored_time = 0.0;
    SnapshotMetadata restored_meta;

    bool restore_success = checkpoint_mgr.restore_from_checkpoint(
        checkpoint_name,
        restored_pos.get(),
        restored_vel.get(),
        restored_mass.get(),
        restored_num,
        restored_time,
        restored_meta
    );

    EXPECT_TRUE(restore_success);
    EXPECT_EQ(restored_num, 100);
    EXPECT_NEAR(restored_time, time_before, 1e-6);
    EXPECT_EQ(restored_meta.step, step_before);
}

// Test: Multiple restart points
TEST_F(CheckpointRestartTest, MultipleRestartPoints) {
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(50)
        .with_box_size(50.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    CheckpointManager checkpoint_mgr(checkpoint_dir);
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(engine->get_context());
    checkpoint_mgr.set_exporter(std::move(writer));
    checkpoint_mgr.enable(true);

    // Create multiple checkpoints during simulation
    std::vector<std::string> checkpoint_names;

    for (int cp = 0; cp < 5; ++cp) {
        // Run a few steps
        for (int i = 0; i < 5; ++i) {
            engine->step();
        }

        // Create checkpoint
        std::string name = "checkpoint_" + std::to_string(cp);
        checkpoint_names.push_back(name);

        SnapshotMetadata metadata;
        metadata.time = engine->get_statistics().current_time;
        metadata.step = engine->get_statistics().current_step;
        metadata.num_particles = 50;

        checkpoint_mgr.create_checkpoint(
            name,
            engine->get_positions(),
            engine->get_velocities(),
            engine->get_masses(),
            50,
            metadata.time,
            metadata
        );
    }

    // Verify all checkpoints exist
    for (const auto& name : checkpoint_names) {
        EXPECT_TRUE(checkpoint_mgr.checkpoint_exists(name));
    }

    // Restore from middle checkpoint
    std::string middle_checkpoint = checkpoint_names[2];

    auto restored_pos = std::make_unique<float[]>(50 * 3);
    auto restored_vel = std::make_unique<float[]>(50 * 3);
    auto restored_mass = std::make_unique<float[]>(50);
    size_t restored_num = 0;
    double restored_time = 0.0;
    SnapshotMetadata restored_meta;

    bool success = checkpoint_mgr.restore_from_checkpoint(
        middle_checkpoint,
        restored_pos.get(),
        restored_vel.get(),
        restored_mass.get(),
        restored_num,
        restored_time,
        restored_meta
    );

    EXPECT_TRUE(success);
}

// Test: Checkpoint frequency
TEST_F(CheckpointRestartTest, AutomaticCheckpointCreation) {
    CheckpointManager checkpoint_mgr(checkpoint_dir);
    checkpoint_mgr.enable(true);
    checkpoint_mgr.set_frequency_by_steps(5);  // Checkpoint every 5 steps

    // Simulate stepping through simulation
    for (size_t step = 0; step < 20; ++step) {
        if (checkpoint_mgr.should_create_checkpoint(step, step * 0.01)) {
            // Would create checkpoint here
            // For this test, just verify the logic
        }
    }

    // At steps 0, 5, 10, 15, should_create_checkpoint should return true
    EXPECT_TRUE(checkpoint_mgr.should_create_checkpoint(0, 0.0));
    EXPECT_FALSE(checkpoint_mgr.should_create_checkpoint(1, 0.01));
    EXPECT_TRUE(checkpoint_mgr.should_create_checkpoint(5, 0.05));
    EXPECT_TRUE(checkpoint_mgr.should_create_checkpoint(10, 0.10));
    EXPECT_TRUE(checkpoint_mgr.should_create_checkpoint(15, 0.15));
}

// Test: Exact state restoration
TEST_F(CheckpointRestartTest, ExactStateRestoration) {
    // Create simulation with specific particle configuration
    SimulationBuilder builder;
    auto engine = builder
        .with_num_particles(100)
        .with_box_size(100.0f)
        .with_time_step(0.01)
        .build();

    ASSERT_NE(engine, nullptr);

    // Run simulation
    for (int i = 0; i < 15; ++i) {
        engine->step();
    }

    // Store exact state
    size_t num_particles = 100;
    auto original_pos = std::make_unique<float[]>(num_particles * 3);
    auto original_vel = std::make_unique<float[]>(num_particles * 3);
    auto original_mass = std::make_unique<float[]>(num_particles);

    std::memcpy(original_pos.get(), engine->get_positions(), num_particles * 3 * sizeof(float));
    std::memcpy(original_vel.get(), engine->get_velocities(), num_particles * 3 * sizeof(float));
    std::memcpy(original_mass.get(), engine->get_masses(), num_particles * sizeof(float));

    double original_time = engine->get_statistics().current_time;
    size_t original_step = engine->get_statistics().current_step;

    // Create checkpoint
    CheckpointManager checkpoint_mgr(checkpoint_dir);
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(engine->get_context());
    checkpoint_mgr.set_exporter(std::move(writer));
    checkpoint_mgr.enable(true);

    SnapshotMetadata metadata;
    metadata.time = original_time;
    metadata.step = original_step;
    metadata.num_particles = num_particles;

    checkpoint_mgr.create_checkpoint(
        "exact_state",
        engine->get_positions(),
        engine->get_velocities(),
        engine->get_masses(),
        num_particles,
        original_time,
        metadata
    );

    // Restore from checkpoint
    auto restored_pos = std::make_unique<float[]>(num_particles * 3);
    auto restored_vel = std::make_unique<float[]>(num_particles * 3);
    auto restored_mass = std::make_unique<float[]>(num_particles);
    size_t restored_num = 0;
    double restored_time = 0.0;
    SnapshotMetadata restored_meta;

    checkpoint_mgr.restore_from_checkpoint(
        "exact_state",
        restored_pos.get(),
        restored_vel.get(),
        restored_mass.get(),
        restored_num,
        restored_time,
        restored_meta
    );

    // Verify exact match
    EXPECT_EQ(restored_num, num_particles);
    EXPECT_NEAR(restored_time, original_time, 1e-10);
    EXPECT_EQ(restored_meta.step, original_step);

    // Verify all particle data matches exactly
    for (size_t i = 0; i < num_particles * 3; ++i) {
        EXPECT_FLOAT_EQ(restored_pos[i], original_pos[i]);
        EXPECT_FLOAT_EQ(restored_vel[i], original_vel[i]);
    }

    for (size_t i = 0; i < num_particles; ++i) {
        EXPECT_FLOAT_EQ(restored_mass[i], original_mass[i]);
    }
}
