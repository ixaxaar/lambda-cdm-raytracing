#include <gtest/gtest.h>
#include "io/checkpoint_manager.hpp"
#include "io/binary_writer.hpp"
#include "core/simulation_context.hpp"
#include <filesystem>

using namespace io;

class CheckpointManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test checkpoint directory
        checkpoint_dir = "test_checkpoints";
        std::filesystem::remove_all(checkpoint_dir);  // Clean start
        std::filesystem::create_directories(checkpoint_dir);

        // Create checkpoint manager
        manager = std::make_unique<CheckpointManager>(checkpoint_dir);

        // Create and set up binary writer as exporter
        auto writer = std::make_unique<BinaryWriter>();
        core::SimulationContext context;
        writer->initialize(context);
        manager->set_exporter(std::move(writer));

        // Create test data
        num_particles = 50;
        positions = std::make_unique<float[]>(num_particles * 3);
        velocities = std::make_unique<float[]>(num_particles * 3);
        masses = std::make_unique<float[]>(num_particles);

        for (size_t i = 0; i < num_particles; ++i) {
            positions[i * 3 + 0] = static_cast<float>(i);
            positions[i * 3 + 1] = static_cast<float>(i) * 2.0f;
            positions[i * 3 + 2] = static_cast<float>(i) * 3.0f;

            velocities[i * 3 + 0] = static_cast<float>(i) * 0.1f;
            velocities[i * 3 + 1] = static_cast<float>(i) * 0.2f;
            velocities[i * 3 + 2] = static_cast<float>(i) * 0.3f;

            masses[i] = 1.0f;
        }

        metadata.time = 1.0;
        metadata.scale_factor = 0.9;
        metadata.redshift = 0.11;
        metadata.step = 100;
        metadata.num_particles = num_particles;
    }

    void TearDown() override {
        // Clean up
        std::filesystem::remove_all(checkpoint_dir);
    }

    std::string checkpoint_dir;
    std::unique_ptr<CheckpointManager> manager;

    size_t num_particles;
    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> velocities;
    std::unique_ptr<float[]> masses;

    SnapshotMetadata metadata;
};

// Test: Initialization
TEST_F(CheckpointManagerTest, Initialization) {
    EXPECT_FALSE(manager->is_enabled());
    EXPECT_EQ(manager->get_checkpoint_count(), 0);
}

// Test: Enable/disable
TEST_F(CheckpointManagerTest, EnableDisable) {
    manager->enable(true);
    EXPECT_TRUE(manager->is_enabled());

    manager->enable(false);
    EXPECT_FALSE(manager->is_enabled());
}

// Test: Create checkpoint
TEST_F(CheckpointManagerTest, CreateCheckpoint) {
    manager->enable(true);

    std::string name = "test_checkpoint_001";
    bool success = manager->create_checkpoint(name, positions.get(), velocities.get(),
                                              masses.get(), num_particles, 1.0, metadata);

    EXPECT_TRUE(success);
    EXPECT_TRUE(manager->checkpoint_exists(name));
    EXPECT_EQ(manager->get_checkpoint_count(), 1);
}

// Test: Restore checkpoint
TEST_F(CheckpointManagerTest, RestoreCheckpoint) {
    manager->enable(true);

    // Create checkpoint
    std::string name = "test_checkpoint_restore";
    manager->create_checkpoint(name, positions.get(), velocities.get(),
                              masses.get(), num_particles, 1.5, metadata);

    // Restore it
    auto restored_pos = std::make_unique<float[]>(num_particles * 3);
    auto restored_vel = std::make_unique<float[]>(num_particles * 3);
    auto restored_mass = std::make_unique<float[]>(num_particles);
    size_t restored_num = 0;
    double restored_time = 0.0;
    SnapshotMetadata restored_meta;

    bool success = manager->restore_from_checkpoint(name, restored_pos.get(), restored_vel.get(),
                                                    restored_mass.get(), restored_num,
                                                    restored_time, restored_meta);

    EXPECT_TRUE(success);
    EXPECT_EQ(restored_num, num_particles);
    EXPECT_NEAR(restored_time, 1.5, 1e-6);

    // Verify data
    for (size_t i = 0; i < num_particles * 3; ++i) {
        EXPECT_FLOAT_EQ(restored_pos[i], positions[i]);
        EXPECT_FLOAT_EQ(restored_vel[i], velocities[i]);
    }
}

// Test: Multiple checkpoints
TEST_F(CheckpointManagerTest, MultipleCheckpoints) {
    manager->enable(true);

    for (int i = 0; i < 5; ++i) {
        std::string name = manager->generate_checkpoint_name(i * 100, i * 0.5);
        manager->create_checkpoint(name, positions.get(), velocities.get(),
                                  masses.get(), num_particles, i * 0.5, metadata);
    }

    EXPECT_EQ(manager->get_checkpoint_count(), 5);
}

// Test: Checkpoint rotation
TEST_F(CheckpointManagerTest, CheckpointRotation) {
    manager->enable(true);
    manager->set_max_checkpoints(3);  // Keep only 3 most recent

    // Create 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        std::string name = "checkpoint_" + std::to_string(i);
        manager->create_checkpoint(name, positions.get(), velocities.get(),
                                  masses.get(), num_particles, i * 0.5, metadata);
    }

    // Should have only 3 most recent
    EXPECT_LE(manager->get_checkpoint_count(), 3);
}

// Test: Get latest checkpoint
TEST_F(CheckpointManagerTest, GetLatestCheckpoint) {
    manager->enable(true);

    std::string name1 = "checkpoint_old";
    std::string name2 = "checkpoint_new";

    manager->create_checkpoint(name1, positions.get(), velocities.get(),
                              masses.get(), num_particles, 1.0, metadata);

    manager->create_checkpoint(name2, positions.get(), velocities.get(),
                              masses.get(), num_particles, 2.0, metadata);

    std::string latest = manager->get_latest_checkpoint();
    EXPECT_EQ(latest, name2);
}

// Test: Delete checkpoint
TEST_F(CheckpointManagerTest, DeleteCheckpoint) {
    manager->enable(true);

    std::string name = "checkpoint_to_delete";
    manager->create_checkpoint(name, positions.get(), velocities.get(),
                              masses.get(), num_particles, 1.0, metadata);

    EXPECT_TRUE(manager->checkpoint_exists(name));

    bool deleted = manager->delete_checkpoint(name);
    EXPECT_TRUE(deleted);
    EXPECT_FALSE(manager->checkpoint_exists(name));
}

// Test: Checkpoint frequency - steps
TEST_F(CheckpointManagerTest, CheckpointFrequencySteps) {
    manager->enable(true);
    manager->set_frequency_by_steps(100);

    EXPECT_FALSE(manager->should_create_checkpoint(50, 0.5));   // No
    EXPECT_TRUE(manager->should_create_checkpoint(100, 1.0));   // Yes
    EXPECT_FALSE(manager->should_create_checkpoint(150, 1.5));  // No
    EXPECT_TRUE(manager->should_create_checkpoint(200, 2.0));   // Yes
}

// Test: Checkpoint name generation
TEST_F(CheckpointManagerTest, CheckpointNameGeneration) {
    std::string name = manager->generate_checkpoint_name(1000, 5.5);

    // Should contain step and time
    EXPECT_NE(name.find("1000"), std::string::npos);
    EXPECT_NE(name.find("5.5"), std::string::npos);
}

// Test: Validate checkpoint
TEST_F(CheckpointManagerTest, ValidateCheckpoint) {
    manager->enable(true);

    std::string name = "checkpoint_validate";
    manager->create_checkpoint(name, positions.get(), velocities.get(),
                              masses.get(), num_particles, 1.0, metadata);

    EXPECT_TRUE(manager->validate_checkpoint(name));
    EXPECT_FALSE(manager->validate_checkpoint("nonexistent_checkpoint"));
}

// Test: List checkpoints
TEST_F(CheckpointManagerTest, ListCheckpoints) {
    manager->enable(true);

    // Create several checkpoints
    std::vector<std::string> names = {"cp1", "cp2", "cp3"};
    for (const auto& name : names) {
        manager->create_checkpoint(name, positions.get(), velocities.get(),
                                  masses.get(), num_particles, 1.0, metadata);
    }

    auto checkpoint_list = manager->list_checkpoints();
    EXPECT_EQ(checkpoint_list.size(), names.size());
}

// Test: Checkpoint when disabled
TEST_F(CheckpointManagerTest, CheckpointWhenDisabled) {
    manager->enable(false);

    bool should_checkpoint = manager->should_create_checkpoint(100, 1.0);
    EXPECT_FALSE(should_checkpoint);
}

// Test: Empty checkpoint list
TEST_F(CheckpointManagerTest, EmptyCheckpointList) {
    std::string latest = manager->get_latest_checkpoint();
    EXPECT_TRUE(latest.empty());

    auto list = manager->list_checkpoints();
    EXPECT_TRUE(list.empty());
}
