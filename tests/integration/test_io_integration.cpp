#include <gtest/gtest.h>
#include "io/binary_writer.hpp"
#include "io/checkpoint_manager.hpp"
#include "core/simulation_context.hpp"
#include <filesystem>

using namespace io;

class IOIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = "test_io_integration";
        std::filesystem::create_directories(test_dir);

        context = std::make_unique<core::SimulationContext>();

        // Create test particle data
        num_particles = 200;
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

            masses[i] = 1.0f + static_cast<float>(i) * 0.01f;
        }
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }

    std::string test_dir;
    std::unique_ptr<core::SimulationContext> context;

    size_t num_particles;
    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> velocities;
    std::unique_ptr<float[]> masses;
};

// Test: Multiple exporters
TEST_F(IOIntegrationTest, MultipleExporters) {
    // Create multiple writers
    auto binary_writer = std::make_unique<BinaryWriter>();
    binary_writer->initialize(*context);

    std::string bin_file = test_dir + "/snapshot.bin";

    SnapshotMetadata metadata;
    metadata.time = 1.0;
    metadata.num_particles = num_particles;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    // Export with binary writer
    bool success_bin = binary_writer->export_snapshot(
        bin_file, positions.get(), velocities.get(),
        masses.get(), num_particles, 1.0, metadata_any
    );

    EXPECT_TRUE(success_bin);
    EXPECT_TRUE(std::filesystem::exists(bin_file));
}

// Test: Large dataset I/O
TEST_F(IOIntegrationTest, LargeDatasetIO) {
    size_t large_num = 50000;
    auto large_pos = std::make_unique<float[]>(large_num * 3);
    auto large_vel = std::make_unique<float[]>(large_num * 3);
    auto large_mass = std::make_unique<float[]>(large_num);

    // Fill with data
    for (size_t i = 0; i < large_num; ++i) {
        for (int j = 0; j < 3; ++j) {
            large_pos[i * 3 + j] = static_cast<float>(i + j);
            large_vel[i * 3 + j] = static_cast<float>(i) * 0.1f;
        }
        large_mass[i] = 1.0f;
    }

    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(*context);

    std::string filename = test_dir + "/large_snapshot.bin";

    SnapshotMetadata metadata;
    metadata.time = 1.0;
    metadata.num_particles = large_num;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    bool success = writer->export_snapshot(
        filename, large_pos.get(), large_vel.get(),
        large_mass.get(), large_num, 1.0, metadata_any
    );

    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Verify file size
    auto file_size = std::filesystem::file_size(filename);
    EXPECT_GT(file_size, large_num * 7 * sizeof(float));
}

// Test: Time series output
TEST_F(IOIntegrationTest, TimeSeriesOutput) {
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(*context);

    int num_snapshots = 10;

    for (int t = 0; t < num_snapshots; ++t) {
        std::string filename = test_dir + "/snapshot_" + std::to_string(t) + ".bin";

        SnapshotMetadata metadata;
        metadata.time = t * 0.1;
        metadata.step = t * 10;
        metadata.num_particles = num_particles;
        std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

        bool success = writer->export_snapshot(
            filename, positions.get(), velocities.get(),
            masses.get(), num_particles, metadata.time, metadata_any
        );

        EXPECT_TRUE(success);
    }

    // Verify all snapshots exist
    for (int t = 0; t < num_snapshots; ++t) {
        std::string filename = test_dir + "/snapshot_" + std::to_string(t) + ".bin";
        EXPECT_TRUE(std::filesystem::exists(filename));
    }
}

// Test: Checkpoint integration with binary writer
TEST_F(IOIntegrationTest, CheckpointIntegration) {
    std::string checkpoint_dir = test_dir + "/checkpoints";

    auto checkpoint_mgr = std::make_unique<CheckpointManager>(checkpoint_dir);
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(*context);

    checkpoint_mgr->set_exporter(std::move(writer));
    checkpoint_mgr->enable(true);

    // Create checkpoints
    for (int i = 0; i < 5; ++i) {
        std::string name = "checkpoint_" + std::to_string(i);

        SnapshotMetadata metadata;
        metadata.time = i * 0.5;
        metadata.step = i * 50;
        metadata.num_particles = num_particles;

        bool success = checkpoint_mgr->create_checkpoint(
            name, positions.get(), velocities.get(),
            masses.get(), num_particles, metadata.time, metadata
        );

        EXPECT_TRUE(success);
    }

    // Verify checkpoint directory structure
    EXPECT_TRUE(std::filesystem::exists(checkpoint_dir));
}

// Test: Export and import round-trip with modifications
TEST_F(IOIntegrationTest, RoundTripWithModifications) {
    auto writer = std::make_unique<BinaryWriter>();
    writer->initialize(*context);

    std::string filename = test_dir + "/modified_snapshot.bin";

    SnapshotMetadata metadata;
    metadata.time = 1.0;
    metadata.num_particles = num_particles;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    // Export initial data
    writer->export_snapshot(filename, positions.get(), velocities.get(),
                           masses.get(), num_particles, 1.0, metadata_any);

    // Read it back
    auto read_pos = std::make_unique<float[]>(num_particles * 3);
    auto read_vel = std::make_unique<float[]>(num_particles * 3);
    auto read_mass = std::make_unique<float[]>(num_particles);
    size_t read_num = 0;
    double read_time = 0.0;
    std::any read_metadata;

    writer->import_snapshot(filename, read_pos.get(), read_vel.get(),
                           read_mass.get(), read_num, read_time, read_metadata);

    // Modify the data
    for (size_t i = 0; i < num_particles * 3; ++i) {
        read_vel[i] *= 2.0f;  // Double all velocities
    }

    // Export modified data
    std::string modified_file = test_dir + "/modified_snapshot_v2.bin";
    metadata.time = 2.0;
    metadata_any = std::make_any<SnapshotMetadata>(metadata);

    writer->export_snapshot(modified_file, read_pos.get(), read_vel.get(),
                           read_mass.get(), read_num, 2.0, metadata_any);

    EXPECT_TRUE(std::filesystem::exists(modified_file));

    // Read the modified file and verify velocities are doubled
    auto final_vel = std::make_unique<float[]>(num_particles * 3);
    writer->import_snapshot(modified_file, read_pos.get(), final_vel.get(),
                           read_mass.get(), read_num, read_time, read_metadata);

    for (size_t i = 0; i < num_particles * 3; ++i) {
        EXPECT_FLOAT_EQ(final_vel[i], velocities[i] * 2.0f);
    }
}
