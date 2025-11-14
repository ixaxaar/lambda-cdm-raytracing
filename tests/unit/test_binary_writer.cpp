#include <gtest/gtest.h>
#include "io/binary_writer.hpp"
#include "core/simulation_context.hpp"
#include <filesystem>
#include <cmath>

using namespace io;

class BinaryWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test output directory
        test_dir = "test_binary_output";
        std::filesystem::create_directories(test_dir);

        // Initialize binary writer
        context = std::make_unique<core::SimulationContext>();
        writer = std::make_unique<BinaryWriter>();
        writer->initialize(*context);

        // Create test data
        num_particles = 100;
        positions = std::make_unique<float[]>(num_particles * 3);
        velocities = std::make_unique<float[]>(num_particles * 3);
        masses = std::make_unique<float[]>(num_particles);

        // Fill with test data
        for (size_t i = 0; i < num_particles; ++i) {
            positions[i * 3 + 0] = static_cast<float>(i);
            positions[i * 3 + 1] = static_cast<float>(i) * 2.0f;
            positions[i * 3 + 2] = static_cast<float>(i) * 3.0f;

            velocities[i * 3 + 0] = static_cast<float>(i) * 0.1f;
            velocities[i * 3 + 1] = static_cast<float>(i) * 0.2f;
            velocities[i * 3 + 2] = static_cast<float>(i) * 0.3f;

            masses[i] = 1.0f + static_cast<float>(i) * 0.01f;
        }

        // Create metadata
        metadata.time = 1.5;
        metadata.scale_factor = 0.8;
        metadata.redshift = 0.25;
        metadata.step = 150;
        metadata.num_particles = num_particles;
        metadata.box_size = 100.0f;
        metadata.omega_m = 0.3;
        metadata.omega_lambda = 0.7;
        metadata.h = 0.7;
        metadata.sigma_8 = 0.8;
        metadata.n_s = 0.96;
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all(test_dir);
    }

    std::string test_dir;
    std::unique_ptr<core::SimulationContext> context;
    std::unique_ptr<BinaryWriter> writer;

    size_t num_particles;
    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> velocities;
    std::unique_ptr<float[]> masses;

    SnapshotMetadata metadata;
};

// Test: Initialization
TEST_F(BinaryWriterTest, Initialization) {
    EXPECT_EQ(writer->get_name(), "BinaryWriter");
    EXPECT_EQ(writer->get_version(), "1.0");
}

// Test: Supported formats
TEST_F(BinaryWriterTest, SupportedFormats) {
    auto formats = writer->supported_formats();
    EXPECT_GE(formats.size(), 1);

    bool has_bin = false;
    for (const auto& fmt : formats) {
        if (fmt == "bin" || fmt == "binary" || fmt == "dat") {
            has_bin = true;
            break;
        }
    }
    EXPECT_TRUE(has_bin);
}

// Test: Write snapshot
TEST_F(BinaryWriterTest, WriteSnapshot) {
    std::string filename = test_dir + "/test_snapshot.bin";

    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    bool success = writer->export_snapshot(
        filename,
        positions.get(),
        velocities.get(),
        masses.get(),
        num_particles,
        metadata.time,
        metadata_any
    );

    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(filename));

    // Check file size is reasonable
    auto file_size = std::filesystem::file_size(filename);
    size_t expected_min_size = 256 + num_particles * (3 + 3 + 1) * sizeof(float);
    EXPECT_GE(file_size, expected_min_size);
}

// Test: Read snapshot
TEST_F(BinaryWriterTest, ReadSnapshot) {
    std::string filename = test_dir + "/test_read.bin";

    // Write data
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);
    writer->export_snapshot(filename, positions.get(), velocities.get(),
                           masses.get(), num_particles, metadata.time, metadata_any);

    // Read it back
    auto read_positions = std::make_unique<float[]>(num_particles * 3);
    auto read_velocities = std::make_unique<float[]>(num_particles * 3);
    auto read_masses = std::make_unique<float[]>(num_particles);
    size_t read_num_particles = 0;
    double read_time = 0.0;
    std::any read_metadata_any;

    bool success = writer->import_snapshot(filename, read_positions.get(),
                                          read_velocities.get(), read_masses.get(),
                                          read_num_particles, read_time, read_metadata_any);

    EXPECT_TRUE(success);
    EXPECT_EQ(read_num_particles, num_particles);
    EXPECT_NEAR(read_time, metadata.time, 1e-6);
}

// Test: Round-trip data integrity
TEST_F(BinaryWriterTest, RoundTripDataIntegrity) {
    std::string filename = test_dir + "/test_roundtrip.bin";

    // Write
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);
    writer->export_snapshot(filename, positions.get(), velocities.get(),
                           masses.get(), num_particles, metadata.time, metadata_any);

    // Read
    auto read_positions = std::make_unique<float[]>(num_particles * 3);
    auto read_velocities = std::make_unique<float[]>(num_particles * 3);
    auto read_masses = std::make_unique<float[]>(num_particles);
    size_t read_num_particles = 0;
    double read_time = 0.0;
    std::any read_metadata_any;

    writer->import_snapshot(filename, read_positions.get(), read_velocities.get(),
                           read_masses.get(), read_num_particles, read_time, read_metadata_any);

    // Verify all data matches
    for (size_t i = 0; i < num_particles * 3; ++i) {
        EXPECT_FLOAT_EQ(read_positions[i], positions[i]);
        EXPECT_FLOAT_EQ(read_velocities[i], velocities[i]);
    }

    for (size_t i = 0; i < num_particles; ++i) {
        EXPECT_FLOAT_EQ(read_masses[i], masses[i]);
    }
}

// Test: Metadata round-trip
TEST_F(BinaryWriterTest, MetadataRoundTrip) {
    std::string filename = test_dir + "/test_metadata.bin";

    // Write
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);
    writer->export_snapshot(filename, positions.get(), velocities.get(),
                           masses.get(), num_particles, metadata.time, metadata_any);

    // Read
    auto read_positions = std::make_unique<float[]>(num_particles * 3);
    auto read_velocities = std::make_unique<float[]>(num_particles * 3);
    auto read_masses = std::make_unique<float[]>(num_particles);
    size_t read_num_particles = 0;
    double read_time = 0.0;
    std::any read_metadata_any;

    writer->import_snapshot(filename, read_positions.get(), read_velocities.get(),
                           read_masses.get(), read_num_particles, read_time, read_metadata_any);

    // Extract metadata
    const auto* read_meta = std::any_cast<SnapshotMetadata>(&read_metadata_any);
    ASSERT_NE(read_meta, nullptr);

    // Verify metadata
    EXPECT_NEAR(read_meta->time, metadata.time, 1e-6);
    EXPECT_NEAR(read_meta->scale_factor, metadata.scale_factor, 1e-6);
    EXPECT_NEAR(read_meta->redshift, metadata.redshift, 1e-6);
    EXPECT_EQ(read_meta->step, metadata.step);
    EXPECT_EQ(read_meta->num_particles, metadata.num_particles);
    EXPECT_FLOAT_EQ(read_meta->box_size, metadata.box_size);
    EXPECT_NEAR(read_meta->omega_m, metadata.omega_m, 1e-6);
    EXPECT_NEAR(read_meta->omega_lambda, metadata.omega_lambda, 1e-6);
    EXPECT_NEAR(read_meta->h, metadata.h, 1e-6);
    EXPECT_NEAR(read_meta->sigma_8, metadata.sigma_8, 1e-6);
    EXPECT_NEAR(read_meta->n_s, metadata.n_s, 1e-6);
}

// Test: Large dataset
TEST_F(BinaryWriterTest, LargeDataset) {
    size_t large_num = 10000;
    auto large_pos = std::make_unique<float[]>(large_num * 3);
    auto large_vel = std::make_unique<float[]>(large_num * 3);
    auto large_mass = std::make_unique<float[]>(large_num);

    for (size_t i = 0; i < large_num * 3; ++i) {
        large_pos[i] = static_cast<float>(i);
        large_vel[i] = static_cast<float>(i) * 0.1f;
    }
    for (size_t i = 0; i < large_num; ++i) {
        large_mass[i] = 1.0f;
    }

    std::string filename = test_dir + "/test_large.bin";

    metadata.num_particles = large_num;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    bool success = writer->export_snapshot(filename, large_pos.get(), large_vel.get(),
                                          large_mass.get(), large_num, 1.0, metadata_any);
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists(filename));
}

// Test: Zero particles edge case
TEST_F(BinaryWriterTest, ZeroParticles) {
    std::string filename = test_dir + "/test_zero.bin";

    metadata.num_particles = 0;
    std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

    bool success = writer->export_snapshot(filename, nullptr, nullptr,
                                          nullptr, 0, 1.0, metadata_any);
    EXPECT_TRUE(success);
}

// Test: Multiple snapshots
TEST_F(BinaryWriterTest, MultipleSnapshots) {
    for (int i = 0; i < 5; ++i) {
        std::string filename = test_dir + "/snapshot_" + std::to_string(i) + ".bin";

        metadata.time = i * 0.5;
        metadata.step = i * 100;
        std::any metadata_any = std::make_any<SnapshotMetadata>(metadata);

        bool success = writer->export_snapshot(filename, positions.get(), velocities.get(),
                                              masses.get(), num_particles, metadata.time, metadata_any);
        EXPECT_TRUE(success);
    }

    // Verify all files exist
    for (int i = 0; i < 5; ++i) {
        std::string filename = test_dir + "/snapshot_" + std::to_string(i) + ".bin";
        EXPECT_TRUE(std::filesystem::exists(filename));
    }
}
