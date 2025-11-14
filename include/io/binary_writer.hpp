#pragma once

#include "data_export.hpp"
#include <string>
#include <fstream>

namespace io {

/**
 * @brief Simple binary format data exporter
 *
 * Exports simulation snapshots to a custom binary format:
 * - Fast I/O (no compression overhead)
 * - Simple format for quick prototyping
 * - Portable across platforms (with endianness handling)
 * - Suitable for restart files and debugging
 *
 * File format:
 * - Header (256 bytes):
 *   - Magic number (8 bytes): "LAMBDACD"
 *   - Version (4 bytes)
 *   - Num particles (8 bytes)
 *   - Time (8 bytes)
 *   - Box size (4 bytes)
 *   - Reserved (224 bytes)
 * - Positions (num_particles * 3 * sizeof(float))
 * - Velocities (num_particles * 3 * sizeof(float))
 * - Masses (num_particles * sizeof(float))
 * - Metadata block (variable size)
 */
class BinaryWriter : public DataExporter {
private:
    static constexpr size_t HEADER_SIZE = 256;
    static constexpr char MAGIC_NUMBER[9] = "LAMBDACD";
    static constexpr uint32_t FORMAT_VERSION = 1;

    bool handle_endianness_;
    bool is_big_endian_;

public:
    BinaryWriter(const std::string& name = "BinaryWriter");
    ~BinaryWriter() override = default;

    // IComponent interface
    bool initialize(const core::SimulationContext& context) override;
    void finalize() override;

    std::vector<std::string> supported_formats() const override;

    // Configuration
    void enable_endianness_conversion(bool enabled = true) { handle_endianness_ = enabled; }

protected:
    bool write_snapshot_impl(const std::string& filename,
                            const float* positions, const float* velocities,
                            const float* masses, size_t num_particles,
                            const SnapshotMetadata& metadata) override;

    bool read_snapshot_impl(const std::string& filename,
                           float* positions, float* velocities,
                           float* masses, size_t& num_particles,
                           SnapshotMetadata& metadata) override;

private:
    // Binary I/O helpers
    bool write_header(std::ofstream& file, size_t num_particles,
                     double time, const SnapshotMetadata& metadata);
    bool read_header(std::ifstream& file, size_t& num_particles,
                    double& time, SnapshotMetadata& metadata);

    bool write_array_3d(std::ofstream& file, const float* data, size_t num_particles);
    bool write_array_1d(std::ofstream& file, const float* data, size_t num_particles);

    bool read_array_3d(std::ifstream& file, float* data, size_t num_particles);
    bool read_array_1d(std::ifstream& file, float* data, size_t num_particles);

    // Endianness handling
    bool check_endianness();
    void swap_endian_float(float& value);
    void swap_endian_double(double& value);
    void swap_endian_uint32(uint32_t& value);
    void swap_endian_uint64(uint64_t& value);
};

}
