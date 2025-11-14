#include "io/binary_writer.hpp"
#include "core/simulation_context.hpp"
#include <iostream>
#include <cstring>

namespace io {

// Initialize static constexpr members
constexpr char BinaryWriter::MAGIC_NUMBER[9];

BinaryWriter::BinaryWriter(const std::string& name)
    : DataExporter(name, "1.0")
    , handle_endianness_(true)
    , is_big_endian_(false)
{
    is_big_endian_ = check_endianness();
}

bool BinaryWriter::initialize(const core::SimulationContext& context) {
    if (!DataExporter::initialize(context)) {
        return false;
    }

    std::cout << "BinaryWriter initialized (endianness: "
              << (is_big_endian_ ? "big" : "little") << ")" << std::endl;
    return true;
}

void BinaryWriter::finalize() {
    DataExporter::finalize();
}

std::vector<std::string> BinaryWriter::supported_formats() const {
    return {"bin", "binary", "dat"};
}

bool BinaryWriter::write_snapshot_impl(const std::string& filename,
                                       const float* positions, const float* velocities,
                                       const float* masses, size_t num_particles,
                                       const SnapshotMetadata& metadata) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write header
    if (!write_header(file, num_particles, metadata.time, metadata)) {
        file.close();
        return false;
    }

    // Write particle data
    bool success = true;
    success = success && write_array_3d(file, positions, num_particles);
    success = success && write_array_3d(file, velocities, num_particles);
    success = success && write_array_1d(file, masses, num_particles);

    file.close();

    if (success) {
        std::cout << "Binary snapshot written: " << filename << std::endl;
    }

    return success;
}

bool BinaryWriter::read_snapshot_impl(const std::string& filename,
                                      float* positions, float* velocities,
                                      float* masses, size_t& num_particles,
                                      SnapshotMetadata& metadata) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    double time;

    // Read header
    if (!read_header(file, num_particles, time, metadata)) {
        file.close();
        return false;
    }

    metadata.time = time;

    // Read particle data
    bool success = true;
    success = success && read_array_3d(file, positions, num_particles);
    success = success && read_array_3d(file, velocities, num_particles);
    success = success && read_array_1d(file, masses, num_particles);

    file.close();

    if (success) {
        std::cout << "Binary snapshot read: " << filename << std::endl;
    }

    return success;
}

bool BinaryWriter::write_header(std::ofstream& file, size_t num_particles,
                                double time, const SnapshotMetadata& metadata) {
    // Prepare header buffer
    char header[HEADER_SIZE];
    std::memset(header, 0, HEADER_SIZE);

    // Write magic number
    std::memcpy(header, MAGIC_NUMBER, 8);

    // Write version
    uint32_t version = FORMAT_VERSION;
    std::memcpy(header + 8, &version, sizeof(uint32_t));

    // Write num_particles
    uint64_t num_p = num_particles;
    std::memcpy(header + 12, &num_p, sizeof(uint64_t));

    // Write time
    std::memcpy(header + 20, &time, sizeof(double));

    // Write box_size
    std::memcpy(header + 28, &metadata.box_size, sizeof(float));

    // Write scale_factor
    double scale_factor = metadata.scale_factor;
    std::memcpy(header + 32, &scale_factor, sizeof(double));

    // Write redshift
    double redshift = metadata.redshift;
    std::memcpy(header + 40, &redshift, sizeof(double));

    // Write step
    uint64_t step = metadata.step;
    std::memcpy(header + 48, &step, sizeof(uint64_t));

    // Write cosmology parameters
    std::memcpy(header + 56, &metadata.omega_m, sizeof(double));
    std::memcpy(header + 64, &metadata.omega_lambda, sizeof(double));
    std::memcpy(header + 72, &metadata.h, sizeof(double));
    std::memcpy(header + 80, &metadata.sigma_8, sizeof(double));
    std::memcpy(header + 88, &metadata.n_s, sizeof(double));

    // Write header to file
    file.write(header, HEADER_SIZE);

    return file.good();
}

bool BinaryWriter::read_header(std::ifstream& file, size_t& num_particles,
                               double& time, SnapshotMetadata& metadata) {
    char header[HEADER_SIZE];
    file.read(header, HEADER_SIZE);

    if (!file.good()) {
        return false;
    }

    // Check magic number
    if (std::memcmp(header, MAGIC_NUMBER, 8) != 0) {
        std::cerr << "Invalid magic number in binary file" << std::endl;
        return false;
    }

    // Read version
    uint32_t version;
    std::memcpy(&version, header + 8, sizeof(uint32_t));
    if (version != FORMAT_VERSION) {
        std::cerr << "Unsupported binary format version: " << version << std::endl;
        return false;
    }

    // Read num_particles
    uint64_t num_p;
    std::memcpy(&num_p, header + 12, sizeof(uint64_t));
    num_particles = num_p;
    metadata.num_particles = num_p;

    // Read time
    std::memcpy(&time, header + 20, sizeof(double));
    metadata.time = time;

    // Read box_size
    std::memcpy(&metadata.box_size, header + 28, sizeof(float));

    // Read scale_factor
    std::memcpy(&metadata.scale_factor, header + 32, sizeof(double));

    // Read redshift
    std::memcpy(&metadata.redshift, header + 40, sizeof(double));

    // Read step
    uint64_t step;
    std::memcpy(&step, header + 48, sizeof(uint64_t));
    metadata.step = step;

    // Read cosmology parameters
    std::memcpy(&metadata.omega_m, header + 56, sizeof(double));
    std::memcpy(&metadata.omega_lambda, header + 64, sizeof(double));
    std::memcpy(&metadata.h, header + 72, sizeof(double));
    std::memcpy(&metadata.sigma_8, header + 80, sizeof(double));
    std::memcpy(&metadata.n_s, header + 88, sizeof(double));

    return true;
}

bool BinaryWriter::write_array_3d(std::ofstream& file, const float* data, size_t num_particles) {
    size_t total_elements = num_particles * 3;
    file.write(reinterpret_cast<const char*>(data), total_elements * sizeof(float));
    return file.good();
}

bool BinaryWriter::write_array_1d(std::ofstream& file, const float* data, size_t num_particles) {
    file.write(reinterpret_cast<const char*>(data), num_particles * sizeof(float));
    return file.good();
}

bool BinaryWriter::read_array_3d(std::ifstream& file, float* data, size_t num_particles) {
    size_t total_elements = num_particles * 3;
    file.read(reinterpret_cast<char*>(data), total_elements * sizeof(float));
    return file.good();
}

bool BinaryWriter::read_array_1d(std::ifstream& file, float* data, size_t num_particles) {
    file.read(reinterpret_cast<char*>(data), num_particles * sizeof(float));
    return file.good();
}

bool BinaryWriter::check_endianness() {
    union {
        uint32_t i;
        char c[4];
    } test = {0x01020304};

    return test.c[0] == 1;  // Big endian if true
}

void BinaryWriter::swap_endian_float(float& value) {
    char* bytes = reinterpret_cast<char*>(&value);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}

void BinaryWriter::swap_endian_double(double& value) {
    char* bytes = reinterpret_cast<char*>(&value);
    std::swap(bytes[0], bytes[7]);
    std::swap(bytes[1], bytes[6]);
    std::swap(bytes[2], bytes[5]);
    std::swap(bytes[3], bytes[4]);
}

void BinaryWriter::swap_endian_uint32(uint32_t& value) {
    value = ((value & 0xFF000000) >> 24) |
            ((value & 0x00FF0000) >> 8) |
            ((value & 0x0000FF00) << 8) |
            ((value & 0x000000FF) << 24);
}

void BinaryWriter::swap_endian_uint64(uint64_t& value) {
    value = ((value & 0xFF00000000000000ULL) >> 56) |
            ((value & 0x00FF000000000000ULL) >> 40) |
            ((value & 0x0000FF0000000000ULL) >> 24) |
            ((value & 0x000000FF00000000ULL) >> 8) |
            ((value & 0x00000000FF000000ULL) << 8) |
            ((value & 0x0000000000FF0000ULL) << 24) |
            ((value & 0x000000000000FF00ULL) << 40) |
            ((value & 0x00000000000000FFULL) << 56);
}

}
