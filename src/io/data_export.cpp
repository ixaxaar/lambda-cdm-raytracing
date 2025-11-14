#include "io/data_export.hpp"
#include "core/simulation_context.hpp"
#include <iostream>

namespace io {

DataExporter::DataExporter(const std::string& name, const std::string& version)
    : name_(name)
    , version_(version)
    , initialized_(false)
    , use_compression_(false)
    , compression_level_(6)
{
}

bool DataExporter::initialize(const core::SimulationContext& context) {
    (void)context;  // Suppress unused parameter warning

    initialized_ = true;
    std::cout << "DataExporter initialized: " << name_ << std::endl;
    return true;
}

void DataExporter::finalize() {
    initialized_ = false;
    std::cout << "DataExporter finalized: " << name_ << std::endl;
}

bool DataExporter::export_snapshot(const std::string& filename,
                                   const float* positions, const float* velocities,
                                   const float* masses, size_t num_particles,
                                   double time, const std::any& metadata) {
    if (!initialized_) {
        std::cerr << "DataExporter not initialized" << std::endl;
        return false;
    }

    // Extract metadata
    SnapshotMetadata snap_metadata = extract_metadata(metadata, time, num_particles);

    // Call derived class implementation
    return write_snapshot_impl(filename, positions, velocities, masses,
                              num_particles, snap_metadata);
}

bool DataExporter::import_snapshot(const std::string& filename,
                                   float* positions, float* velocities,
                                   float* masses, size_t& num_particles,
                                   double& time, std::any& metadata) {
    if (!initialized_) {
        std::cerr << "DataExporter not initialized" << std::endl;
        return false;
    }

    SnapshotMetadata snap_metadata;

    // Call derived class implementation
    bool success = read_snapshot_impl(filename, positions, velocities, masses,
                                      num_particles, snap_metadata);

    if (success) {
        time = snap_metadata.time;
        metadata = pack_metadata(snap_metadata);
    }

    return success;
}

std::vector<std::string> DataExporter::supported_formats() const {
    return {"generic"};
}

SnapshotMetadata DataExporter::extract_metadata(const std::any& metadata_any,
                                                double time,
                                                size_t num_particles) const {
    SnapshotMetadata metadata;
    metadata.time = time;
    metadata.num_particles = num_particles;
    metadata.scale_factor = 1.0;
    metadata.redshift = 0.0;
    metadata.step = 0;
    metadata.box_size = 100.0f;

    // Default cosmological parameters
    metadata.omega_m = 0.3;
    metadata.omega_lambda = 0.7;
    metadata.h = 0.7;
    metadata.sigma_8 = 0.8;
    metadata.n_s = 0.96;

    // Try to extract from std::any if provided
    try {
        if (metadata_any.has_value()) {
            const auto* meta_ptr = std::any_cast<SnapshotMetadata>(&metadata_any);
            if (meta_ptr) {
                metadata = *meta_ptr;
            }
        }
    } catch (const std::bad_any_cast& e) {
        // Silently use defaults
    }

    return metadata;
}

std::any DataExporter::pack_metadata(const SnapshotMetadata& metadata) const {
    return std::make_any<SnapshotMetadata>(metadata);
}

}
