#pragma once

#include "core/interfaces.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

namespace io {

// Metadata container for simulation snapshots
struct SnapshotMetadata {
    double time;
    double scale_factor;
    double redshift;
    size_t step;
    size_t num_particles;
    float box_size;

    // Cosmological parameters
    double omega_m;
    double omega_lambda;
    double h;
    double sigma_8;
    double n_s;

    // Additional metadata
    std::map<std::string, std::string> attributes;
};

// Base class for data exporters
class DataExporter : public core::IDataExporter {
protected:
    std::string name_;
    std::string version_;
    bool initialized_;
    bool use_compression_;
    int compression_level_;

public:
    DataExporter(const std::string& name, const std::string& version = "1.0");
    virtual ~DataExporter() = default;

    // IComponent interface
    bool initialize(const core::SimulationContext& context) override;
    void finalize() override;
    std::string get_type() const override { return "DataExporter"; }
    std::string get_name() const override { return name_; }
    std::string get_version() const override { return version_; }

    // IDataExporter interface
    bool export_snapshot(const std::string& filename,
                        const float* positions, const float* velocities,
                        const float* masses, size_t num_particles,
                        double time, const std::any& metadata = {}) override;

    bool import_snapshot(const std::string& filename,
                        float* positions, float* velocities,
                        float* masses, size_t& num_particles,
                        double& time, std::any& metadata) override;

    std::vector<std::string> supported_formats() const override;

    // Configuration
    void set_compression(bool enabled, int level = 6) {
        use_compression_ = enabled;
        compression_level_ = level;
    }

protected:
    // Helper methods for derived classes
    virtual bool write_snapshot_impl(const std::string& filename,
                                     const float* positions, const float* velocities,
                                     const float* masses, size_t num_particles,
                                     const SnapshotMetadata& metadata) = 0;

    virtual bool read_snapshot_impl(const std::string& filename,
                                    float* positions, float* velocities,
                                    float* masses, size_t& num_particles,
                                    SnapshotMetadata& metadata) = 0;

    SnapshotMetadata extract_metadata(const std::any& metadata_any, double time,
                                      size_t num_particles) const;
    std::any pack_metadata(const SnapshotMetadata& metadata) const;
};

}
