#pragma once

#include "data_export.hpp"
#include <string>

#ifdef HAVE_HDF5
    #include <hdf5.h>
#endif

namespace io {

/**
 * @brief HDF5-based data exporter with parallel I/O support
 *
 * Exports simulation snapshots to HDF5 format with:
 * - Parallel I/O using MPI-IO (when MPI is available)
 * - Compression support (gzip)
 * - Chunked datasets for efficient access
 * - Complete metadata storage
 *
 * File structure:
 * /PartData/
 *   - Positions [N, 3]
 *   - Velocities [N, 3]
 *   - Masses [N]
 * /Header/
 *   - Time, Redshift, NumParticles, BoxSize, etc.
 * /Cosmology/
 *   - OmegaM, OmegaLambda, h, sigma_8, n_s
 */
class HDF5Writer : public DataExporter {
private:
#ifdef HAVE_HDF5
    hid_t file_property_list_;
    hid_t dataset_property_list_;
    bool use_parallel_io_;
#endif

    int mpi_rank_;
    int mpi_size_;

public:
    HDF5Writer(const std::string& name = "HDF5Writer");
    ~HDF5Writer() override;

    // IComponent interface
    bool initialize(const core::SimulationContext& context) override;
    void finalize() override;

    std::vector<std::string> supported_formats() const override;

    // Configuration
    void enable_parallel_io(bool enabled = true);
    void set_chunk_size(size_t chunk_size);

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
#ifdef HAVE_HDF5
    // HDF5 helper methods
    bool create_file(const std::string& filename, hid_t& file_id);
    bool open_file(const std::string& filename, hid_t& file_id);
    void close_file(hid_t file_id);

    bool write_dataset_3d(hid_t file_id, const std::string& path,
                         const float* data, size_t num_particles);
    bool write_dataset_1d(hid_t file_id, const std::string& path,
                         const float* data, size_t num_particles);

    bool read_dataset_3d(hid_t file_id, const std::string& path,
                        float* data, size_t num_particles);
    bool read_dataset_1d(hid_t file_id, const std::string& path,
                        float* data, size_t num_particles);

    bool write_metadata(hid_t file_id, const SnapshotMetadata& metadata);
    bool read_metadata(hid_t file_id, SnapshotMetadata& metadata);

    bool write_attribute_double(hid_t group_id, const std::string& name, double value);
    bool write_attribute_int(hid_t group_id, const std::string& name, int64_t value);
    bool write_attribute_string(hid_t group_id, const std::string& name, const std::string& value);

    bool read_attribute_double(hid_t group_id, const std::string& name, double& value);
    bool read_attribute_int(hid_t group_id, const std::string& name, int64_t& value);
    bool read_attribute_string(hid_t group_id, const std::string& name, std::string& value);
#endif
};

}
