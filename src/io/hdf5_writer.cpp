#include "io/hdf5_writer.hpp"
#include "core/simulation_context.hpp"
#include <iostream>
#include <cstring>

namespace io {

HDF5Writer::HDF5Writer(const std::string& name)
    : DataExporter(name, "1.0")
#ifdef HAVE_HDF5
    , file_property_list_(H5P_DEFAULT)
    , dataset_property_list_(H5P_DEFAULT)
    , use_parallel_io_(false)
#endif
    , mpi_rank_(0)
    , mpi_size_(1)
{
}

HDF5Writer::~HDF5Writer() {
#ifdef HAVE_HDF5
    if (file_property_list_ != H5P_DEFAULT) {
        H5Pclose(file_property_list_);
    }
    if (dataset_property_list_ != H5P_DEFAULT) {
        H5Pclose(dataset_property_list_);
    }
#endif
}

bool HDF5Writer::initialize(const core::SimulationContext& context) {
    if (!DataExporter::initialize(context)) {
        return false;
    }

#ifdef HAVE_HDF5
    // Get MPI information if available
#ifdef HAVE_MPI
    mpi_rank_ = context.get_mpi_rank();
    mpi_size_ = context.get_mpi_size();
#endif

    // Create property lists
    file_property_list_ = H5Pcreate(H5P_FILE_ACCESS);
    dataset_property_list_ = H5Pcreate(H5P_DATASET_CREATE);

#ifdef HAVE_MPI
    if (use_parallel_io_ && mpi_size_ > 1) {
        // Set up parallel HDF5
        H5Pset_fapl_mpio(file_property_list_, MPI_COMM_WORLD, MPI_INFO_NULL);
    }
#endif

    // Set up compression if enabled
    if (use_compression_) {
        hsize_t chunk_dims[2] = {1024, 3};  // Chunk size
        H5Pset_chunk(dataset_property_list_, 2, chunk_dims);
        H5Pset_deflate(dataset_property_list_, compression_level_);
    }

    std::cout << "HDF5Writer initialized (rank " << mpi_rank_
              << "/" << mpi_size_ << ")" << std::endl;
    return true;
#else
    std::cerr << "HDF5 support not available" << std::endl;
    return false;
#endif
}

void HDF5Writer::finalize() {
    DataExporter::finalize();
}

std::vector<std::string> HDF5Writer::supported_formats() const {
    return {"hdf5", "h5"};
}

void HDF5Writer::enable_parallel_io(bool enabled) {
#ifdef HAVE_HDF5
    use_parallel_io_ = enabled;
#else
    (void)enabled;
    std::cerr << "HDF5 not available, parallel I/O cannot be enabled" << std::endl;
#endif
}

void HDF5Writer::set_chunk_size(size_t chunk_size) {
#ifdef HAVE_HDF5
    hsize_t chunk_dims[2] = {static_cast<hsize_t>(chunk_size), 3};
    H5Pset_chunk(dataset_property_list_, 2, chunk_dims);
#else
    (void)chunk_size;
#endif
}

bool HDF5Writer::write_snapshot_impl(const std::string& filename,
                                     const float* positions, const float* velocities,
                                     const float* masses, size_t num_particles,
                                     const SnapshotMetadata& metadata) {
#ifdef HAVE_HDF5
    hid_t file_id;

    // Create file
    if (!create_file(filename, file_id)) {
        return false;
    }

    // Write datasets
    bool success = true;
    success = success && write_dataset_3d(file_id, "/PartData/Positions", positions, num_particles);
    success = success && write_dataset_3d(file_id, "/PartData/Velocities", velocities, num_particles);
    success = success && write_dataset_1d(file_id, "/PartData/Masses", masses, num_particles);

    // Write metadata
    success = success && write_metadata(file_id, metadata);

    // Close file
    close_file(file_id);

    if (success && mpi_rank_ == 0) {
        std::cout << "HDF5 snapshot written: " << filename << std::endl;
    }

    return success;
#else
    (void)filename; (void)positions; (void)velocities;
    (void)masses; (void)num_particles; (void)metadata;
    std::cerr << "HDF5 not available" << std::endl;
    return false;
#endif
}

bool HDF5Writer::read_snapshot_impl(const std::string& filename,
                                    float* positions, float* velocities,
                                    float* masses, size_t& num_particles,
                                    SnapshotMetadata& metadata) {
#ifdef HAVE_HDF5
    hid_t file_id;

    // Open file
    if (!open_file(filename, file_id)) {
        return false;
    }

    // Read metadata first to get num_particles
    if (!read_metadata(file_id, metadata)) {
        close_file(file_id);
        return false;
    }

    num_particles = metadata.num_particles;

    // Read datasets
    bool success = true;
    success = success && read_dataset_3d(file_id, "/PartData/Positions", positions, num_particles);
    success = success && read_dataset_3d(file_id, "/PartData/Velocities", velocities, num_particles);
    success = success && read_dataset_1d(file_id, "/PartData/Masses", masses, num_particles);

    // Close file
    close_file(file_id);

    if (success && mpi_rank_ == 0) {
        std::cout << "HDF5 snapshot read: " << filename << std::endl;
    }

    return success;
#else
    (void)filename; (void)positions; (void)velocities;
    (void)masses; (void)num_particles; (void)metadata;
    std::cerr << "HDF5 not available" << std::endl;
    return false;
#endif
}

#ifdef HAVE_HDF5

bool HDF5Writer::create_file(const std::string& filename, hid_t& file_id) {
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, file_property_list_);
    if (file_id < 0) {
        std::cerr << "Failed to create HDF5 file: " << filename << std::endl;
        return false;
    }
    return true;
}

bool HDF5Writer::open_file(const std::string& filename, hid_t& file_id) {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, file_property_list_);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filename << std::endl;
        return false;
    }
    return true;
}

void HDF5Writer::close_file(hid_t file_id) {
    H5Fclose(file_id);
}

bool HDF5Writer::write_dataset_3d(hid_t file_id, const std::string& path,
                                  const float* data, size_t num_particles) {
    // Create groups if needed
    std::string group_path = path.substr(0, path.find_last_of('/'));
    hid_t group_id = H5Gcreate(file_id, group_path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
        group_id = H5Gopen(file_id, group_path.c_str(), H5P_DEFAULT);
    }

    // Create dataspace
    hsize_t dims[2] = {num_particles, 3};
    hid_t dataspace = H5Screate_simple(2, dims, nullptr);

    // Create dataset
    hid_t dataset = H5Dcreate(file_id, path.c_str(), H5T_NATIVE_FLOAT, dataspace,
                             H5P_DEFAULT, dataset_property_list_, H5P_DEFAULT);

    // Write data
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // Cleanup
    H5Dclose(dataset);
    H5Sclose(dataspace);
    if (group_id >= 0) H5Gclose(group_id);

    return status >= 0;
}

bool HDF5Writer::write_dataset_1d(hid_t file_id, const std::string& path,
                                  const float* data, size_t num_particles) {
    // Create dataspace
    hsize_t dims[1] = {num_particles};
    hid_t dataspace = H5Screate_simple(1, dims, nullptr);

    // Create dataset
    hid_t dataset = H5Dcreate(file_id, path.c_str(), H5T_NATIVE_FLOAT, dataspace,
                             H5P_DEFAULT, dataset_property_list_, H5P_DEFAULT);

    // Write data
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // Cleanup
    H5Dclose(dataset);
    H5Sclose(dataspace);

    return status >= 0;
}

bool HDF5Writer::read_dataset_3d(hid_t file_id, const std::string& path,
                                 float* data, size_t num_particles) {
    hid_t dataset = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) return false;

    herr_t status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);

    return status >= 0;
}

bool HDF5Writer::read_dataset_1d(hid_t file_id, const std::string& path,
                                 float* data, size_t num_particles) {
    hid_t dataset = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);
    if (dataset < 0) return false;

    herr_t status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);

    return status >= 0;
}

bool HDF5Writer::write_metadata(hid_t file_id, const SnapshotMetadata& metadata) {
    // Create Header group
    hid_t header_group = H5Gcreate(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    bool success = true;
    success = success && write_attribute_double(header_group, "Time", metadata.time);
    success = success && write_attribute_double(header_group, "ScaleFactor", metadata.scale_factor);
    success = success && write_attribute_double(header_group, "Redshift", metadata.redshift);
    success = success && write_attribute_int(header_group, "NumParticles", metadata.num_particles);
    success = success && write_attribute_int(header_group, "Step", metadata.step);

    H5Gclose(header_group);

    // Create Cosmology group
    hid_t cosmo_group = H5Gcreate(file_id, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    success = success && write_attribute_double(cosmo_group, "OmegaM", metadata.omega_m);
    success = success && write_attribute_double(cosmo_group, "OmegaLambda", metadata.omega_lambda);
    success = success && write_attribute_double(cosmo_group, "h", metadata.h);
    success = success && write_attribute_double(cosmo_group, "sigma_8", metadata.sigma_8);
    success = success && write_attribute_double(cosmo_group, "n_s", metadata.n_s);

    H5Gclose(cosmo_group);

    return success;
}

bool HDF5Writer::read_metadata(hid_t file_id, SnapshotMetadata& metadata) {
    hid_t header_group = H5Gopen(file_id, "/Header", H5P_DEFAULT);
    if (header_group < 0) return false;

    bool success = true;
    int64_t num_particles_int, step_int;
    success = success && read_attribute_double(header_group, "Time", metadata.time);
    success = success && read_attribute_double(header_group, "ScaleFactor", metadata.scale_factor);
    success = success && read_attribute_double(header_group, "Redshift", metadata.redshift);
    success = success && read_attribute_int(header_group, "NumParticles", num_particles_int);
    success = success && read_attribute_int(header_group, "Step", step_int);

    metadata.num_particles = num_particles_int;
    metadata.step = step_int;

    H5Gclose(header_group);

    hid_t cosmo_group = H5Gopen(file_id, "/Cosmology", H5P_DEFAULT);
    if (cosmo_group >= 0) {
        read_attribute_double(cosmo_group, "OmegaM", metadata.omega_m);
        read_attribute_double(cosmo_group, "OmegaLambda", metadata.omega_lambda);
        read_attribute_double(cosmo_group, "h", metadata.h);
        read_attribute_double(cosmo_group, "sigma_8", metadata.sigma_8);
        read_attribute_double(cosmo_group, "n_s", metadata.n_s);
        H5Gclose(cosmo_group);
    }

    return success;
}

bool HDF5Writer::write_attribute_double(hid_t group_id, const std::string& name, double value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate(group_id, name.c_str(), H5T_NATIVE_DOUBLE, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attribute);
    H5Sclose(dataspace);
    return status >= 0;
}

bool HDF5Writer::write_attribute_int(hid_t group_id, const std::string& name, int64_t value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate(group_id, name.c_str(), H5T_NATIVE_INT64, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attribute, H5T_NATIVE_INT64, &value);
    H5Aclose(attribute);
    H5Sclose(dataspace);
    return status >= 0;
}

bool HDF5Writer::write_attribute_string(hid_t group_id, const std::string& name, const std::string& value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, value.size());
    hid_t attribute = H5Acreate(group_id, name.c_str(), string_type, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attribute, string_type, value.c_str());
    H5Aclose(attribute);
    H5Tclose(string_type);
    H5Sclose(dataspace);
    return status >= 0;
}

bool HDF5Writer::read_attribute_double(hid_t group_id, const std::string& name, double& value) {
    hid_t attribute = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    if (attribute < 0) return false;
    herr_t status = H5Aread(attribute, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attribute);
    return status >= 0;
}

bool HDF5Writer::read_attribute_int(hid_t group_id, const std::string& name, int64_t& value) {
    hid_t attribute = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    if (attribute < 0) return false;
    herr_t status = H5Aread(attribute, H5T_NATIVE_INT64, &value);
    H5Aclose(attribute);
    return status >= 0;
}

bool HDF5Writer::read_attribute_string(hid_t group_id, const std::string& name, std::string& value) {
    hid_t attribute = H5Aopen(group_id, name.c_str(), H5P_DEFAULT);
    if (attribute < 0) return false;

    hid_t string_type = H5Tcopy(H5T_C_S1);
    size_t size = H5Aget_storage_size(attribute);
    H5Tset_size(string_type, size);

    char* buffer = new char[size + 1];
    herr_t status = H5Aread(attribute, string_type, buffer);
    buffer[size] = '\0';
    value = buffer;
    delete[] buffer;

    H5Tclose(string_type);
    H5Aclose(attribute);
    return status >= 0;
}

#endif  // HAVE_HDF5

}
