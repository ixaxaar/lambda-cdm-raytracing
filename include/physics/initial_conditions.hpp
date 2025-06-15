#pragma once

#include "core/math_types.hpp"
#include "physics/cosmology_model.hpp"
#include <vector>
#include <memory>
#include <random>
#include <complex>
#include <functional>

namespace physics {

// Power spectrum models
enum class PowerSpectrumType {
    EISENSTEIN_HU,      // Eisenstein & Hu (1998) transfer function
    CAMB_LIKE,          // CAMB-like parametrization
    CDM_ONLY,           // Cold Dark Matter only (no baryons)
    CUSTOM              // User-provided power spectrum
};

// Initial conditions parameters
struct InitialConditionsParams {
    // Grid parameters
    size_t grid_size = 256;           // Number of grid points per dimension
    float box_size = 100.0f;          // Box size in Mpc/h
    
    // Redshift parameters
    double z_initial = 49.0;          // Initial redshift (z=49 is common)
    bool normalize_at_z0 = true;      // Normalize power spectrum at z=0
    
    // Power spectrum parameters
    PowerSpectrumType ps_type = PowerSpectrumType::EISENSTEIN_HU;
    double k_min = 0.001;             // Minimum wavenumber in h/Mpc
    double k_max = 100.0;             // Maximum wavenumber in h/Mpc
    
    // Random seed
    uint32_t random_seed = 12345;
    bool fix_amplitude = false;       // Fix amplitude of modes (for reproducibility)
    
    // Output options
    bool generate_velocities = true;  // Generate velocity field
    bool apply_glass = false;         // Apply glass-like pre-initial conditions
    bool use_2lpt = false;            // Use 2LPT instead of Zel'dovich
    
    // Custom power spectrum function (if ps_type == CUSTOM)
    std::function<double(double)> custom_power_spectrum;
};

// Fourier mode structure
struct FourierMode {
    std::complex<double> delta_k;     // Density field in Fourier space
    float3 k_vector;                  // Wavevector
    double k_magnitude;               // |k|
    double power;                     // P(k) value
};

// Initial conditions generator using Zel'dovich approximation
class InitialConditionsGenerator {
private:
    InitialConditionsParams params_;
    CosmologyModel cosmology_;
    
    // Random number generator
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<double> normal_dist_;
    
    // Grid parameters
    size_t n_total_;                  // Total number of grid points
    float dk_;                        // Grid spacing in k-space
    float dx_;                        // Grid spacing in real space
    
    // Power spectrum data
    std::vector<double> k_values_;
    std::vector<double> power_values_;
    bool power_spectrum_initialized_;
    
    // FFT workspace (if needed)
    std::vector<std::complex<double>> fourier_workspace_;
    
public:
    explicit InitialConditionsGenerator(
        const InitialConditionsParams& params,
        const CosmologyModel& cosmology);
    
    ~InitialConditionsGenerator() = default;
    
    // Main generation methods
    void generate_initial_conditions(
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses) const;
    
    void generate_particles(
        size_t num_particles,
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses) const;
    
    // Power spectrum initialization
    void initialize_power_spectrum();
    double get_power_spectrum(double k) const;
    
    // Transfer functions
    double eisenstein_hu_transfer(double k) const;
    double cdm_transfer_function(double k) const;
    
    // Power spectrum methods
    double eisenstein_hu_power_spectrum(double k) const;
    double cdm_power_spectrum(double k) const;
    double compute_sigma8() const;
    
    // Zel'dovich approximation methods
    void generate_gaussian_field(
        std::vector<std::complex<double>>& delta_k) const;
    
    void apply_zeldovich_approximation(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<float3>& positions,
        std::vector<float3>& velocities) const;
    
    void compute_displacement_field(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<float3>& displacements) const;
    
    void compute_velocity_field(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<float3>& velocities) const;
    
    // Grid-based methods
    void grid_to_particles(
        const std::vector<float3>& grid_positions,
        const std::vector<float3>& grid_velocities,
        size_t num_particles,
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses) const;
    
    // Utility methods
    float3 grid_index_to_position(size_t i, size_t j, size_t k) const;
    float3 grid_index_to_k_vector(size_t i, size_t j, size_t k) const;
    double k_vector_magnitude(const float3& k_vec) const;
    
    size_t flatten_index(size_t i, size_t j, size_t k) const {
        return i * params_.grid_size * params_.grid_size + j * params_.grid_size + k;
    }
    
    void unflatten_index(size_t flat_idx, size_t& i, size_t& j, size_t& k) const {
        i = flat_idx / (params_.grid_size * params_.grid_size);
        j = (flat_idx % (params_.grid_size * params_.grid_size)) / params_.grid_size;
        k = flat_idx % params_.grid_size;
    }
    
    // Growth factor and velocity scaling
    double get_growth_factor() const;
    double get_growth_rate() const;
    double get_velocity_scaling() const;
    
    // Diagnostics and validation
    void validate_power_spectrum() const;
    void print_statistics(const std::vector<float3>& positions,
                         const std::vector<float3>& velocities) const;
    
    double compute_rms_displacement(const std::vector<float3>& positions) const;
    double compute_rms_velocity(const std::vector<float3>& velocities) const;
    
    // Parameter access
    const InitialConditionsParams& get_params() const { return params_; }
    const CosmologyModel& get_cosmology() const { return cosmology_; }
    
    // Advanced methods for higher-order perturbation theory (2LPT)
    void apply_second_order_corrections(
        std::vector<float3>& positions,
        std::vector<float3>& velocities) const;
    
    // 2LPT implementation methods
    void compute_2lpt_displacement_field(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<float3>& displacements_2lpt) const;
    
    void compute_second_order_kernel(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<std::complex<double>>& delta_2_k) const;
    
    void apply_2lpt_approximation(
        const std::vector<std::complex<double>>& delta_k,
        std::vector<float3>& positions,
        std::vector<float3>& velocities) const;
    
    // Glass-like initial conditions
    void generate_glass_positions(std::vector<float3>& positions) const;
    void relax_glass_configuration(std::vector<float3>& positions, int n_iterations = 10) const;
    
private:
    // Internal helper methods
    void setup_k_space_grid();
    void initialize_fft_workspace();
    
    // Fourier transforms (simple implementation - could be optimized with FFTW)
    void forward_fft_3d(std::vector<std::complex<double>>& data) const;
    void inverse_fft_3d(std::vector<std::complex<double>>& data) const;
    void fft_1d(std::vector<std::complex<double>>& data, bool inverse = false) const;
    
    // Interpolation methods
    float3 interpolate_displacement(const std::vector<float3>& grid_displacements,
                                   const float3& position) const;
    float3 interpolate_velocity(const std::vector<float3>& grid_velocities,
                               const float3& position) const;
    
    // Window functions for grid assignment
    double ngp_window(double x) const;      // Nearest grid point
    double cic_window(double x) const;      // Cloud-in-cell
    double tsc_window(double x) const;      // Triangular shaped cloud
    
    // Random number utilities
    double generate_gaussian() const;
    std::complex<double> generate_complex_gaussian() const;
    
    // Normalization and scaling
    void normalize_power_spectrum();
    double sigma8_normalization() const;
    
    // Constraint equations for special modes
    void enforce_reality_constraint(std::vector<std::complex<double>>& delta_k) const;
    void enforce_zero_mode_constraint(std::vector<std::complex<double>>& delta_k) const;
};

// Standalone utility functions
namespace initial_conditions_utils {
    
    // Convert grid-based initial conditions to particle-based
    void grid_to_particles_simple(
        const std::vector<float3>& grid_positions,
        const std::vector<float3>& grid_velocities,
        size_t grid_size,
        float box_size,
        size_t num_particles,
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses);
    
    // Generate random particle distribution (for testing)
    void generate_random_particles(
        size_t num_particles,
        float box_size,
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses,
        uint32_t seed = 12345);
    
    // Load initial conditions from file
    bool load_initial_conditions_from_file(
        const std::string& filename,
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        std::vector<float>& masses);
    
    // Save initial conditions to file
    bool save_initial_conditions_to_file(
        const std::string& filename,
        const std::vector<float3>& positions,
        const std::vector<float3>& velocities,
        const std::vector<float>& masses);
    
    // Compute power spectrum from particle positions
    std::vector<double> compute_power_spectrum_from_particles(
        const std::vector<float3>& positions,
        float box_size,
        size_t n_bins = 100);
    
    // Validate initial conditions
    bool validate_initial_conditions(
        const std::vector<float3>& positions,
        const std::vector<float3>& velocities,
        const std::vector<float>& masses,
        float box_size);
}

} // namespace physics