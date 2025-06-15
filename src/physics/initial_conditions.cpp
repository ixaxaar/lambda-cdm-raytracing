#include "physics/initial_conditions.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <fstream>

namespace physics {

InitialConditionsGenerator::InitialConditionsGenerator(
    const InitialConditionsParams& params,
    const CosmologyModel& cosmology)
    : params_(params)
    , cosmology_(cosmology)
    , rng_(params.random_seed)
    , normal_dist_(0.0, 1.0)
    , power_spectrum_initialized_(false)
{
    n_total_ = params_.grid_size * params_.grid_size * params_.grid_size;
    dx_ = params_.box_size / params_.grid_size;
    dk_ = 2.0 * M_PI / params_.box_size;
    
    // Initialize power spectrum
    initialize_power_spectrum();
    
    // Setup FFT workspace
    initialize_fft_workspace();
    
    std::cout << "InitialConditionsGenerator initialized:\n"
              << "  Grid size: " << params_.grid_size << "³\n"
              << "  Box size: " << params_.box_size << " Mpc/h\n"
              << "  Initial redshift: " << params_.z_initial << "\n"
              << "  dx = " << dx_ << " Mpc/h, dk = " << dk_ << " h/Mpc" << std::endl;
}

void InitialConditionsGenerator::initialize_power_spectrum() {
    const int n_k = 1000;
    k_values_.resize(n_k);
    power_values_.resize(n_k);
    
    // Logarithmic spacing in k
    double log_k_min = std::log(params_.k_min);
    double log_k_max = std::log(params_.k_max);
    double dlog_k = (log_k_max - log_k_min) / (n_k - 1);
    
    for (int i = 0; i < n_k; i++) {
        double log_k = log_k_min + i * dlog_k;
        double k = std::exp(log_k);
        k_values_[i] = k;
        
        // Compute power spectrum based on type
        switch (params_.ps_type) {
            case PowerSpectrumType::EISENSTEIN_HU:
                power_values_[i] = eisenstein_hu_power_spectrum(k);
                break;
            case PowerSpectrumType::CDM_ONLY:
                power_values_[i] = cdm_power_spectrum(k);
                break;
            case PowerSpectrumType::CUSTOM:
                if (params_.custom_power_spectrum) {
                    power_values_[i] = params_.custom_power_spectrum(k);
                } else {
                    throw std::runtime_error("Custom power spectrum function not provided");
                }
                break;
            default:
                power_values_[i] = eisenstein_hu_power_spectrum(k);
                break;
        }
    }
    
    power_spectrum_initialized_ = true;
    
    // Normalize to sigma_8 if requested
    if (params_.normalize_at_z0) {
        normalize_power_spectrum();
    }
    
    std::cout << "Power spectrum initialized with " << n_k << " k-modes\n"
              << "  k range: [" << params_.k_min << ", " << params_.k_max << "] h/Mpc" << std::endl;
}

double InitialConditionsGenerator::eisenstein_hu_transfer(double k) const {
    // Eisenstein & Hu (1998) transfer function
    const CosmologyParams& cosmo = cosmology_.get_params();
    
    // Simplified CDM transfer function
    double Gamma = cosmo.omega_m * cosmo.h;
    double q = k / (Gamma);
    
    double T_cdm = std::log(1.0 + 2.34 * q) / (2.34 * q) *
                   std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) + 
                           std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
    
    return T_cdm;
}

double InitialConditionsGenerator::eisenstein_hu_power_spectrum(double k) const {
    // Use simplified transfer function and primordial power spectrum
    const CosmologyParams& cosmo = cosmology_.get_params();
    
    double T_k = eisenstein_hu_transfer(k);
    double primordial = std::pow(k, cosmo.n_s);
    
    return primordial * T_k * T_k;
}

double InitialConditionsGenerator::cdm_transfer_function(double k) const {
    // Transfer function for CDM (Bardeen et al. 1986)
    const CosmologyParams& cosmo = cosmology_.get_params();
    double Gamma = cosmo.omega_m * cosmo.h;
    double q = k / (Gamma);
    
    double T_cdm = std::log(1.0 + 2.34 * q) / (2.34 * q) *
                   std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) + 
                           std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
    
    return T_cdm;
}

double InitialConditionsGenerator::cdm_power_spectrum(double k) const {
    // Simple CDM-only power spectrum
    const CosmologyParams& cosmo = cosmology_.get_params();
    
    double T_cdm = cdm_transfer_function(k);
    double primordial = std::pow(k, cosmo.n_s);
    
    return primordial * T_cdm * T_cdm;
}

void InitialConditionsGenerator::normalize_power_spectrum() {
    // Compute current sigma_8 and rescale
    double current_sigma8 = compute_sigma8();
    double target_sigma8 = cosmology_.get_params().sigma_8;
    
    double normalization = std::pow(target_sigma8 / current_sigma8, 2);
    
    for (auto& power : power_values_) {
        power *= normalization;
    }
    
    std::cout << "Power spectrum normalized: sigma_8 = " << target_sigma8 
              << " (factor: " << std::sqrt(normalization) << ")" << std::endl;
}

double InitialConditionsGenerator::compute_sigma8() const {
    // Compute σ₈ by integrating power spectrum with top-hat window
    const double R8 = 8.0; // 8 Mpc/h sphere
    double sigma8_squared = 0.0;
    
    const int n_integration = 1000;
    double log_k_min = std::log(0.001);
    double log_k_max = std::log(100.0);
    double dlog_k = (log_k_max - log_k_min) / n_integration;
    
    for (int i = 0; i < n_integration; i++) {
        double log_k = log_k_min + (i + 0.5) * dlog_k;
        double k = std::exp(log_k);
        double kR = k * R8;
        
        // Top-hat window function
        double W = 3.0 * (std::sin(kR) - kR * std::cos(kR)) / (kR * kR * kR);
        
        double power = get_power_spectrum(k);
        sigma8_squared += power * W * W * k * k * k * dlog_k;
    }
    
    sigma8_squared *= 1.0 / (2.0 * M_PI * M_PI);
    
    return std::sqrt(sigma8_squared);
}

double InitialConditionsGenerator::get_power_spectrum(double k) const {
    if (!power_spectrum_initialized_) {
        throw std::runtime_error("Power spectrum not initialized");
    }
    
    // Linear interpolation in log space
    if (k <= k_values_.front()) return power_values_.front();
    if (k >= k_values_.back()) return power_values_.back();
    
    // Find bracketing indices
    auto it = std::lower_bound(k_values_.begin(), k_values_.end(), k);
    size_t i = std::distance(k_values_.begin(), it);
    
    if (i == 0) return power_values_[0];
    if (i >= k_values_.size()) return power_values_.back();
    
    // Linear interpolation in log-log space
    double log_k = std::log(k);
    double log_k1 = std::log(k_values_[i-1]);
    double log_k2 = std::log(k_values_[i]);
    double log_p1 = std::log(power_values_[i-1]);
    double log_p2 = std::log(power_values_[i]);
    
    double log_p = log_p1 + (log_p2 - log_p1) * (log_k - log_k1) / (log_k2 - log_k1);
    
    return std::exp(log_p);
}

void InitialConditionsGenerator::generate_initial_conditions(
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses) const {
    
    std::cout << "Generating initial conditions using Zel'dovich approximation..." << std::endl;
    
    // Generate Gaussian density field
    std::vector<std::complex<double>> delta_k(n_total_);
    generate_gaussian_field(delta_k);
    
    // Apply Zel'dovich approximation
    std::vector<float3> grid_positions(n_total_);
    std::vector<float3> grid_velocities(n_total_);
    
    apply_zeldovich_approximation(delta_k, grid_positions, grid_velocities);
    
    // Convert grid to particles
    size_t num_particles = positions.size();
    grid_to_particles(grid_positions, grid_velocities, num_particles, 
                     positions, velocities, masses);
    
    std::cout << "Initial conditions generated for " << num_particles << " particles" << std::endl;
    print_statistics(positions, velocities);
}

void InitialConditionsGenerator::generate_gaussian_field(
    std::vector<std::complex<double>>& delta_k) const {
    
    delta_k.resize(n_total_);
    
    for (size_t flat_idx = 0; flat_idx < n_total_; flat_idx++) {
        size_t i, j, k;
        unflatten_index(flat_idx, i, j, k);
        
        float3 k_vec = grid_index_to_k_vector(i, j, k);
        double k_mag = k_vector_magnitude(k_vec);
        
        if (k_mag > 0.0) {
            // Generate complex Gaussian random field
            std::complex<double> gaussian_amp = generate_complex_gaussian();
            
            // Scale by power spectrum
            double power = get_power_spectrum(k_mag);
            double amplitude = std::sqrt(power * params_.box_size * params_.box_size * params_.box_size);
            
            delta_k[flat_idx] = gaussian_amp * amplitude;
        } else {
            // DC mode - set to zero for mean-zero field
            delta_k[flat_idx] = std::complex<double>(0.0, 0.0);
        }
    }
    
    // Enforce reality constraint (delta_k(-k) = delta_k*(k))
    enforce_reality_constraint(delta_k);
    
    std::cout << "Generated Gaussian density field in Fourier space" << std::endl;
}

void InitialConditionsGenerator::apply_zeldovich_approximation(
    const std::vector<std::complex<double>>& delta_k,
    std::vector<float3>& positions,
    std::vector<float3>& velocities) const {
    
    // Compute displacement field
    std::vector<float3> displacements(n_total_);
    compute_displacement_field(delta_k, displacements);
    
    // Compute velocity field
    compute_velocity_field(delta_k, velocities);
    
    // Set up regular grid and apply displacements
    for (size_t flat_idx = 0; flat_idx < n_total_; flat_idx++) {
        size_t i, j, k;
        unflatten_index(flat_idx, i, j, k);
        
        // Regular grid position
        float3 grid_pos = grid_index_to_position(i, j, k);
        
        // Apply Zel'dovich displacement
        positions[flat_idx].x = grid_pos.x + displacements[flat_idx].x;
        positions[flat_idx].y = grid_pos.y + displacements[flat_idx].y;
        positions[flat_idx].z = grid_pos.z + displacements[flat_idx].z;
        
        // Apply periodic boundary conditions (ensure positive values)
        while (positions[flat_idx].x < 0.0f) positions[flat_idx].x += params_.box_size;
        while (positions[flat_idx].x >= params_.box_size) positions[flat_idx].x -= params_.box_size;
        while (positions[flat_idx].y < 0.0f) positions[flat_idx].y += params_.box_size;
        while (positions[flat_idx].y >= params_.box_size) positions[flat_idx].y -= params_.box_size;
        while (positions[flat_idx].z < 0.0f) positions[flat_idx].z += params_.box_size;
        while (positions[flat_idx].z >= params_.box_size) positions[flat_idx].z -= params_.box_size;
    }
    
    std::cout << "Applied Zel'dovich approximation to generate positions and velocities" << std::endl;
}

void InitialConditionsGenerator::compute_displacement_field(
    const std::vector<std::complex<double>>& delta_k,
    std::vector<float3>& displacements) const {
    
    displacements.resize(n_total_);
    
    // Compute displacement as Psi = -i k_i / k^2 * delta_k
    for (size_t flat_idx = 0; flat_idx < n_total_; flat_idx++) {
        size_t i, j, k;
        unflatten_index(flat_idx, i, j, k);
        
        float3 k_vec = grid_index_to_k_vector(i, j, k);
        double k_mag = k_vector_magnitude(k_vec);
        
        if (k_mag > 0.0) {
            std::complex<double> factor = -std::complex<double>(0.0, 1.0) / (k_mag * k_mag);
            std::complex<double> delta = delta_k[flat_idx];
            
            // Growth factor scaling
            double D = get_growth_factor();
            
            displacements[flat_idx].x = (factor * double(k_vec.x) * delta * D).real();
            displacements[flat_idx].y = (factor * double(k_vec.y) * delta * D).real();
            displacements[flat_idx].z = (factor * double(k_vec.z) * delta * D).real();
        } else {
            displacements[flat_idx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

void InitialConditionsGenerator::compute_velocity_field(
    const std::vector<std::complex<double>>& delta_k,
    std::vector<float3>& velocities) const {
    
    velocities.resize(n_total_);
    
    // Velocity field: v = a H f Psi (where f is growth rate)
    double a_init = cosmology_.z_to_a(params_.z_initial);
    double H_init = cosmology_.hubble_parameter_a(a_init);
    double f_init = get_growth_rate();
    double velocity_factor = a_init * H_init * f_init * get_velocity_scaling();
    
    // First compute displacement field for velocities
    std::vector<float3> vel_displacements(n_total_);
    compute_displacement_field(delta_k, vel_displacements);
    
    // Scale by velocity factor
    for (size_t flat_idx = 0; flat_idx < n_total_; flat_idx++) {
        velocities[flat_idx].x = vel_displacements[flat_idx].x * velocity_factor;
        velocities[flat_idx].y = vel_displacements[flat_idx].y * velocity_factor;
        velocities[flat_idx].z = vel_displacements[flat_idx].z * velocity_factor;
    }
}

void InitialConditionsGenerator::grid_to_particles(
    const std::vector<float3>& grid_positions,
    const std::vector<float3>& grid_velocities,
    size_t num_particles,
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses) const {
    
    positions.resize(num_particles);
    velocities.resize(num_particles);
    masses.resize(num_particles);
    
    // For simplicity, subsample the grid uniformly
    size_t grid_total = params_.grid_size * params_.grid_size * params_.grid_size;
    size_t skip = std::max(size_t(1), grid_total / num_particles);
    
    size_t particle_idx = 0;
    for (size_t grid_idx = 0; grid_idx < grid_total && particle_idx < num_particles; grid_idx += skip) {
        positions[particle_idx] = grid_positions[grid_idx];
        velocities[particle_idx] = grid_velocities[grid_idx];
        masses[particle_idx] = 1.0f; // Equal mass particles
        particle_idx++;
    }
    
    // Fill remaining particles if needed
    while (particle_idx < num_particles) {
        size_t random_idx = rng_() % grid_total;
        positions[particle_idx] = grid_positions[random_idx];
        velocities[particle_idx] = grid_velocities[random_idx];
        masses[particle_idx] = 1.0f;
        particle_idx++;
    }
    
    std::cout << "Converted " << grid_total << " grid points to " 
              << num_particles << " particles" << std::endl;
}

double InitialConditionsGenerator::get_growth_factor() const {
    double a_init = cosmology_.z_to_a(params_.z_initial);
    return cosmology_.growth_factor(a_init);
}

double InitialConditionsGenerator::get_growth_rate() const {
    double a_init = cosmology_.z_to_a(params_.z_initial);
    return cosmology_.growth_rate(a_init);
}

double InitialConditionsGenerator::get_velocity_scaling() const {
    // Convert from comoving to physical velocities
    // In Zel'dovich: v_physical = a * H(a) * f(a) * displacement
    return 1.0; // Already included in compute_velocity_field
}

// Utility methods
float3 InitialConditionsGenerator::grid_index_to_position(size_t i, size_t j, size_t k) const {
    return make_float3(
        (i + 0.5f) * dx_,
        (j + 0.5f) * dx_,
        (k + 0.5f) * dx_
    );
}

float3 InitialConditionsGenerator::grid_index_to_k_vector(size_t i, size_t j, size_t k) const {
    // Convert grid indices to wavevector components
    int ni = (i <= params_.grid_size / 2) ? int(i) : int(i) - int(params_.grid_size);
    int nj = (j <= params_.grid_size / 2) ? int(j) : int(j) - int(params_.grid_size);
    int nk = (k <= params_.grid_size / 2) ? int(k) : int(k) - int(params_.grid_size);
    
    return make_float3(
        ni * dk_,
        nj * dk_,
        nk * dk_
    );
}

double InitialConditionsGenerator::k_vector_magnitude(const float3& k_vec) const {
    return std::sqrt(k_vec.x * k_vec.x + k_vec.y * k_vec.y + k_vec.z * k_vec.z);
}

std::complex<double> InitialConditionsGenerator::generate_complex_gaussian() const {
    double real_part = normal_dist_(rng_);
    double imag_part = normal_dist_(rng_);
    return std::complex<double>(real_part, imag_part) / std::sqrt(2.0);
}

void InitialConditionsGenerator::enforce_reality_constraint(
    std::vector<std::complex<double>>& delta_k) const {
    
    // For real fields, delta_k(-k) = delta_k*(k)
    // This is automatically satisfied if we generate the field correctly
    // For now, we'll implement a simple symmetric generation
    
    size_t n = params_.grid_size;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                // Find the complex conjugate mode
                size_t i_conj = (i == 0) ? 0 : n - i;
                size_t j_conj = (j == 0) ? 0 : n - j;
                size_t k_conj = (k == 0) ? 0 : n - k;
                
                size_t idx = flatten_index(i, j, k);
                size_t idx_conj = flatten_index(i_conj, j_conj, k_conj);
                
                if (idx != idx_conj && idx < idx_conj) {
                    // Make them complex conjugates
                    std::complex<double> avg = 0.5 * (delta_k[idx] + std::conj(delta_k[idx_conj]));
                    delta_k[idx] = avg;
                    delta_k[idx_conj] = std::conj(avg);
                }
            }
        }
    }
}

void InitialConditionsGenerator::initialize_fft_workspace() {
    fourier_workspace_.resize(n_total_);
}

void InitialConditionsGenerator::print_statistics(
    const std::vector<float3>& positions,
    const std::vector<float3>& velocities) const {
    
    double rms_displacement = compute_rms_displacement(positions);
    double rms_velocity = compute_rms_velocity(velocities);
    
    std::cout << "Initial conditions statistics:\n"
              << "  RMS displacement: " << rms_displacement << " Mpc/h\n"
              << "  RMS velocity: " << rms_velocity << " km/s\n"
              << "  Box size: " << params_.box_size << " Mpc/h\n"
              << "  Initial redshift: " << params_.z_initial << std::endl;
}

double InitialConditionsGenerator::compute_rms_displacement(
    const std::vector<float3>& positions) const {
    
    double sum_sq = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < positions.size(); i++) {
        // Compute displacement from regular grid
        size_t gi = i % params_.grid_size;
        size_t gj = (i / params_.grid_size) % params_.grid_size;
        size_t gk = i / (params_.grid_size * params_.grid_size);
        
        if (gk < params_.grid_size) {
            float3 grid_pos = grid_index_to_position(gi, gj, gk);
            
            float dx = positions[i].x - grid_pos.x;
            float dy = positions[i].y - grid_pos.y;
            float dz = positions[i].z - grid_pos.z;
            
            // Handle periodic boundary conditions
            if (dx > params_.box_size / 2) dx -= params_.box_size;
            if (dx < -params_.box_size / 2) dx += params_.box_size;
            if (dy > params_.box_size / 2) dy -= params_.box_size;
            if (dy < -params_.box_size / 2) dy += params_.box_size;
            if (dz > params_.box_size / 2) dz -= params_.box_size;
            if (dz < -params_.box_size / 2) dz += params_.box_size;
            
            sum_sq += dx*dx + dy*dy + dz*dz;
            count++;
        }
    }
    
    return (count > 0) ? std::sqrt(sum_sq / count) : 0.0;
}

double InitialConditionsGenerator::compute_rms_velocity(
    const std::vector<float3>& velocities) const {
    
    double sum_sq = 0.0;
    
    for (const auto& vel : velocities) {
        sum_sq += vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
    }
    
    return std::sqrt(sum_sq / velocities.size());
}

void InitialConditionsGenerator::generate_particles(
    size_t num_particles,
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses) const {
    
    // Resize output vectors
    positions.resize(num_particles);
    velocities.resize(num_particles);
    masses.resize(num_particles);
    
    // Generate initial conditions and subsample
    generate_initial_conditions(positions, velocities, masses);
}

void InitialConditionsGenerator::validate_power_spectrum() const {
    if (!power_spectrum_initialized_) {
        throw std::runtime_error("Power spectrum not initialized");
    }
    
    std::cout << "Power spectrum validation:\n"
              << "  Number of k-modes: " << k_values_.size() << "\n"
              << "  k range: [" << k_values_.front() << ", " << k_values_.back() << "] h/Mpc\n"
              << "  Computed sigma_8: " << compute_sigma8() << "\n"
              << "  Target sigma_8: " << cosmology_.get_params().sigma_8 << std::endl;
}

void InitialConditionsGenerator::apply_second_order_corrections(
    std::vector<float3>& positions,
    std::vector<float3>& velocities) const {
    // Placeholder for 2LPT implementation
    (void)positions; (void)velocities;
    std::cout << "Second-order corrections not yet implemented" << std::endl;
}

void InitialConditionsGenerator::generate_glass_positions(std::vector<float3>& positions) const {
    // Simple random positions for now - could implement proper glass generation
    positions.clear();
    size_t n_glass = params_.grid_size * params_.grid_size * params_.grid_size;
    positions.resize(n_glass);
    
    std::uniform_real_distribution<float> uniform(0.0f, params_.box_size);
    for (auto& pos : positions) {
        pos.x = uniform(rng_);
        pos.y = uniform(rng_);
        pos.z = uniform(rng_);
    }
}

void InitialConditionsGenerator::relax_glass_configuration(
    std::vector<float3>& positions, int n_iterations) const {
    // Placeholder for glass relaxation
    (void)positions; (void)n_iterations;
    std::cout << "Glass relaxation not yet implemented" << std::endl;
}

void InitialConditionsGenerator::setup_k_space_grid() {
    // Already handled in constructor
}

// Namespace utility functions
namespace initial_conditions_utils {

void grid_to_particles_simple(
    const std::vector<float3>& grid_positions,
    const std::vector<float3>& grid_velocities,
    size_t grid_size,
    float box_size,
    size_t num_particles,
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses) {
    
    positions.resize(num_particles);
    velocities.resize(num_particles);
    masses.resize(num_particles);
    
    size_t grid_total = grid_size * grid_size * grid_size;
    size_t skip = std::max(size_t(1), grid_total / num_particles);
    
    size_t particle_idx = 0;
    for (size_t grid_idx = 0; grid_idx < grid_total && particle_idx < num_particles; grid_idx += skip) {
        if (grid_idx < grid_positions.size()) {
            positions[particle_idx] = grid_positions[grid_idx];
            velocities[particle_idx] = grid_velocities[grid_idx];
            masses[particle_idx] = 1.0f;
            particle_idx++;
        }
    }
    
    // Fill remaining with uniform mass
    while (particle_idx < num_particles) {
        masses[particle_idx] = 1.0f;
        particle_idx++;
    }
}

void generate_random_particles(
    size_t num_particles,
    float box_size,
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses,
    uint32_t seed) {
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, box_size);
    std::normal_distribution<float> normal(0.0f, 100.0f); // 100 km/s velocity dispersion
    
    positions.resize(num_particles);
    velocities.resize(num_particles);
    masses.resize(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        positions[i] = make_float3(uniform(rng), uniform(rng), uniform(rng));
        velocities[i] = make_float3(normal(rng), normal(rng), normal(rng));
        masses[i] = 1.0f;
    }
}

bool validate_initial_conditions(
    const std::vector<float3>& positions,
    const std::vector<float3>& velocities,
    const std::vector<float>& masses,
    float box_size) {
    
    if (positions.size() != velocities.size() || positions.size() != masses.size()) {
        std::cerr << "Error: Array sizes don't match" << std::endl;
        return false;
    }
    
    // Check bounds
    for (const auto& pos : positions) {
        if (pos.x < 0 || pos.x >= box_size ||
            pos.y < 0 || pos.y >= box_size ||
            pos.z < 0 || pos.z >= box_size) {
            std::cerr << "Error: Particle outside box bounds" << std::endl;
            return false;
        }
    }
    
    // Check for reasonable masses
    for (float mass : masses) {
        if (mass <= 0.0f || !std::isfinite(mass)) {
            std::cerr << "Error: Invalid particle mass" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool load_initial_conditions_from_file(
    const std::string& filename,
    std::vector<float3>& positions,
    std::vector<float3>& velocities,
    std::vector<float>& masses) {
    
    // Placeholder implementation
    (void)filename; (void)positions; (void)velocities; (void)masses;
    std::cerr << "File I/O not yet implemented" << std::endl;
    return false;
}

bool save_initial_conditions_to_file(
    const std::string& filename,
    const std::vector<float3>& positions,
    const std::vector<float3>& velocities,
    const std::vector<float>& masses) {
    
    // Placeholder implementation
    (void)filename; (void)positions; (void)velocities; (void)masses;
    std::cerr << "File I/O not yet implemented" << std::endl;
    return false;
}

std::vector<double> compute_power_spectrum_from_particles(
    const std::vector<float3>& positions,
    float box_size,
    size_t n_bins) {
    
    // Placeholder implementation
    (void)positions; (void)box_size;
    std::vector<double> power_spectrum(n_bins, 0.0);
    std::cerr << "Power spectrum computation from particles not yet implemented" << std::endl;
    return power_spectrum;
}

} // namespace initial_conditions_utils

} // namespace physics