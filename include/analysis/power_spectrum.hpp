#pragma once

#include "physics/lambda_cdm.hpp"
#include <vector>
#include <complex>
#include <memory>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace analysis {

struct PowerSpectrumData {
    std::vector<float> k_values;      // Wavenumber bins
    std::vector<float> power_values;  // P(k) values
    std::vector<int> mode_counts;     // Number of modes in each bin
    float box_size;
    int grid_size;
    float nyquist_frequency;
    double total_power;
    double shot_noise;
};

class PowerSpectrumAnalyzer {
private:
    int grid_size_;
    float box_size_;
    float dk_;
    int num_k_bins_;
    
    // CPU arrays
    std::vector<std::complex<float>> density_field_;
    std::vector<float> density_grid_;
    
#ifdef HAVE_CUDA
    // GPU arrays
    float* d_density_grid_;
    cufftComplex* d_density_field_;
    cufftHandle fft_plan_;
    cudaStream_t stream_;
    bool gpu_enabled_;
#endif

    std::vector<float> k_bin_centers_;
    std::vector<float> k_bin_edges_;

public:
    PowerSpectrumAnalyzer(int grid_size, float box_size, bool use_gpu = true);
    ~PowerSpectrumAnalyzer();

    // Main analysis functions
    PowerSpectrumData compute_power_spectrum(const std::vector<physics::Particle>& particles,
                                           bool apply_shot_noise_correction = true);
    
    // Cross power spectrum between two particle sets
    PowerSpectrumData compute_cross_power_spectrum(const std::vector<physics::Particle>& particles1,
                                                  const std::vector<physics::Particle>& particles2);

    // Specialized power spectrum calculations
    PowerSpectrumData compute_real_space_power_spectrum(const std::vector<physics::Particle>& particles);
    PowerSpectrumData compute_redshift_space_power_spectrum(const std::vector<physics::Particle>& particles,
                                                          const float3& observer_direction);

    // Utility functions
    void set_k_binning(int num_bins, float k_min = 0.0f, float k_max = -1.0f);
    void apply_mass_weighting(bool enable) { mass_weighted_ = enable; }
    void set_cloud_in_cell_assignment(bool enable) { use_cic_ = enable; }

    // Analysis tools
    std::vector<float> compute_multipoles(const std::vector<physics::Particle>& particles,
                                        const float3& observer_direction,
                                        const std::vector<int>& l_values = {0, 2, 4});

    float compute_sigma8(const PowerSpectrumData& ps_data, float R = 8.0f) const;
    float compute_effective_spectral_index(const PowerSpectrumData& ps_data,
                                         float k_ref = 0.05f) const;

    // I/O functions
    void save_power_spectrum(const PowerSpectrumData& ps_data, const std::string& filename) const;
    PowerSpectrumData load_power_spectrum(const std::string& filename) const;

    // Getters
    int get_grid_size() const { return grid_size_; }
    float get_box_size() const { return box_size_; }
    float get_fundamental_frequency() const { return 2.0f * M_PI / box_size_; }

private:
    // Core computation functions
    void assign_particles_to_grid(const std::vector<physics::Particle>& particles);
    void assign_particles_to_grid_cic(const std::vector<physics::Particle>& particles);
    void assign_particles_to_grid_ngp(const std::vector<physics::Particle>& particles);
    
    void compute_density_contrast();
    void apply_fft_forward();
    void apply_fft_backward();
    
    PowerSpectrumData bin_power_spectrum(bool apply_shot_noise_correction);
    void compute_k_binning();
    
    // GPU-specific functions
#ifdef HAVE_CUDA
    void initialize_gpu_arrays();
    void cleanup_gpu_arrays();
    void copy_grid_to_gpu();
    void copy_field_from_gpu();
    void compute_fft_gpu();
#endif

    // Analysis parameters
    bool mass_weighted_;
    bool use_cic_;  // Cloud-in-cell vs nearest-grid-point
    float particle_mass_;
};

// Utility functions for theoretical power spectra
namespace theory {
    // Linear matter power spectrum (Eisenstein & Hu 1998)
    std::vector<float> eisenstein_hu_power_spectrum(const std::vector<float>& k_values,
                                                   float omega_m, float omega_b, float h,
                                                   float sigma8, float n_s = 0.96f);

    // BBKS transfer function
    std::vector<float> bbks_transfer_function(const std::vector<float>& k_values,
                                            float omega_m, float h);

    // Linear growth factor
    float linear_growth_factor(float z, float omega_m, float omega_lambda);
    
    // Variance in spheres of radius R
    float sigma_R(const PowerSpectrumData& ps_data, float R);
    
    // Window functions
    float tophat_window(float kR);
    float gaussian_window(float kR);
}

}