#include <fftw3.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "analysis/power_spectrum.hpp"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#include <cufft.h>
#endif

namespace analysis {

PowerSpectrumAnalyzer::PowerSpectrumAnalyzer(int grid_size, float box_size, bool use_gpu)
    : grid_size_(grid_size),
      box_size_(box_size),
      mass_weighted_(true),
      use_cic_(true),
      particle_mass_(1.0f) {
    dk_ = 2.0f * M_PI / box_size_;
    num_k_bins_ = grid_size_ / 2;

    // Initialize CPU arrays
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;
    density_grid_.resize(grid_total, 0.0f);
    density_field_.resize(grid_total);

    compute_k_binning();

#ifdef HAVE_CUDA
    gpu_enabled_ = use_gpu;
    if (gpu_enabled_) {
        initialize_gpu_arrays();
    }
#else
    (void)use_gpu;  // Suppress unused parameter warning
#endif
}

PowerSpectrumAnalyzer::~PowerSpectrumAnalyzer() {
#ifdef HAVE_CUDA
    if (gpu_enabled_) {
        cleanup_gpu_arrays();
    }
#endif
}

PowerSpectrumData PowerSpectrumAnalyzer::compute_power_spectrum(
    const std::vector<physics::Particle>& particles,
    bool apply_shot_noise_correction) {
    // Clear the density grid
    std::fill(density_grid_.begin(), density_grid_.end(), 0.0f);

    // Assign particles to grid
    if (use_cic_) {
        assign_particles_to_grid_cic(particles);
    } else {
        assign_particles_to_grid_ngp(particles);
    }

    // Compute density contrast
    compute_density_contrast();

    // Apply FFT
#ifdef HAVE_CUDA
    if (gpu_enabled_) {
        copy_grid_to_gpu();
        compute_fft_gpu();
        copy_field_from_gpu();
    } else {
        apply_fft_forward();
    }
#else
    apply_fft_forward();
#endif

    // Bin the power spectrum
    return bin_power_spectrum(apply_shot_noise_correction);
}

void PowerSpectrumAnalyzer::assign_particles_to_grid_cic(
    const std::vector<physics::Particle>& particles) {
    float grid_spacing = box_size_ / grid_size_;
    float inv_grid_spacing = 1.0f / grid_spacing;

    for (const auto& particle : particles) {
        // Convert position to grid coordinates
        float x = particle.position.x * inv_grid_spacing;
        float y = particle.position.y * inv_grid_spacing;
        float z = particle.position.z * inv_grid_spacing;

        // Handle periodic boundary conditions
        x = fmod(x + grid_size_, grid_size_);
        y = fmod(y + grid_size_, grid_size_);
        z = fmod(z + grid_size_, grid_size_);

        // Get the indices of the lower-left-front corner
        int ix = static_cast<int>(x);
        int iy = static_cast<int>(y);
        int iz = static_cast<int>(z);

        // Get fractional parts
        float fx = x - ix;
        float fy = y - iy;
        float fz = z - iz;

        // Cloud-in-cell weights
        float weights[2][2][2] = {
            {{(1 - fx) * (1 - fy) * (1 - fz), (1 - fx) * (1 - fy) * fz},
             {(1 - fx) * fy * (1 - fz), (1 - fx) * fy * fz}},
            {{fx * (1 - fy) * (1 - fz), fx * (1 - fy) * fz}, {fx * fy * (1 - fz), fx * fy * fz}}};

        float mass = mass_weighted_ ? particle.mass : 1.0f;

        // Distribute particle mass to 8 neighboring grid points
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int gx = (ix + dx) % grid_size_;
                    int gy = (iy + dy) % grid_size_;
                    int gz = (iz + dz) % grid_size_;

                    int idx = gx * grid_size_ * grid_size_ + gy * grid_size_ + gz;
                    density_grid_[idx] += mass * weights[dx][dy][dz];
                }
            }
        }
    }
}

void PowerSpectrumAnalyzer::assign_particles_to_grid_ngp(
    const std::vector<physics::Particle>& particles) {
    float grid_spacing = box_size_ / grid_size_;
    float inv_grid_spacing = 1.0f / grid_spacing;

    for (const auto& particle : particles) {
        // Convert position to grid coordinates
        int ix = static_cast<int>(particle.position.x * inv_grid_spacing) % grid_size_;
        int iy = static_cast<int>(particle.position.y * inv_grid_spacing) % grid_size_;
        int iz = static_cast<int>(particle.position.z * inv_grid_spacing) % grid_size_;

        // Handle negative indices (periodic boundaries)
        if (ix < 0)
            ix += grid_size_;
        if (iy < 0)
            iy += grid_size_;
        if (iz < 0)
            iz += grid_size_;

        int idx = ix * grid_size_ * grid_size_ + iy * grid_size_ + iz;
        float mass = mass_weighted_ ? particle.mass : 1.0f;
        density_grid_[idx] += mass;
    }
}

void PowerSpectrumAnalyzer::compute_density_contrast() {
    // Calculate mean density
    double total_mass = 0.0;
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;

    for (size_t i = 0; i < grid_total; ++i) {
        total_mass += density_grid_[i];
    }

    float mean_density = total_mass / grid_total;

    // Convert to density contrast: delta = (rho - rho_mean) / rho_mean
    for (size_t i = 0; i < grid_total; ++i) {
        if (mean_density > 0.0f) {
            density_grid_[i] = (density_grid_[i] - mean_density) / mean_density;
        } else {
            density_grid_[i] = 0.0f;
        }
    }
}

void PowerSpectrumAnalyzer::apply_fft_forward() {
    // Use FFTW for CPU FFT
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;

    // Create FFTW plan
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(grid_size_,
                                            grid_size_,
                                            grid_size_,
                                            density_grid_.data(),
                                            reinterpret_cast<fftwf_complex*>(density_field_.data()),
                                            FFTW_ESTIMATE);

    // Execute FFT
    fftwf_execute(plan);

    // Cleanup
    fftwf_destroy_plan(plan);

    // Normalize
    float norm = 1.0f / (grid_size_ * grid_size_ * grid_size_);
    for (size_t i = 0; i < grid_total; ++i) {
        density_field_[i] *= norm;
    }
}

PowerSpectrumData PowerSpectrumAnalyzer::bin_power_spectrum(bool apply_shot_noise_correction) {
    PowerSpectrumData result;
    result.box_size = box_size_;
    result.grid_size = grid_size_;
    result.nyquist_frequency = M_PI * grid_size_ / box_size_;

    // Initialize bins
    result.k_values = k_bin_centers_;
    result.power_values.resize(num_k_bins_, 0.0f);
    result.mode_counts.resize(num_k_bins_, 0);

    float dk = 2.0f * M_PI / box_size_;
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;

    // Loop over all k-modes
    for (int kx = 0; kx < grid_size_; ++kx) {
        for (int ky = 0; ky < grid_size_; ++ky) {
            for (int kz = 0; kz <= grid_size_ / 2; ++kz) {  // Only half due to real FFT

                // Calculate k-vector components
                float kx_val = (kx <= grid_size_ / 2) ? kx * dk : (kx - grid_size_) * dk;
                float ky_val = (ky <= grid_size_ / 2) ? ky * dk : (ky - grid_size_) * dk;
                float kz_val = kz * dk;

                // Calculate |k|
                float k_mag = sqrt(kx_val * kx_val + ky_val * ky_val + kz_val * kz_val);

                if (k_mag == 0.0f)
                    continue;  // Skip DC mode

                // Find k-bin
                int bin = static_cast<int>(k_mag / dk);
                if (bin >= num_k_bins_)
                    continue;

                // Get density field value
                int idx = kx * grid_size_ * (grid_size_ / 2 + 1) + ky * (grid_size_ / 2 + 1) + kz;
                std::complex<float> field_val = density_field_[idx];

                // Calculate power
                float power = std::norm(field_val);

                // Account for FFT symmetry (except for kz=0 and kz=Nyquist planes)
                int multiplicity = (kz == 0 || kz == grid_size_ / 2) ? 1 : 2;

                result.power_values[bin] += power * multiplicity;
                result.mode_counts[bin] += multiplicity;
            }
        }
    }

    // Normalize by number of modes and apply volume factor
    float volume = box_size_ * box_size_ * box_size_;
    for (int i = 0; i < num_k_bins_; ++i) {
        if (result.mode_counts[i] > 0) {
            result.power_values[i] /= result.mode_counts[i];
            result.power_values[i] *= volume;  // Convert to physical units
        }
    }

    // Apply shot noise correction
    if (apply_shot_noise_correction) {
        float shot_noise = volume / grid_total;  // Approximate shot noise
        result.shot_noise = shot_noise;
        for (int i = 0; i < num_k_bins_; ++i) {
            result.power_values[i] -= shot_noise;
        }
    }

    // Calculate total power
    result.total_power = 0.0;
    for (int i = 0; i < num_k_bins_; ++i) {
        if (result.mode_counts[i] > 0) {
            result.total_power += result.power_values[i] * result.mode_counts[i];
        }
    }

    return result;
}

void PowerSpectrumAnalyzer::compute_k_binning() {
    float dk = 2.0f * M_PI / box_size_;
    k_bin_centers_.resize(num_k_bins_);
    k_bin_edges_.resize(num_k_bins_ + 1);

    for (int i = 0; i <= num_k_bins_; ++i) {
        k_bin_edges_[i] = i * dk;
    }

    for (int i = 0; i < num_k_bins_; ++i) {
        k_bin_centers_[i] = (k_bin_edges_[i] + k_bin_edges_[i + 1]) * 0.5f;
    }
}

void PowerSpectrumAnalyzer::save_power_spectrum(const PowerSpectrumData& ps_data,
                                                const std::string& filename) const {
    std::ofstream file(filename);
    file << "# k [h/Mpc]  P(k) [(Mpc/h)^3]  N_modes  Error\n";
    file << "# Box size: " << ps_data.box_size << " Mpc/h\n";
    file << "# Grid size: " << ps_data.grid_size << "\n";
    file << "# Shot noise: " << ps_data.shot_noise << "\n";

    for (size_t i = 0; i < ps_data.k_values.size(); ++i) {
        if (ps_data.mode_counts[i] > 0) {
            float error = ps_data.power_values[i] / sqrt(ps_data.mode_counts[i]);
            file << ps_data.k_values[i] << " " << ps_data.power_values[i] << " "
                 << ps_data.mode_counts[i] << " " << error << "\n";
        }
    }
}

float PowerSpectrumAnalyzer::compute_sigma8(const PowerSpectrumData& ps_data, float R) const {
    double sigma2 = 0.0;
    double dk = (ps_data.k_values.size() > 1) ? (ps_data.k_values[1] - ps_data.k_values[0]) : 1.0;

    for (size_t i = 0; i < ps_data.k_values.size(); ++i) {
        if (ps_data.mode_counts[i] > 0) {
            float k = ps_data.k_values[i];
            float kR = k * R;
            float W = theory::tophat_window(kR);
            sigma2 += ps_data.power_values[i] * W * W * k * k * k * dk / (2.0 * M_PI * M_PI);
        }
    }

    return sqrt(sigma2);
}

#ifdef HAVE_CUDA
void PowerSpectrumAnalyzer::initialize_gpu_arrays() {
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;
    size_t field_size = grid_size_ * grid_size_ * (grid_size_ / 2 + 1);

    // Allocate GPU memory
    cudaMalloc(&d_density_grid_, grid_total * sizeof(float));
    cudaMalloc(&d_density_field_, field_size * sizeof(cufftComplex));

    // Create CUFFT plan
    cufftPlan3d(&fft_plan_, grid_size_, grid_size_, grid_size_, CUFFT_R2C);

    // Create CUDA stream
    cudaStreamCreate(&stream_);
    cufftSetStream(fft_plan_, stream_);
}

void PowerSpectrumAnalyzer::cleanup_gpu_arrays() {
    if (d_density_grid_) {
        cudaFree(d_density_grid_);
        d_density_grid_ = nullptr;
    }
    if (d_density_field_) {
        cudaFree(d_density_field_);
        d_density_field_ = nullptr;
    }
    if (fft_plan_) {
        cufftDestroy(fft_plan_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void PowerSpectrumAnalyzer::copy_grid_to_gpu() {
    size_t grid_total = grid_size_ * grid_size_ * grid_size_;
    cudaMemcpyAsync(d_density_grid_,
                    density_grid_.data(),
                    grid_total * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_);
}

void PowerSpectrumAnalyzer::copy_field_from_gpu() {
    size_t field_size = grid_size_ * grid_size_ * (grid_size_ / 2 + 1);
    cudaMemcpyAsync(density_field_.data(),
                    d_density_field_,
                    field_size * sizeof(cufftComplex),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
}

void PowerSpectrumAnalyzer::compute_fft_gpu() {
    cufftExecR2C(fft_plan_, d_density_grid_, d_density_field_);

    // Normalize on GPU
    size_t field_size = grid_size_ * grid_size_ * (grid_size_ / 2 + 1);
    float norm = 1.0f / (grid_size_ * grid_size_ * grid_size_);

    // Simple kernel to normalize (would need proper CUDA kernel implementation)
    cudaDeviceSynchronize();
}
#endif

// Theory namespace implementations
namespace theory {

float tophat_window(float kR) {
    if (kR < 1e-6f)
        return 1.0f;
    return 3.0f * (sin(kR) - kR * cos(kR)) / (kR * kR * kR);
}

float gaussian_window(float kR) {
    return exp(-0.5f * kR * kR);
}

float linear_growth_factor(float z, float omega_m, float omega_lambda) {
    float a = 1.0f / (1.0f + z);
    float omega_m_a = omega_m / (omega_m + omega_lambda * a * a * a);
    float omega_l_a = 1.0f - omega_m_a;

    // Approximate growth factor (Carroll, Press & Turner 1992)
    float g = 2.5f * omega_m_a /
              (pow(omega_m_a, 4.0f / 7.0f) - omega_l_a +
               (1.0f + omega_m_a / 2.0f) * (1.0f + omega_l_a / 70.0f));

    return g * a;
}

std::vector<float> eisenstein_hu_power_spectrum(const std::vector<float>& k_values,
                                                float omega_m,
                                                float omega_b,
                                                float h,
                                                float sigma8,
                                                float n_s) {
    std::vector<float> power_spectrum(k_values.size());

    // Eisenstein & Hu (1998) fitting formulae
    float theta_cmb = 2.7f / 2.7f;  // CMB temperature in units of 2.7K
    float omega_m_h2 = omega_m * h * h;
    float omega_b_h2 = omega_b * h * h;
    float f_baryon = omega_b / omega_m;

    // Sound horizon and silk damping scale
    float z_eq = 2.5e4f * omega_m_h2 * pow(theta_cmb, -4.0f);
    float k_eq = 7.46e-2f * omega_m_h2 * pow(theta_cmb, -2.0f);  // h/Mpc

    float z_drag =
        1291.0f * pow(omega_m_h2, 0.251f) / (1.0f + 0.659f * pow(omega_m_h2, 0.828f)) *
        (1.0f + 0.313f * pow(omega_m_h2, -0.419f) * (1.0f + 0.607f * pow(omega_m_h2, 0.674f)));
    z_drag = z_drag / (1.0f + 0.238f * pow(omega_m_h2, 0.223f));

    float R_drag = 31.5f * omega_b_h2 * pow(theta_cmb, -4.0f) * (1000.0f / z_drag);
    float R_eq = 31.5f * omega_b_h2 * pow(theta_cmb, -4.0f) * (1000.0f / z_eq);

    float sound_horizon = 2.0f / (3.0f * k_eq) * sqrt(6.0f / R_eq) *
                          log((sqrt(1.0f + R_drag) + sqrt(R_drag + R_eq)) / (1.0f + sqrt(R_eq)));

    // Compute transfer function and power spectrum
    for (size_t i = 0; i < k_values.size(); ++i) {
        float k = k_values[i];
        float q = k / (13.41f * k_eq);
        float ks = k * sound_horizon;

        // CDM transfer function
        float T_c = f_baryon * log(1.8f * q) / (log(1.8f * q) + pow(14.2f * q, 2.0f)) +
                    (1.0f - f_baryon) * log(1.8f * q) / (log(1.8f * q) + pow(14.2f * q, 2.0f));

        // Baryon oscillations
        float alpha_c = pow(46.9f * omega_m_h2, 0.670f) * (1.0f + pow(32.1f * omega_m_h2, -0.532f));
        float beta_c = pow(12.0f * omega_m_h2, 0.424f) * (1.0f + pow(45.0f * omega_m_h2, -0.582f));
        float f = 1.0f / (1.0f + pow(ks / 5.4f, 4.0f));

        float T_b = (log(1.8f * q) / (log(1.8f * q) + pow(14.2f * q, 2.0f)) * f +
                     alpha_c * exp(-pow(ks / beta_c, 1.4f)) * (1.0f - f)) /
                    (1.0f + pow(ks / 5.2f, 2.0f));

        float T_total = f_baryon * T_b + (1.0f - f_baryon) * T_c;

        // Primordial power spectrum
        float primordial = pow(k, n_s);

        // Total power spectrum (need to normalize to sigma8)
        power_spectrum[i] = primordial * T_total * T_total;
    }

    // Normalize to sigma8 (simplified)
    float norm = sigma8 * sigma8 / 0.8f;  // Approximate normalization
    for (float& p : power_spectrum) {
        p *= norm;
    }

    return power_spectrum;
}

}  // namespace theory

}  // namespace analysis