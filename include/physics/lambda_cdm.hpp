#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace physics {

struct CosmologyParams {
    double omega_m = 0.31;      // Matter density parameter
    double omega_lambda = 0.69;  // Dark energy density parameter
    double omega_b = 0.049;      // Baryon density parameter
    double h = 0.67;             // Hubble parameter
    double sigma_8 = 0.81;       // Amplitude of matter fluctuations
    double n_s = 0.96;           // Spectral index
};

struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    int id;
};

class LambdaCDMSimulation {
private:
    CosmologyParams params_;
    std::vector<Particle> particles_;
    std::unique_ptr<float[]> forces_;

    float* d_positions_;
    float* d_velocities_;
    float* d_masses_;
    float* d_forces_;

    size_t num_particles_;
    float box_size_;
    float time_step_;
    double scale_factor_;

public:
    LambdaCDMSimulation(size_t num_particles, float box_size, const CosmologyParams& params);
    ~LambdaCDMSimulation();

    void initialize_particles();
    void step(double dt);
    void compute_forces();
    void update_positions(double dt);
    void update_scale_factor(double dt);

    double hubble_function(double a) const;
    double growth_factor(double a) const;

    const std::vector<Particle>& get_particles() const { return particles_; }
    double get_scale_factor() const { return scale_factor_; }
    double get_redshift() const { return 1.0 / scale_factor_ - 1.0; }
};

}