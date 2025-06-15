#include "physics/lambda_cdm.hpp"
#include "physics/lambda_cdm_kernels.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace physics {

// Initialize CUDA random number generator
__global__ void init_curand_states(curandState* states, int num_particles, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// Generate initial particle positions and velocities
__global__ void generate_initial_conditions(
    float4* positions,
    float3* velocities, 
    curandState* states,
    int num_particles,
    float box_size,
    float vel_dispersion) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_particles) return;
    
    curandState local_state = states[id];
    
    // Random uniform positions
    positions[id].x = curand_uniform(&local_state) * box_size;
    positions[id].y = curand_uniform(&local_state) * box_size;
    positions[id].z = curand_uniform(&local_state) * box_size;
    positions[id].w = 1.0f;  // Mass (normalized)
    
    // Random Gaussian velocities
    velocities[id].x = curand_normal(&local_state) * vel_dispersion;
    velocities[id].y = curand_normal(&local_state) * vel_dispersion;
    velocities[id].z = curand_normal(&local_state) * vel_dispersion;
    
    states[id] = local_state;
}

// Implementation class that contains all CUDA-specific code
class LambdaCDMSimulationImpl {
public:
    // Simulation parameters
    CosmologyParams params_;
    std::unique_ptr<CosmologyModel> cosmology_;
    size_t num_particles_;
    float box_size_;
    double time_step_;
    double scale_factor_;
    float softening_;
    size_t current_step_;
    
    // GPU memory pointers (using Structure of Arrays for performance)
    float4* d_positions_;     // x, y, z, mass
    float3* d_velocities_;    // vx, vy, vz
    float3* d_forces_;        // fx, fy, fz
    curandState* d_curand_states_;  // Random number generator states
    
    // Energy tracking
    float* d_kinetic_energy_;
    float* d_potential_energy_;
    float h_kinetic_energy_;
    float h_potential_energy_;
    float h_total_energy_;
    
    // CUDA streams for overlapping computation
    cudaStream_t streams_[2];
    
    LambdaCDMSimulationImpl(size_t num_particles, float box_size, const CosmologyParams& params)
        : params_(params), cosmology_(std::make_unique<CosmologyModel>(params)),
          num_particles_(num_particles), box_size_(box_size),
          time_step_(0.01f), scale_factor_(1.0), softening_(0.01f), current_step_(0),
          d_positions_(nullptr), d_velocities_(nullptr), d_forces_(nullptr),
          d_curand_states_(nullptr), d_kinetic_energy_(nullptr), d_potential_energy_(nullptr) {
        
        // Check CUDA device
        if (!kernels::check_cuda_device()) {
            throw std::runtime_error("No suitable CUDA device found");
        }
        
        // Print device info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using CUDA device: " << prop.name << std::endl;
        kernels::print_cuda_memory_info();
        
        // Allocate GPU memory using float4 for positions (better alignment)
        CUDA_CHECK(cudaMalloc(&d_positions_, num_particles_ * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_velocities_, num_particles_ * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&d_forces_, num_particles_ * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&d_curand_states_, num_particles_ * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_kinetic_energy_, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_potential_energy_, sizeof(float)));
        
        // Initialize CUDA streams
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
        
        // Allocate host memory for energy tracking
        h_kinetic_energy_ = 0.0f;
        h_potential_energy_ = 0.0f;
    }
    
    ~LambdaCDMSimulationImpl() {
        if (d_positions_) cudaFree(d_positions_);
        if (d_velocities_) cudaFree(d_velocities_);
        if (d_forces_) cudaFree(d_forces_);
        if (d_curand_states_) cudaFree(d_curand_states_);
        if (d_kinetic_energy_) cudaFree(d_kinetic_energy_);
        if (d_potential_energy_) cudaFree(d_potential_energy_);
        
        for (int i = 0; i < 2; i++) {
            cudaStreamDestroy(streams_[i]);
        }
    }
    
    void initialize_particles() {
        const int block_size = 256;
        const int grid_size = (num_particles_ + block_size - 1) / block_size;
        
        // Initialize random number generator states
        unsigned long seed = std::chrono::steady_clock::now().time_since_epoch().count();
        init_curand_states<<<grid_size, block_size>>>(d_curand_states_, num_particles_, seed);
        CUDA_CHECK_LAST_ERROR();
        
        // Generate initial conditions on GPU
        float vel_dispersion = 100.0f * sqrt(params_.omega_m);  // Scale with Omega_m
        generate_initial_conditions<<<grid_size, block_size>>>(
            d_positions_, d_velocities_, d_curand_states_,
            num_particles_, box_size_, vel_dispersion);
        CUDA_CHECK_LAST_ERROR();
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::cout << "Initialized " << num_particles_ << " particles" << std::endl;
    }
    
    void step(double dt) {
        // Use leapfrog integration: Kick-Drift-Kick scheme
        // First half kick
        kernels::launch_leapfrog_update(
            d_positions_, d_velocities_, d_forces_,
            num_particles_, dt * 0.5, box_size_, scale_factor_,
            true,  // update velocity
            streams_[0]);
        
        // Drift
        kernels::launch_leapfrog_update(
            d_positions_, d_velocities_, d_forces_,
            num_particles_, dt, box_size_, scale_factor_,
            false,  // update position
            streams_[1]);
        
        // Update scale factor
        update_scale_factor(dt);
        
        // Compute new forces
        compute_forces();
        
        // Second half kick
        kernels::launch_leapfrog_update(
            d_positions_, d_velocities_, d_forces_,
            num_particles_, dt * 0.5, box_size_, scale_factor_,
            true,  // update velocity
            streams_[0]);
        
        // Synchronize streams
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams_[1]));
        
        current_step_++;
    }
    
    void compute_forces() {
        kernels::launch_force_computation(
            d_positions_, d_forces_,
            num_particles_, box_size_, softening_,
            streams_[0]);
        
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
    }
    
    void compute_energy() {
        kernels::launch_energy_computation(
            d_positions_, d_velocities_,
            d_kinetic_energy_, d_potential_energy_,
            num_particles_, box_size_, softening_,
            streams_[0]);
        
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
        
        // Copy energy values to host
        CUDA_CHECK(cudaMemcpy(&h_kinetic_energy_, d_kinetic_energy_, 
                             sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_potential_energy_, d_potential_energy_, 
                             sizeof(float), cudaMemcpyDeviceToHost));
        
        h_total_energy_ = h_kinetic_energy_ + h_potential_energy_;
    }
    
    double hubble_function(double a) const {
        return cosmology_->hubble_parameter_a(a);
    }
    
    double growth_factor(double a) const {
        return cosmology_->growth_factor(a);
    }
    
    double growth_rate(double a) const {
        return cosmology_->growth_rate(a);
    }
    
    double power_spectrum(double k) const {
        // Power spectrum at current redshift
        double z = 1.0 / scale_factor_ - 1.0;
        return cosmology_->power_spectrum(k, z);
    }
    
    void update_scale_factor(double dt) {
        // Use proper Friedmann equation integration
        // da/dt = a * H(a)
        double H = cosmology_->hubble_parameter_a(scale_factor_);
        scale_factor_ += scale_factor_ * H * dt;
        
        // Convert dt from code units to physical units if needed
        // For N-body simulations, dt is typically in units of H0^-1
    }
};

// Public interface implementation
LambdaCDMSimulation::LambdaCDMSimulation(size_t num_particles, float box_size, 
                                       const CosmologyParams& params)
    : pImpl(std::make_unique<LambdaCDMSimulationImpl>(num_particles, box_size, params)) {
}

LambdaCDMSimulation::~LambdaCDMSimulation() = default;

LambdaCDMSimulation::LambdaCDMSimulation(LambdaCDMSimulation&&) noexcept = default;
LambdaCDMSimulation& LambdaCDMSimulation::operator=(LambdaCDMSimulation&&) noexcept = default;

void LambdaCDMSimulation::initialize_particles() {
    pImpl->initialize_particles();
}

void LambdaCDMSimulation::set_softening(float softening) {
    pImpl->softening_ = softening;
}

void LambdaCDMSimulation::step(double dt) {
    pImpl->step(dt);
}

void LambdaCDMSimulation::compute_forces() {
    pImpl->compute_forces();
}

void LambdaCDMSimulation::compute_energy() {
    pImpl->compute_energy();
}

void LambdaCDMSimulation::update_scale_factor(double dt) {
    pImpl->update_scale_factor(dt);
}

double LambdaCDMSimulation::hubble_function(double a) const {
    return pImpl->hubble_function(a);
}

double LambdaCDMSimulation::growth_factor(double a) const {
    return pImpl->growth_factor(a);
}

double LambdaCDMSimulation::get_scale_factor() const {
    return pImpl->scale_factor_;
}

double LambdaCDMSimulation::get_redshift() const {
    return 1.0 / pImpl->scale_factor_ - 1.0;
}

size_t LambdaCDMSimulation::get_num_particles() const {
    return pImpl->num_particles_;
}

float LambdaCDMSimulation::get_box_size() const {
    return pImpl->box_size_;
}

float LambdaCDMSimulation::get_kinetic_energy() const {
    return pImpl->h_kinetic_energy_;
}

float LambdaCDMSimulation::get_potential_energy() const {
    return pImpl->h_potential_energy_;
}

float LambdaCDMSimulation::get_total_energy() const {
    return pImpl->h_total_energy_;
}

size_t LambdaCDMSimulation::get_current_step() const {
    return pImpl->current_step_;
}

}