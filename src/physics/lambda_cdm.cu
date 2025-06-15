#include "physics/lambda_cdm.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>

namespace physics {

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__global__ void compute_gravitational_forces(
    const float3* positions, const float* masses, float3* forces,
    int num_particles, float softening, float box_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = positions[i];
    
    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;
        
        float3 pos_j = positions[j];
        float3 dr = pos_j - pos_i;
        
        // Periodic boundary conditions
        if (dr.x > box_size * 0.5f) dr.x -= box_size;
        if (dr.x < -box_size * 0.5f) dr.x += box_size;
        if (dr.y > box_size * 0.5f) dr.y -= box_size;
        if (dr.y < -box_size * 0.5f) dr.y += box_size;
        if (dr.z > box_size * 0.5f) dr.z -= box_size;
        if (dr.z < -box_size * 0.5f) dr.z += box_size;
        
        float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + softening * softening;
        float r = sqrtf(r2);
        float r3 = r2 * r;
        
        float f_mag = masses[j] / r3;
        force = force + dr * f_mag;
    }
    
    forces[i] = force;
}

__global__ void leapfrog_kick(float3* velocities, const float3* forces, 
                             int num_particles, float dt, double scale_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    // Convert to comoving coordinates
    float a_factor = 1.0f / (scale_factor * scale_factor);
    velocities[i] = velocities[i] + forces[i] * (dt * a_factor);
}

__global__ void leapfrog_drift(float3* positions, const float3* velocities,
                              int num_particles, float dt, float box_size,
                              double scale_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float3 pos = positions[i] + velocities[i] * dt;
    
    // Periodic boundary conditions
    if (pos.x >= box_size) pos.x -= box_size;
    if (pos.x < 0.0f) pos.x += box_size;
    if (pos.y >= box_size) pos.y -= box_size;
    if (pos.y < 0.0f) pos.y += box_size;
    if (pos.z >= box_size) pos.z -= box_size;
    if (pos.z < 0.0f) pos.z += box_size;
    
    positions[i] = pos;
}

LambdaCDMSimulation::LambdaCDMSimulation(size_t num_particles, float box_size, 
                                       const CosmologyParams& params)
    : params_(params), num_particles_(num_particles), box_size_(box_size),
      time_step_(0.01f), scale_factor_(1.0) {
    
    particles_.resize(num_particles_);
    forces_ = std::make_unique<float[]>(num_particles_ * 3);
    
    // Allocate GPU memory
    cudaMalloc(&d_positions_, num_particles_ * 3 * sizeof(float));
    cudaMalloc(&d_velocities_, num_particles_ * 3 * sizeof(float));
    cudaMalloc(&d_masses_, num_particles_ * sizeof(float));
    cudaMalloc(&d_forces_, num_particles_ * 3 * sizeof(float));
}

LambdaCDMSimulation::~LambdaCDMSimulation() {
    cudaFree(d_positions_);
    cudaFree(d_velocities_);
    cudaFree(d_masses_);
    cudaFree(d_forces_);
}

void LambdaCDMSimulation::initialize_particles() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0.0f, box_size_);
    std::normal_distribution<float> vel_dist(0.0f, 100.0f);
    
    for (size_t i = 0; i < num_particles_; ++i) {
        particles_[i].position = make_float3(
            pos_dist(gen), pos_dist(gen), pos_dist(gen));
        particles_[i].velocity = make_float3(
            vel_dist(gen), vel_dist(gen), vel_dist(gen));
        particles_[i].mass = 1.0f;
        particles_[i].id = i;
    }
    
    // Copy to GPU
    cudaMemcpy(d_positions_, &particles_[0].position, 
               num_particles_ * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities_, &particles_[0].velocity,
               num_particles_ * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses_, &particles_[0].mass,
               num_particles_ * sizeof(float), cudaMemcpyHostToDevice);
}

void LambdaCDMSimulation::step(double dt) {
    compute_forces();
    update_positions(dt);
    update_scale_factor(dt);
}

void LambdaCDMSimulation::compute_forces() {
    const int block_size = 256;
    const int grid_size = (num_particles_ + block_size - 1) / block_size;
    
    compute_gravitational_forces<<<grid_size, block_size>>>(
        (float3*)d_positions_, d_masses_, (float3*)d_forces_,
        num_particles_, 0.01f, box_size_);
    
    cudaDeviceSynchronize();
}

void LambdaCDMSimulation::update_positions(double dt) {
    const int block_size = 256;
    const int grid_size = (num_particles_ + block_size - 1) / block_size;
    
    // Leapfrog integration
    leapfrog_kick<<<grid_size, block_size>>>(
        (float3*)d_velocities_, (float3*)d_forces_, 
        num_particles_, dt, scale_factor_);
    
    leapfrog_drift<<<grid_size, block_size>>>(
        (float3*)d_positions_, (float3*)d_velocities_,
        num_particles_, dt, box_size_, scale_factor_);
    
    cudaDeviceSynchronize();
    
    // Copy back to host
    cudaMemcpy(&particles_[0].position, d_positions_,
               num_particles_ * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&particles_[0].velocity, d_velocities_,
               num_particles_ * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

double LambdaCDMSimulation::hubble_function(double a) const {
    double omega_m_a3 = params_.omega_m / (a * a * a);
    double omega_lambda = params_.omega_lambda;
    return params_.h * sqrt(omega_m_a3 + omega_lambda);
}

double LambdaCDMSimulation::growth_factor(double a) const {
    // Approximate growth factor for Lambda-CDM
    double omega_m_a3 = params_.omega_m / (a * a * a);
    double omega_lambda = params_.omega_lambda;
    double omega_total = omega_m_a3 + omega_lambda;
    
    double omega_m_z = omega_m_a3 / omega_total;
    double omega_lambda_z = omega_lambda / omega_total;
    
    double f1 = pow(omega_m_z, 4.0/7.0);
    double f2 = omega_lambda_z / (1.0 + omega_m_z/2.0);
    
    return a * f1 - f2;
}

void LambdaCDMSimulation::update_scale_factor(double dt) {
    double H = hubble_function(scale_factor_);
    scale_factor_ += scale_factor_ * H * dt;
}

}