#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "core/math_types.hpp"

namespace physics {
namespace kernels {

// CUDA kernel functions (for direct access)
__global__ void compute_forces_direct(
    const float4* __restrict__ positions,
    float3* __restrict__ forces,
    const int num_particles,
    const float box_size,
    const float softening2);

__global__ void update_velocities(
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const int num_particles,
    const float dt);

__global__ void update_positions(
    float4* __restrict__ positions,
    const float3* __restrict__ velocities,
    const int num_particles,
    const float dt,
    const float box_size);

// CUDA kernel launch functions
void launch_force_computation(
    const float4* d_positions,  // x, y, z, mass
    float3* d_forces,
    int num_particles,
    float box_size,
    float softening,
    cudaStream_t stream = 0);

void launch_leapfrog_update(
    float4* d_positions,
    float3* d_velocities,
    const float3* d_forces,
    int num_particles,
    float dt,
    float box_size,
    double scale_factor,
    bool update_velocity_first,
    cudaStream_t stream = 0);

void launch_energy_computation(
    const float4* d_positions,
    const float3* d_velocities,
    float* d_kinetic_energy,
    float* d_potential_energy,
    int num_particles,
    float box_size,
    float softening,
    cudaStream_t stream = 0);

// Helper class for CUDA stream management
class CudaStreamPool;

// Error checking utilities
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + \
                                cudaGetErrorString(error)); \
    } \
} while(0)

#define CUDA_CHECK_LAST_ERROR() do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + \
                                cudaGetErrorString(error)); \
    } \
} while(0)

// Device capability checking
inline bool check_cuda_device() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Require at least compute capability 6.0 (Pascal)
    return prop.major >= 6;
}

// Memory info utilities
inline void print_cuda_memory_info() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    std::printf("GPU Memory: %.2f GB free, %.2f GB total\n", 
           free_bytes / (1024.0 * 1024.0 * 1024.0),
           total_bytes / (1024.0 * 1024.0 * 1024.0));
}

} // namespace kernels
} // namespace physics