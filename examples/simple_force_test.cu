#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "physics/lambda_cdm_kernels.hpp"

using namespace std::chrono;

int main() {
    const int num_particles = 1000;
    const float box_size = 100.0f;
    const float softening = 0.1f;
    
    std::cout << "Testing direct force computation with " << num_particles << " particles\n";
    
    // Allocate device memory
    float4* d_positions;
    float3* d_forces;
    cudaMalloc(&d_positions, num_particles * sizeof(float4));
    cudaMalloc(&d_forces, num_particles * sizeof(float3));
    
    // Initialize particles on device
    cudaMemset(d_positions, 0, num_particles * sizeof(float4));
    cudaMemset(d_forces, 0, num_particles * sizeof(float3));
    
    // Launch kernel
    int blocks = (num_particles + 255) / 256;
    
    auto start = high_resolution_clock::now();
    physics::kernels::compute_forces_direct<<<blocks, 256>>>(
        d_positions, d_forces, num_particles, box_size, softening * softening);
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Force computation took: " << duration.count() << " ms\n";
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_forces);
    
    std::cout << "Test completed successfully!\n";
    return 0;
}