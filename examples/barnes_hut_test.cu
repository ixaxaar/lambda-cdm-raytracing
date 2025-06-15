#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "forces/barnes_hut_tree.hpp"
#include "physics/lambda_cdm_kernels.hpp"

using namespace forces;
using namespace std::chrono;

// Simple integration kernels for testing
__global__ void update_velocities_kernel(
    float3* velocities,
    const float3* forces,
    int num_particles,
    float dt) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    float3 vel = velocities[tid];
    float3 force = forces[tid];
    
    vel.x += force.x * dt;
    vel.y += force.y * dt;
    vel.z += force.z * dt;
    
    velocities[tid] = vel;
}

__global__ void update_positions_kernel(
    float4* positions,
    const float3* velocities,
    int num_particles,
    float dt,
    float box_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    float4 pos = positions[tid];
    float3 vel = velocities[tid];
    
    // Update position
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    // Apply periodic boundary conditions
    pos.x = pos.x - box_size * floorf(pos.x / box_size);
    pos.y = pos.y - box_size * floorf(pos.y / box_size);
    pos.z = pos.z - box_size * floorf(pos.z / box_size);
    
    positions[tid] = pos;
}

// Helper function to generate random particles
void generate_particles(std::vector<float4>& positions, std::vector<float3>& velocities,
                       size_t num_particles, float box_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0.0f, box_size);
    std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> mass_dist(0.8f, 1.2f);
    
    positions.resize(num_particles);
    velocities.resize(num_particles);
    
    for (size_t i = 0; i < num_particles; i++) {
        positions[i] = make_float4(pos_dist(gen), pos_dist(gen), pos_dist(gen), mass_dist(gen));
        velocities[i] = make_float3(vel_dist(gen), vel_dist(gen), vel_dist(gen));
    }
}

// Compute total energy
float compute_energy(const std::vector<float4>& positions, const std::vector<float3>& velocities,
                    const std::vector<float3>& forces, float box_size, float softening) {
    float kinetic = 0.0f;
    float potential = 0.0f;
    
    // Kinetic energy
    for (size_t i = 0; i < positions.size(); i++) {
        float v2 = velocities[i].x * velocities[i].x +
                  velocities[i].y * velocities[i].y +
                  velocities[i].z * velocities[i].z;
        kinetic += 0.5f * positions[i].w * v2;
    }
    
    // Potential energy (using forces to avoid double computation)
    // U = -0.5 * sum(m_i * Phi_i) where F_i = -m_i * grad(Phi_i)
    // Since we already have forces, we can't directly compute potential
    // For now, return only kinetic energy
    return kinetic;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        size_t num_particles = 50000;
        float theta = 0.5f;
        int num_steps = 10;
        
        if (argc > 1) {
            num_particles = std::stoul(argv[1]);
        }
        if (argc > 2) {
            theta = std::stof(argv[2]);
        }
        if (argc > 3) {
            num_steps = std::stoi(argv[3]);
        }
        
        std::cout << "\n=== Barnes-Hut Tree Algorithm Test ===" << std::endl;
        std::cout << "Number of particles: " << num_particles << std::endl;
        std::cout << "Opening angle (theta): " << theta << std::endl;
        std::cout << "Number of steps: " << num_steps << std::endl;
        
        // Set up simulation parameters
        float box_size = 100.0f;
        float softening = box_size / 1000.0f;
        float dt = 0.001f;
        
        // Generate particles
        std::vector<float4> h_positions;
        std::vector<float3> h_velocities;
        generate_particles(h_positions, h_velocities, num_particles, box_size);
        
        // Allocate device memory
        float4* d_positions;
        float3* d_velocities;
        float3* d_forces_tree;
        float3* d_forces_direct;
        
        cudaMalloc(&d_positions, num_particles * sizeof(float4));
        cudaMalloc(&d_velocities, num_particles * sizeof(float3));
        cudaMalloc(&d_forces_tree, num_particles * sizeof(float3));
        cudaMalloc(&d_forces_direct, num_particles * sizeof(float3));
        
        // Copy data to device
        cudaMemcpy(d_positions, h_positions.data(), num_particles * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_velocities, h_velocities.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
        
        // Create Barnes-Hut tree
        BarnesHutTree tree(num_particles, box_size, theta);
        
        std::cout << "\nPerforming accuracy comparison..." << std::endl;
        
        // Build tree
        tree.build_tree(d_positions);
        
        // Compute forces using tree
        auto tree_start = high_resolution_clock::now();
        tree.compute_forces(d_positions, d_forces_tree, softening);
        cudaDeviceSynchronize();
        auto tree_end = high_resolution_clock::now();
        
        // Compute forces using direct method (for comparison)
        auto direct_start = high_resolution_clock::now();
        int blocks = (num_particles + 255) / 256;
        physics::kernels::compute_forces_direct<<<blocks, 256>>>(
            d_positions, d_forces_direct, num_particles, box_size, softening * softening);
        cudaDeviceSynchronize();
        auto direct_end = high_resolution_clock::now();
        
        // Copy forces back to host for comparison
        std::vector<float3> h_forces_tree(num_particles);
        std::vector<float3> h_forces_direct(num_particles);
        cudaMemcpy(h_forces_tree.data(), d_forces_tree, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_forces_direct.data(), d_forces_direct, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);
        
        // Compute force error
        float max_error = 0.0f;
        float avg_error = 0.0f;
        for (size_t i = 0; i < num_particles; i++) {
            float fx_err = std::abs(h_forces_tree[i].x - h_forces_direct[i].x);
            float fy_err = std::abs(h_forces_tree[i].y - h_forces_direct[i].y);
            float fz_err = std::abs(h_forces_tree[i].z - h_forces_direct[i].z);
            
            float f_mag = std::sqrt(h_forces_direct[i].x * h_forces_direct[i].x +
                                   h_forces_direct[i].y * h_forces_direct[i].y +
                                   h_forces_direct[i].z * h_forces_direct[i].z);
            
            float rel_error = std::sqrt(fx_err*fx_err + fy_err*fy_err + fz_err*fz_err) / (f_mag + 1e-10f);
            max_error = std::max(max_error, rel_error);
            avg_error += rel_error;
        }
        avg_error /= num_particles;
        
        auto tree_duration = duration_cast<microseconds>(tree_end - tree_start);
        auto direct_duration = duration_cast<microseconds>(direct_end - direct_start);
        
        std::cout << "\nForce computation times:" << std::endl;
        std::cout << "  Direct O(N²): " << direct_duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Barnes-Hut:   " << tree_duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Speedup:      " << std::fixed << std::setprecision(2) 
                  << (float)direct_duration.count() / tree_duration.count() << "x" << std::endl;
        
        std::cout << "\nAccuracy comparison:" << std::endl;
        std::cout << "  Average relative error: " << std::scientific << avg_error << std::endl;
        std::cout << "  Maximum relative error: " << std::scientific << max_error << std::endl;
        
        // Run simulation steps
        std::cout << "\nRunning simulation steps..." << std::endl;
        
        auto sim_start = high_resolution_clock::now();
        
        for (int step = 0; step < num_steps; step++) {
            // Build tree
            tree.build_tree(d_positions);
            
            // Compute forces
            tree.compute_forces(d_positions, d_forces_tree, softening);
            
            // Update velocities and positions (leapfrog)
            blocks = (num_particles + 255) / 256;
            update_velocities_kernel<<<blocks, 256>>>(
                d_velocities, d_forces_tree, num_particles, dt * 0.5f);
            update_positions_kernel<<<blocks, 256>>>(
                d_positions, d_velocities, num_particles, dt, box_size);
            
            // Recompute forces at new positions
            tree.build_tree(d_positions);
            tree.compute_forces(d_positions, d_forces_tree, softening);
            
            // Final velocity update
            update_velocities_kernel<<<blocks, 256>>>(
                d_velocities, d_forces_tree, num_particles, dt * 0.5f);
            
            if ((step + 1) % 10 == 0) {
                std::cout << "  Step " << std::setw(3) << (step + 1) << " completed" << std::endl;
            }
        }
        
        cudaDeviceSynchronize();
        auto sim_end = high_resolution_clock::now();
        
        auto sim_duration = duration_cast<milliseconds>(sim_end - sim_start);
        
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "Total simulation time: " << sim_duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Time per step: " << sim_duration.count() / (double)num_steps << " ms" << std::endl;
        std::cout << "Tree depth: ~" << tree.get_tree_depth() << std::endl;
        std::cout << "Tree nodes: " << tree.get_num_nodes() << std::endl;
        
        // Performance metrics
        double particles_per_second = (double)num_particles * num_steps * 2 / (sim_duration.count() / 1000.0);
        std::cout << "\nPerformance: " << std::scientific << particles_per_second 
                  << " particle-updates/second" << std::endl;
        
        std::cout << "\nExpected complexity:" << std::endl;
        std::cout << "  Direct method: O(N²) = " << (double)num_particles * num_particles << " operations" << std::endl;
        std::cout << "  Barnes-Hut:    O(N log N) = " << (double)num_particles * std::log2(num_particles) 
                  << " operations" << std::endl;
        
        // Cleanup
        cudaFree(d_positions);
        cudaFree(d_velocities);
        cudaFree(d_forces_tree);
        cudaFree(d_forces_direct);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}