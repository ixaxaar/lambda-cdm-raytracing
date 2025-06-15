#pragma once

#include <cuda_runtime.h>
#include "core/math_types.hpp"
#include <vector>
#include <memory>

namespace forces {

// Morton code utilities for spatial indexing
__device__ __host__ inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expand_bits((uint32_t)x);
    uint32_t yy = expand_bits((uint32_t)y);
    uint32_t zz = expand_bits((uint32_t)z);
    return xx * 4 + yy * 2 + zz;
}

// Tree node structure
struct TreeNode {
    float4 center_of_mass;  // x, y, z, total_mass
    float3 bounds_min;      // Bounding box minimum
    float3 bounds_max;      // Bounding box maximum
    int child_indices[8];   // Octree children (-1 if empty)
    int particle_index;     // Particle index (if leaf node, -1 otherwise)
    int num_particles;      // Number of particles in subtree
    float size;             // Node size (max dimension)
    
    __device__ __host__ TreeNode() {
        center_of_mass = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bounds_min = make_float3(0.0f, 0.0f, 0.0f);
        bounds_max = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 8; i++) {
            child_indices[i] = -1;
        }
        particle_index = -1;
        num_particles = 0;
        size = 0.0f;
    }
};

// Barnes-Hut tree builder
class BarnesHutTree {
private:
    // Device memory
    TreeNode* d_nodes_;
    int* d_node_count_;
    uint32_t* d_morton_codes_;
    uint32_t* d_sorted_morton_codes_;
    int* d_particle_indices_;
    int* d_sorted_indices_;
    
    // Tree parameters
    size_t max_nodes_;
    size_t num_particles_;
    float box_size_;
    float theta_;  // Opening angle
    
    // CUDA streams
    cudaStream_t build_stream_;
    cudaStream_t compute_stream_;
    
public:
    BarnesHutTree(size_t num_particles, float box_size, float theta = 0.5f);
    ~BarnesHutTree();
    
    // Build tree from particle positions
    void build_tree(const float4* d_positions, cudaStream_t stream = 0);
    
    // Compute forces using tree
    void compute_forces(const float4* d_positions, float3* d_forces,
                       float softening, cudaStream_t stream = 0);
    
    // Get tree statistics
    int get_num_nodes() const;
    float get_tree_depth() const;
    
private:
    // Tree construction steps
    void compute_morton_codes(const float4* d_positions);
    void sort_particles();
    void build_tree_structure();
    void compute_mass_centers();
};

// CUDA kernels for tree operations
namespace kernels {

// Compute Morton codes for particles
__global__ void compute_morton_codes_kernel(
    const float4* positions,
    uint32_t* morton_codes,
    int num_particles,
    float box_size);

// Build tree structure from sorted particles
__global__ void build_tree_kernel(
    const uint32_t* sorted_morton_codes,
    const int* sorted_indices,
    TreeNode* nodes,
    int* node_count,
    int num_particles);

// Compute center of mass for tree nodes
__global__ void compute_mass_centers_kernel(
    const float4* positions,
    const int* sorted_indices,
    TreeNode* nodes,
    int num_nodes);

// Tree traversal for force computation
__global__ void tree_force_kernel(
    const float4* positions,
    const TreeNode* nodes,
    float3* forces,
    int num_particles,
    float theta,
    float softening2);

// Optimized tree walk with shared memory
__global__ void tree_force_shared_kernel(
    const float4* positions,
    const TreeNode* nodes,
    float3* forces,
    int num_particles,
    float theta,
    float softening2);

} // namespace kernels
} // namespace forces