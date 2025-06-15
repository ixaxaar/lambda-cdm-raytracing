#include "forces/barnes_hut_tree.hpp"
#include "physics/lambda_cdm_kernels.hpp"
#include <cuda.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace forces {

// Helper to check for CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while(0)

// Constants for tree building
constexpr int WARP_SIZE = 32;
constexpr int MAX_PARTICLES_PER_LEAF = 1;
constexpr int TREE_BUILD_THREADS = 256;

namespace kernels {

// Compute Morton codes for particles
__global__ void compute_morton_codes_kernel(
    const float4* positions,
    uint32_t* morton_codes,
    int num_particles,
    float box_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    float4 pos = positions[tid];
    
    // Normalize positions to [0, 1]
    float x = pos.x / box_size;
    float y = pos.y / box_size;
    float z = pos.z / box_size;
    
    // Wrap periodic boundaries
    x = x - floorf(x);
    y = y - floorf(y);
    z = z - floorf(z);
    
    // Compute Morton code
    morton_codes[tid] = morton3D(x, y, z);
}

// Find common prefix length between two Morton codes
__device__ inline int common_prefix_length(uint32_t a, uint32_t b) {
    if (a == b) return 32; // All bits match
    return __clz(a ^ b);
}

// Build internal nodes of the tree
__global__ void build_internal_nodes_kernel(
    const uint32_t* sorted_morton_codes,
    TreeNode* nodes,
    int* node_count,
    int num_particles) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles - 1) return;
    
    // Determine direction of the range (+1 or -1)
    int d = 1;
    if (tid > 0) {
        int delta_plus = common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid + 1]);
        int delta_minus = common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid - 1]);
        d = (delta_plus > delta_minus) ? 1 : -1;
    }
    
    // Compute upper bound for the length of the range
    int delta_min = common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid - d]);
    int l_max = 2;
    
    while (tid + l_max * d >= 0 && tid + l_max * d < num_particles &&
           common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid + l_max * d]) > delta_min) {
        l_max = l_max * 2;
    }
    
    // Find the other end using binary search
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (tid + (l + t) * d >= 0 && tid + (l + t) * d < num_particles &&
            common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid + (l + t) * d]) > delta_min) {
            l = l + t;
        }
    }
    
    int j = tid + l * d;
    
    // Find the split position using binary search
    int delta_node = common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[j]);
    int s = 0;
    for (int t = (l + 1) / 2; t >= 1; t = (t + 1) / 2) {
        if (tid + (s + t) * d < num_particles &&
            common_prefix_length(sorted_morton_codes[tid], sorted_morton_codes[tid + (s + t) * d]) > delta_node) {
            s = s + t;
        }
    }
    
    int gamma = tid + s * d + (d > 0 ? 1 : 0);
    
    // Allocate internal node
    int node_idx = num_particles + tid;
    TreeNode& node = nodes[node_idx];
    
    // Set children
    if (min(tid, j) == gamma) {
        node.child_indices[0] = gamma;  // Leaf
    } else {
        node.child_indices[0] = num_particles + gamma;  // Internal
    }
    
    if (max(tid, j) == gamma + 1) {
        node.child_indices[1] = gamma + 1;  // Leaf
    } else {
        node.child_indices[1] = num_particles + gamma + 1;  // Internal
    }
    
    // Initialize other children as empty
    for (int i = 2; i < 8; i++) {
        node.child_indices[i] = -1;
    }
}

// Initialize leaf nodes
__global__ void init_leaf_nodes_kernel(
    const float4* positions,
    const int* sorted_indices,
    TreeNode* nodes,
    int num_particles) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    int particle_idx = sorted_indices[tid];
    float4 pos = positions[particle_idx];
    
    TreeNode& node = nodes[tid];
    node.center_of_mass = pos;
    node.bounds_min = make_float3(pos.x, pos.y, pos.z);
    node.bounds_max = make_float3(pos.x, pos.y, pos.z);
    node.particle_index = particle_idx;
    node.num_particles = 1;
    node.size = 0.0f;
    
    // Leaf nodes have no children
    for (int i = 0; i < 8; i++) {
        node.child_indices[i] = -1;
    }
}

// Bottom-up pass to compute mass centers and bounds
__global__ void compute_mass_centers_kernel(
    TreeNode* nodes,
    int* flags,
    int num_internal_nodes) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_internal_nodes) return;
    
    // Process internal nodes: they are indexed starting from num_particles
    int internal_idx = num_internal_nodes + tid;
    TreeNode& node = nodes[internal_idx];
    
    // Compute center of mass and bounds from children
    float total_mass = 0.0f;
    float3 weighted_pos = make_float3(0.0f, 0.0f, 0.0f);
    float3 bounds_min = make_float3(1e30f, 1e30f, 1e30f);
    float3 bounds_max = make_float3(-1e30f, -1e30f, -1e30f);
    int total_particles = 0;
    
    // For simplified tree, only use first two children
    for (int i = 0; i < 2; i++) {
        int child_idx = node.child_indices[i];
        if (child_idx == -1) continue;
        
        TreeNode& child = nodes[child_idx];
        
        // Skip if child not initialized
        if (child.num_particles == 0) continue;
        
        float mass = child.center_of_mass.w;
        
        weighted_pos.x += child.center_of_mass.x * mass;
        weighted_pos.y += child.center_of_mass.y * mass;
        weighted_pos.z += child.center_of_mass.z * mass;
        total_mass += mass;
        
        bounds_min.x = fminf(bounds_min.x, child.bounds_min.x);
        bounds_min.y = fminf(bounds_min.y, child.bounds_min.y);
        bounds_min.z = fminf(bounds_min.z, child.bounds_min.z);
        
        bounds_max.x = fmaxf(bounds_max.x, child.bounds_max.x);
        bounds_max.y = fmaxf(bounds_max.y, child.bounds_max.y);
        bounds_max.z = fmaxf(bounds_max.z, child.bounds_max.z);
        
        total_particles += child.num_particles;
    }
    
    // Update node properties
    if (total_mass > 0.0f) {
        node.center_of_mass.x = weighted_pos.x / total_mass;
        node.center_of_mass.y = weighted_pos.y / total_mass;
        node.center_of_mass.z = weighted_pos.z / total_mass;
        node.center_of_mass.w = total_mass;
        
        node.bounds_min = bounds_min;
        node.bounds_max = bounds_max;
        node.num_particles = total_particles;
        
        float dx = bounds_max.x - bounds_min.x;
        float dy = bounds_max.y - bounds_min.y;
        float dz = bounds_max.z - bounds_min.z;
        node.size = fmaxf(fmaxf(dx, dy), dz);
    }
}

// Tree traversal for force computation
__device__ void traverse_tree(
    float4 particle_pos,
    const TreeNode* nodes,
    int node_idx,
    float3& force,
    float theta2,
    float softening2,
    float box_size) {
    
    const TreeNode& node = nodes[node_idx];
    
    // Skip empty nodes
    if (node.num_particles == 0) return;
    
    // Calculate distance to center of mass
    float dx = node.center_of_mass.x - particle_pos.x;
    float dy = node.center_of_mass.y - particle_pos.y;
    float dz = node.center_of_mass.z - particle_pos.z;
    
    // Apply periodic boundary conditions
    dx = dx - box_size * roundf(dx / box_size);
    dy = dy - box_size * roundf(dy / box_size);
    dz = dz - box_size * roundf(dz / box_size);
    
    float r2 = dx*dx + dy*dy + dz*dz + softening2;
    
    // Check if we can use this node
    if (node.particle_index != -1 || node.size * node.size < theta2 * r2) {
        // Use this node's center of mass
        float r = sqrtf(r2);
        float r3 = r2 * r;
        float f = node.center_of_mass.w / r3;
        
        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    } else {
        // Recurse to children
        for (int i = 0; i < 8; i++) {
            int child_idx = node.child_indices[i];
            if (child_idx != -1) {
                traverse_tree(particle_pos, nodes, child_idx, force, theta2, softening2, box_size);
            }
        }
    }
}

// Tree force kernel
__global__ void tree_force_kernel(
    const float4* positions,
    const TreeNode* nodes,
    float3* forces,
    int num_particles,
    float theta,
    float softening2,
    float box_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    float4 pos = positions[tid];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float theta2 = theta * theta;
    
    // Start traversal from root
    int root_idx = 2 * num_particles - 2;
    traverse_tree(pos, nodes, root_idx, force, theta2, softening2, box_size);
    
    forces[tid] = force;
}

// Optimized tree walk with shared memory (for small trees)
__global__ void tree_force_shared_kernel(
    const float4* positions,
    const TreeNode* nodes,
    float3* forces,
    int num_particles,
    float theta,
    float softening2,
    float box_size) {
    
    extern __shared__ TreeNode shared_nodes[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    
    // Load frequently accessed nodes into shared memory
    const int nodes_per_block = blockDim.x;
    for (int i = lid; i < nodes_per_block && i < num_particles * 2 - 1; i += blockDim.x) {
        shared_nodes[i] = nodes[i];
    }
    __syncthreads();
    
    if (tid >= num_particles) return;
    
    float4 pos = positions[tid];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float theta2 = theta * theta;
    
    // Use shared memory for first levels, global memory for deeper levels
    int root_idx = 2 * num_particles - 2;
    if (root_idx < nodes_per_block) {
        traverse_tree(pos, shared_nodes, root_idx, force, theta2, softening2, box_size);
    } else {
        traverse_tree(pos, nodes, root_idx, force, theta2, softening2, box_size);
    }
    
    forces[tid] = force;
}

} // namespace kernels

// BarnesHutTree implementation
BarnesHutTree::BarnesHutTree(size_t num_particles, float box_size, float theta)
    : num_particles_(num_particles)
    , box_size_(box_size)
    , theta_(theta)
    , max_nodes_(2 * num_particles - 1)  // Perfect binary tree
{
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_nodes_, max_nodes_ * sizeof(TreeNode)));
    CUDA_CHECK(cudaMalloc(&d_node_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_morton_codes_, num_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_morton_codes_, num_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_particle_indices_, num_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_indices_, num_particles * sizeof(int)));
    
    // Initialize particle indices
    thrust::sequence(thrust::device, d_particle_indices_, d_particle_indices_ + num_particles);
    
    // Create streams
    CUDA_CHECK(cudaStreamCreate(&build_stream_));
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
}

BarnesHutTree::~BarnesHutTree() {
    cudaFree(d_nodes_);
    cudaFree(d_node_count_);
    cudaFree(d_morton_codes_);
    cudaFree(d_sorted_morton_codes_);
    cudaFree(d_particle_indices_);
    cudaFree(d_sorted_indices_);
    
    cudaStreamDestroy(build_stream_);
    cudaStreamDestroy(compute_stream_);
}

void BarnesHutTree::compute_morton_codes(const float4* d_positions) {
    int blocks = (num_particles_ + TREE_BUILD_THREADS - 1) / TREE_BUILD_THREADS;
    kernels::compute_morton_codes_kernel<<<blocks, TREE_BUILD_THREADS, 0, build_stream_>>>(
        d_positions, d_morton_codes_, num_particles_, box_size_);
}

void BarnesHutTree::sort_particles() {
    // Copy morton codes and indices for sorting
    CUDA_CHECK(cudaMemcpyAsync(d_sorted_morton_codes_, d_morton_codes_, 
                               num_particles_ * sizeof(uint32_t), 
                               cudaMemcpyDeviceToDevice, build_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_sorted_indices_, d_particle_indices_, 
                               num_particles_ * sizeof(int), 
                               cudaMemcpyDeviceToDevice, build_stream_));
    
    // Sort by Morton code
    thrust::sort_by_key(thrust::cuda::par.on(build_stream_),
                        d_sorted_morton_codes_, 
                        d_sorted_morton_codes_ + num_particles_,
                        d_sorted_indices_);
}

void BarnesHutTree::build_tree_structure() {
    // Build internal nodes
    int blocks = (num_particles_ - 1 + TREE_BUILD_THREADS - 1) / TREE_BUILD_THREADS;
    kernels::build_internal_nodes_kernel<<<blocks, TREE_BUILD_THREADS, 0, build_stream_>>>(
        d_sorted_morton_codes_, d_nodes_, d_node_count_, num_particles_);
}

void BarnesHutTree::compute_mass_centers() {
    // Allocate flags for synchronization
    int* d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, max_nodes_ * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_flags, 0, max_nodes_ * sizeof(int), build_stream_));
    
    // Process internal nodes in multiple passes for proper bottom-up
    int num_internal = num_particles_ - 1;
    
    // Process in log(N) passes for proper bottom-up traversal
    int max_levels = (int)ceil(log2f(num_particles_));
    for (int level = 0; level < max_levels; level++) {
        int blocks = (num_internal + TREE_BUILD_THREADS - 1) / TREE_BUILD_THREADS;
        
        kernels::compute_mass_centers_kernel<<<blocks, TREE_BUILD_THREADS, 0, build_stream_>>>(
            d_nodes_, d_flags, num_internal);
        
        CUDA_CHECK(cudaStreamSynchronize(build_stream_));
    }
    
    CUDA_CHECK(cudaFree(d_flags));
}

void BarnesHutTree::build_tree(const float4* d_positions, cudaStream_t stream) {
    cudaStream_t use_stream = (stream == 0) ? build_stream_ : stream;
    
    // 1. Compute Morton codes
    compute_morton_codes(d_positions);
    
    // 2. Sort particles by Morton code
    sort_particles();
    
    // 3. Initialize leaf nodes
    int blocks = (num_particles_ + TREE_BUILD_THREADS - 1) / TREE_BUILD_THREADS;
    kernels::init_leaf_nodes_kernel<<<blocks, TREE_BUILD_THREADS, 0, use_stream>>>(
        d_positions, d_sorted_indices_, d_nodes_, num_particles_);
    
    // 4. Build tree structure
    build_tree_structure();
    
    // 5. Compute mass centers bottom-up
    compute_mass_centers();
    
    // Synchronize build stream
    CUDA_CHECK(cudaStreamSynchronize(use_stream));
}

void BarnesHutTree::compute_forces(const float4* d_positions, float3* d_forces,
                                   float softening, cudaStream_t stream) {
    cudaStream_t use_stream = (stream == 0) ? compute_stream_ : stream;
    
    int blocks = (num_particles_ + TREE_BUILD_THREADS - 1) / TREE_BUILD_THREADS;
    
    // Use shared memory optimization for smaller trees
    if (num_particles_ < 10000) {
        size_t shared_size = min((size_t)(2 * num_particles_ - 1), (size_t)1024) * sizeof(TreeNode);
        kernels::tree_force_shared_kernel<<<blocks, TREE_BUILD_THREADS, shared_size, use_stream>>>(
            d_positions, d_nodes_, d_forces, num_particles_, theta_, softening * softening, box_size_);
    } else {
        kernels::tree_force_kernel<<<blocks, TREE_BUILD_THREADS, 0, use_stream>>>(
            d_positions, d_nodes_, d_forces, num_particles_, theta_, softening * softening, box_size_);
    }
}

int BarnesHutTree::get_num_nodes() const {
    return max_nodes_;
}

float BarnesHutTree::get_tree_depth() const {
    // For a balanced tree with n particles, depth is approximately log2(n)
    return log2f(num_particles_);
}

} // namespace forces