#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "physics/lambda_cdm.hpp"

namespace physics {
namespace kernels {

// Direct O(N^2) force computation for comparison
__global__ void compute_forces_direct(
    const float4* __restrict__ positions,
    float3* __restrict__ forces,
    const int num_particles,
    const float box_size,
    const float softening2) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos_i = positions[i];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    
    // Compute force from all other particles
    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;
        
        float4 pos_j = positions[j];
        
        // Compute distance with periodic boundary conditions
        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        
        // Apply minimum image convention
        dx = dx - box_size * roundf(dx / box_size);
        dy = dy - box_size * roundf(dy / box_size);
        dz = dz - box_size * roundf(dz / box_size);
        
        float r2 = dx*dx + dy*dy + dz*dz + softening2;
        float r = sqrtf(r2);
        float r3 = r2 * r;
        
        // Gravitational force (G=1)
        float f = pos_j.w / r3;
        
        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    }
    
    forces[i] = force;
}

// Simple velocity update
__global__ void update_velocities(
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const int num_particles,
    const float dt) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float3 vel = velocities[i];
    float3 force = forces[i];
    
    vel.x += force.x * dt;
    vel.y += force.y * dt;
    vel.z += force.z * dt;
    
    velocities[i] = vel;
}

// Simple position update
__global__ void update_positions(
    float4* __restrict__ positions,
    const float3* __restrict__ velocities,
    const int num_particles,
    const float dt,
    const float box_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos = positions[i];
    float3 vel = velocities[i];
    
    // Update position
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    // Apply periodic boundary conditions
    pos.x = pos.x - box_size * floorf(pos.x / box_size);
    pos.y = pos.y - box_size * floorf(pos.y / box_size);
    pos.z = pos.z - box_size * floorf(pos.z / box_size);
    
    positions[i] = pos;
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Constants for optimized force computation
__constant__ float G_CONSTANT = 1.0f;  // In simulation units
__constant__ float THETA = 0.5f;       // Barnes-Hut opening angle
__constant__ float EPS2 = 0.0001f;     // Softening squared

// Shared memory tile size
constexpr int TILE_SIZE = 256;

// Periodic boundary conditions with minimum image convention
__device__ inline float3 minimum_image(float3 dr, float box_size) {
    const float half_box = box_size * 0.5f;
    
    if (dr.x > half_box) dr.x -= box_size;
    else if (dr.x < -half_box) dr.x += box_size;
    
    if (dr.y > half_box) dr.y -= box_size;
    else if (dr.y < -half_box) dr.y += box_size;
    
    if (dr.z > half_box) dr.z -= box_size;
    else if (dr.z < -half_box) dr.z += box_size;
    
    return dr;
}

// Optimized N-body force computation using tiling
__global__ void compute_forces_tiled(
    const float4* __restrict__ positions,  // x,y,z,mass
    float3* __restrict__ forces,
    const int num_particles,
    const float box_size,
    const float softening2) {
    
    // Shared memory for tile of particles
    __shared__ float4 sh_positions[TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    float3 my_pos;
    float my_mass = 0.0f;
    float3 acc;
    acc.x = acc.y = acc.z = 0.0f;
    
    // Load my particle
    if (gid < num_particles) {
        float4 pos_mass = positions[gid];
        my_pos.x = pos_mass.x;
        my_pos.y = pos_mass.y;
        my_pos.z = pos_mass.z;
        my_mass = pos_mass.w;
    }
    
    // Loop over all tiles
    const int num_tiles = (num_particles + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Collaboratively load tile into shared memory
        const int idx = tile * TILE_SIZE + tid;
        if (idx < num_particles) {
            sh_positions[tid] = positions[idx];
        } else {
            sh_positions[tid].x = sh_positions[tid].y = sh_positions[tid].z = sh_positions[tid].w = 0.0f;
        }
        __syncthreads();
        
        // Compute forces with particles in this tile
        if (gid < num_particles) {
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                const int jdx = tile * TILE_SIZE + j;
                if (jdx < num_particles && jdx != gid) {
                    float3 other_pos;
                    other_pos.x = sh_positions[j].x;
                    other_pos.y = sh_positions[j].y;
                    other_pos.z = sh_positions[j].z;
                    float other_mass = sh_positions[j].w;
                    
                    float3 dr;
                    dr.x = other_pos.x - my_pos.x;
                    dr.y = other_pos.y - my_pos.y;
                    dr.z = other_pos.z - my_pos.z;
                    dr = minimum_image(dr, box_size);
                    
                    const float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + softening2;
                    const float r_inv = rsqrtf(r2);
                    const float r3_inv = r_inv * r_inv * r_inv;
                    
                    const float f_over_r = G_CONSTANT * other_mass * r3_inv;
                    acc.x += dr.x * f_over_r;
                    acc.y += dr.y * f_over_r;
                    acc.z += dr.z * f_over_r;
                }
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (gid < num_particles) {
        forces[gid].x = acc.x * my_mass;  // F = ma, so we store acceleration * mass
        forces[gid].y = acc.y * my_mass;
        forces[gid].z = acc.z * my_mass;
    }
}

// All-pairs force computation with warp-level optimizations
__global__ void compute_forces_all_pairs(
    const float4* __restrict__ positions,
    float3* __restrict__ forces,
    const int num_particles,
    const float box_size,
    const float softening2) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (gid >= num_particles) return;
    
    const float4 my_pos_mass = positions[gid];
    float3 my_pos;
    my_pos.x = my_pos_mass.x;
    my_pos.y = my_pos_mass.y;
    my_pos.z = my_pos_mass.z;
    
    float3 acc;
    acc.x = acc.y = acc.z = 0.0f;
    
    // Process particles in warps for better memory coalescing
    for (int j = lane_id; j < num_particles; j += 32) {
        if (j != gid) {
            const float4 other_pos_mass = positions[j];
            float3 other_pos;
            other_pos.x = other_pos_mass.x;
            other_pos.y = other_pos_mass.y;
            other_pos.z = other_pos_mass.z;
            const float other_mass = other_pos_mass.w;
            
            float3 dr;
            dr.x = other_pos.x - my_pos.x;
            dr.y = other_pos.y - my_pos.y;
            dr.z = other_pos.z - my_pos.z;
            dr = minimum_image(dr, box_size);
            
            const float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + softening2;
            const float r_inv = rsqrtf(r2);
            const float r3_inv = r_inv * r_inv * r_inv;
            
            const float f_over_r = G_CONSTANT * other_mass * r3_inv;
            acc.x += dr.x * f_over_r;
            acc.y += dr.y * f_over_r;
            acc.z += dr.z * f_over_r;
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc.x += __shfl_down_sync(0xffffffff, acc.x, offset);
        acc.y += __shfl_down_sync(0xffffffff, acc.y, offset);
        acc.z += __shfl_down_sync(0xffffffff, acc.z, offset);
    }
    
    // First thread in warp writes result
    if (lane_id == 0) {
        forces[gid].x = acc.x * my_pos_mass.w;
        forces[gid].y = acc.y * my_pos_mass.w;
        forces[gid].z = acc.z * my_pos_mass.w;
    }
}

// Leapfrog integration with better memory access patterns
__global__ void leapfrog_update(
    float4* __restrict__ positions,
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const int num_particles,
    const float dt,
    const float box_size,
    const double scale_factor,
    const bool update_velocity_first) {
    
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_particles) return;
    
    float4 pos_mass = positions[gid];
    float3 pos;
    pos.x = pos_mass.x;
    pos.y = pos_mass.y;
    pos.z = pos_mass.z;
    float3 vel = velocities[gid];
    float3 force = forces[gid];
    
    const float mass_inv = 1.0f / pos_mass.w;
    const float a2_inv = 1.0f / (scale_factor * scale_factor);
    
    if (update_velocity_first) {
        // Kick step
        vel.x += force.x * mass_inv * dt * a2_inv;
        vel.y += force.y * mass_inv * dt * a2_inv;
        vel.z += force.z * mass_inv * dt * a2_inv;
        velocities[gid] = vel;
    } else {
        // Drift step
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
        
        // Apply periodic boundary conditions
        pos.x = fmodf(pos.x + box_size, box_size);
        pos.y = fmodf(pos.y + box_size, box_size);
        pos.z = fmodf(pos.z + box_size, box_size);
        
        positions[gid].x = pos.x;
        positions[gid].y = pos.y;
        positions[gid].z = pos.z;
        positions[gid].w = pos_mass.w;
    }
}

// Kernel for computing total energy (kinetic + potential)
__global__ void compute_energy(
    const float4* __restrict__ positions,
    const float3* __restrict__ velocities,
    float* __restrict__ kinetic_energy,
    float* __restrict__ potential_energy,
    const int num_particles,
    const float box_size,
    const float softening2) {
    
    extern __shared__ float sh_energy[];
    float* sh_kinetic = sh_energy;
    float* sh_potential = sh_energy + blockDim.x;
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    float ke = 0.0f;
    float pe = 0.0f;
    
    if (gid < num_particles) {
        const float4 my_pos_mass = positions[gid];
        float3 my_pos;
        my_pos.x = my_pos_mass.x;
        my_pos.y = my_pos_mass.y;
        my_pos.z = my_pos_mass.z;
        const float my_mass = my_pos_mass.w;
        const float3 my_vel = velocities[gid];
        
        // Kinetic energy
        ke = 0.5f * my_mass * (my_vel.x * my_vel.x + 
                               my_vel.y * my_vel.y + 
                               my_vel.z * my_vel.z);
        
        // Potential energy (half to avoid double counting)
        for (int j = gid + 1; j < num_particles; j++) {
            const float4 other_pos_mass = positions[j];
            float3 other_pos;
            other_pos.x = other_pos_mass.x;
            other_pos.y = other_pos_mass.y;
            other_pos.z = other_pos_mass.z;
            const float other_mass = other_pos_mass.w;
            
            float3 dr;
            dr.x = other_pos.x - my_pos.x;
            dr.y = other_pos.y - my_pos.y;
            dr.z = other_pos.z - my_pos.z;
            dr = minimum_image(dr, box_size);
            
            const float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + softening2;
            const float r = sqrtf(r2);
            
            pe += -G_CONSTANT * my_mass * other_mass / r;
        }
    }
    
    // Reduce within block
    sh_kinetic[tid] = ke;
    sh_potential[tid] = pe;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sh_kinetic[tid] += sh_kinetic[tid + stride];
            sh_potential[tid] += sh_potential[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results
    if (tid == 0) {
        atomicAdd(kinetic_energy, sh_kinetic[0]);
        atomicAdd(potential_energy, sh_potential[0]);
    }
}

// Initialize CUDA streams for asynchronous operations
class CudaStreamPool {
private:
    static constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    int current_stream = 0;
    
public:
    CudaStreamPool() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~CudaStreamPool() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
    
    cudaStream_t get_next_stream() {
        cudaStream_t stream = streams[current_stream];
        current_stream = (current_stream + 1) % NUM_STREAMS;
        return stream;
    }
    
    void synchronize_all() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }
};

// Host-side wrapper functions
void launch_force_computation(
    const float4* d_positions,
    float3* d_forces,
    int num_particles,
    float box_size,
    float softening,
    cudaStream_t stream) {
    
    const float softening2 = softening * softening;
    
    // Choose kernel based on particle count
    if (num_particles < 10000) {
        // For small N, use simple all-pairs
        const int block_size = 256;
        const int grid_size = (num_particles + block_size - 1) / block_size;
        
        compute_forces_all_pairs<<<grid_size, block_size, 0, stream>>>(
            d_positions, d_forces, num_particles, box_size, softening2);
    } else {
        // For large N, use tiled approach
        const int block_size = TILE_SIZE;
        const int grid_size = (num_particles + block_size - 1) / block_size;
        
        compute_forces_tiled<<<grid_size, block_size, 0, stream>>>(
            d_positions, d_forces, num_particles, box_size, softening2);
    }
}

void launch_leapfrog_update(
    float4* d_positions,
    float3* d_velocities,
    const float3* d_forces,
    int num_particles,
    float dt,
    float box_size,
    double scale_factor,
    bool update_velocity_first,
    cudaStream_t stream) {
    
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    leapfrog_update<<<grid_size, block_size, 0, stream>>>(
        d_positions, d_velocities, d_forces, num_particles,
        dt, box_size, scale_factor, update_velocity_first);
}

void launch_energy_computation(
    const float4* d_positions,
    const float3* d_velocities,
    float* d_kinetic_energy,
    float* d_potential_energy,
    int num_particles,
    float box_size,
    float softening,
    cudaStream_t stream) {
    
    const float softening2 = softening * softening;
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    // Zero the energy counters
    CUDA_CHECK(cudaMemsetAsync(d_kinetic_energy, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_potential_energy, 0, sizeof(float), stream));
    
    compute_energy<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_positions, d_velocities, d_kinetic_energy, d_potential_energy,
        num_particles, box_size, softening2);
}

} // namespace kernels
} // namespace physics