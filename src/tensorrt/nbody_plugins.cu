#include "tensorrt/nbody_plugins.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace tensorrt {

// CUDA kernels for N-body force computation
__global__ void nbody_force_kernel(const float* positions, const float* masses,
                                  float* forces, float* potential,
                                  int num_particles, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    float mass_i = masses[i];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float phi = 0.0f;

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = make_float3(positions[j*3], positions[j*3+1], positions[j*3+2]);
        float mass_j = masses[j];

        float3 r = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
        float r2 = r.x*r.x + r.y*r.y + r.z*r.z + softening*softening;
        float r_inv = rsqrtf(r2);
        float r3_inv = r_inv * r_inv * r_inv;

        float f_mag = mass_j * r3_inv;
        force.x += f_mag * r.x;
        force.y += f_mag * r.y;
        force.z += f_mag * r.z;

        phi -= mass_j * r_inv;
    }

    forces[i*3] = force.x;
    forces[i*3+1] = force.y;
    forces[i*3+2] = force.z;
    potential[i] = phi;
}

// Optimized shared memory version
__global__ void nbody_force_kernel_shared(const float* positions, const float* masses,
                                         float* forces, float* potential,
                                         int num_particles, float softening) {
    extern __shared__ float shared_data[];
    float* shared_pos = shared_data;
    float* shared_mass = &shared_data[blockDim.x * 3];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    float3 pos_i = make_float3(0.0f, 0.0f, 0.0f);
    float mass_i = 0.0f;
    
    if (i < num_particles) {
        pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
        mass_i = masses[i];
    }

    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float phi = 0.0f;

    for (int tile = 0; tile < gridDim.x; ++tile) {
        int j = tile * blockDim.x + tid;
        
        // Load tile data into shared memory
        if (j < num_particles) {
            shared_pos[tid*3] = positions[j*3];
            shared_pos[tid*3+1] = positions[j*3+1];
            shared_pos[tid*3+2] = positions[j*3+2];
            shared_mass[tid] = masses[j];
        } else {
            shared_pos[tid*3] = 0.0f;
            shared_pos[tid*3+1] = 0.0f;
            shared_pos[tid*3+2] = 0.0f;
            shared_mass[tid] = 0.0f;
        }
        
        __syncthreads();

        // Compute forces with particles in this tile
        if (i < num_particles) {
            for (int k = 0; k < blockDim.x; ++k) {
                int global_j = tile * blockDim.x + k;
                if (global_j >= num_particles || global_j == i) continue;

                float3 pos_j = make_float3(shared_pos[k*3], shared_pos[k*3+1], shared_pos[k*3+2]);
                float mass_j = shared_mass[k];

                float3 r = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
                float r2 = r.x*r.x + r.y*r.y + r.z*r.z + softening*softening;
                float r_inv = rsqrtf(r2);
                float r3_inv = r_inv * r_inv * r_inv;

                float f_mag = mass_j * r3_inv;
                force.x += f_mag * r.x;
                force.y += f_mag * r.y;
                force.z += f_mag * r.z;

                phi -= mass_j * r_inv;
            }
        }
        
        __syncthreads();
    }

    if (i < num_particles) {
        forces[i*3] = force.x;
        forces[i*3+1] = force.y;
        forces[i*3+2] = force.z;
        potential[i] = phi;
    }
}

// Tree force kernel (simplified version - would need full octree implementation)
__global__ void tree_force_kernel(const float* positions, const float* masses,
                                 float* forces, int num_particles, float theta,
                                 void* workspace, size_t workspace_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Placeholder for tree traversal algorithm
    // In practice, this would implement Barnes-Hut tree traversal
    // For now, fall back to direct computation with reduced precision
    
    float3 pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Simple approximation - skip distant particles based on distance
    for (int j = 0; j < num_particles; j += max(1, (int)(1.0f/theta))) {
        if (i == j) continue;

        float3 pos_j = make_float3(positions[j*3], positions[j*3+1], positions[j*3+2]);
        float mass_j = masses[j];

        float3 r = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
        float r2 = r.x*r.x + r.y*r.y + r.z*r.z + 0.001f;
        float r_inv = rsqrtf(r2);
        float r3_inv = r_inv * r_inv * r_inv;

        float f_mag = mass_j * r3_inv;
        force.x += f_mag * r.x;
        force.y += f_mag * r.y;
        force.z += f_mag * r.z;
    }

    forces[i*3] = force.x;
    forces[i*3+1] = force.y;
    forces[i*3+2] = force.z;
}

// C interface functions
extern "C" {
    void launch_nbody_force_kernel(const float* positions, const float* masses,
                                 float* forces, float* potential,
                                 int num_particles, float softening,
                                 cudaStream_t stream) {
        const int block_size = 256;
        const int grid_size = (num_particles + block_size - 1) / block_size;
        
        // Use shared memory version for better performance
        size_t shared_mem_size = block_size * (3 + 1) * sizeof(float);
        
        nbody_force_kernel_shared<<<grid_size, block_size, shared_mem_size, stream>>>(
            positions, masses, forces, potential, num_particles, softening);
    }

    void launch_tree_force_kernel(const float* positions, const float* masses,
                                float* forces, int num_particles, float theta,
                                void* workspace, size_t workspace_size,
                                cudaStream_t stream) {
        const int block_size = 256;
        const int grid_size = (num_particles + block_size - 1) / block_size;
        
        tree_force_kernel<<<grid_size, block_size, 0, stream>>>(
            positions, masses, forces, num_particles, theta, workspace, workspace_size);
    }
}

// NBodyForcePlugin implementation
NBodyForcePlugin::NBodyForcePlugin(float softening, int max_particles)
    : softening_length_(softening), max_particles_(max_particles) {}

NBodyForcePlugin::NBodyForcePlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    softening_length_ = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    max_particles_ = *reinterpret_cast<const int*>(d);
}

nvinfer1::IPluginV2DynamicExt* NBodyForcePlugin::clone() const noexcept {
    return new NBodyForcePlugin(softening_length_, max_particles_);
}

nvinfer1::DimsExprs NBodyForcePlugin::getOutputDimensions(int outputIndex, 
                                                        const nvinfer1::DimsExprs* inputs,
                                                        int nbInputs, 
                                                        nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (outputIndex == 0) {
        // Forces output: same as positions input
        return inputs[0];
    } else if (outputIndex == 1) {
        // Potential output: [N, 1]
        nvinfer1::DimsExprs output;
        output.nbDims = 2;
        output.d[0] = inputs[0].d[0];  // batch size
        output.d[1] = exprBuilder.constant(1);
        return output;
    }
    return nvinfer1::DimsExprs{};
}

bool NBodyForcePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                int nbInputs, int nbOutputs) noexcept {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && 
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void NBodyForcePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    // Configuration is handled during runtime
}

size_t NBodyForcePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0; // No additional workspace needed for direct N-body
}

int NBodyForcePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                            const nvinfer1::PluginTensorDesc* outputDesc,
                            const void* const* inputs, void* const* outputs,
                            void* workspace, cudaStream_t stream) noexcept {
    int num_particles = inputDesc[0].dims.d[0];
    
    const float* positions = static_cast<const float*>(inputs[0]);
    const float* masses = static_cast<const float*>(inputs[1]);
    float* forces = static_cast<float*>(outputs[0]);
    float* potential = static_cast<float*>(outputs[1]);

    launch_nbody_force_kernel(positions, masses, forces, potential,
                            num_particles, softening_length_, stream);

    return 0;
}

nvinfer1::DataType NBodyForcePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                      int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

const char* NBodyForcePlugin::getPluginType() const noexcept {
    return "NBodyForce";
}

const char* NBodyForcePlugin::getPluginVersion() const noexcept {
    return "1";
}

int NBodyForcePlugin::getNbOutputs() const noexcept {
    return 2; // forces and potential
}

int NBodyForcePlugin::initialize() noexcept {
    return 0;
}

void NBodyForcePlugin::terminate() noexcept {
}

size_t NBodyForcePlugin::getSerializationSize() const noexcept {
    return sizeof(float) + sizeof(int);
}

void NBodyForcePlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<float*>(d) = softening_length_;
    d += sizeof(float);
    *reinterpret_cast<int*>(d) = max_particles_;
}

void NBodyForcePlugin::destroy() noexcept {
    delete this;
}

void NBodyForcePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* NBodyForcePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// NBodyForcePluginCreator implementation
NBodyForcePluginCreator::NBodyForcePluginCreator() {
    field_collection_.clear();
    field_collection_.emplace_back("softening", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    field_collection_.emplace_back("max_particles", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    
    fc_.nbFields = field_collection_.size();
    fc_.fields = field_collection_.data();
}

const char* NBodyForcePluginCreator::getPluginName() const noexcept {
    return "NBodyForce";
}

const char* NBodyForcePluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* NBodyForcePluginCreator::getFieldNames() noexcept {
    return &fc_;
}

nvinfer1::IPluginV2* NBodyForcePluginCreator::createPlugin(const char* name,
                                                         const nvinfer1::PluginFieldCollection* fc) noexcept {
    float softening = 0.01f;
    int max_particles = 100000;

    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fc->fields[i].name, "softening") == 0) {
            softening = *static_cast<const float*>(fc->fields[i].data);
        } else if (strcmp(fc->fields[i].name, "max_particles") == 0) {
            max_particles = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    return new NBodyForcePlugin(softening, max_particles);
}

nvinfer1::IPluginV2* NBodyForcePluginCreator::deserializePlugin(const char* name,
                                                               const void* serialData,
                                                               size_t serialLength) noexcept {
    return new NBodyForcePlugin(serialData, serialLength);
}

void NBodyForcePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* NBodyForcePluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// TreeForcePlugin implementation (simplified - would need full implementation)
TreeForcePlugin::TreeForcePlugin(float theta, int max_particles)
    : theta_(theta), max_particles_(max_particles) {}

TreeForcePlugin::TreeForcePlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    theta_ = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    max_particles_ = *reinterpret_cast<const int*>(d);
}

nvinfer1::IPluginV2DynamicExt* TreeForcePlugin::clone() const noexcept {
    return new TreeForcePlugin(theta_, max_particles_);
}

nvinfer1::DimsExprs TreeForcePlugin::getOutputDimensions(int outputIndex,
                                                       const nvinfer1::DimsExprs* inputs,
                                                       int nbInputs,
                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept {
    return inputs[0]; // Same as positions input
}

bool TreeForcePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                               int nbInputs, int nbOutputs) noexcept {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && 
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void TreeForcePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
}

size_t TreeForcePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    int num_particles = inputs[0].dims.d[0];
    // Simplified workspace calculation - would need proper octree memory estimation
    return num_particles * 8 * sizeof(float); // Rough estimate for tree nodes
}

int TreeForcePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                           const nvinfer1::PluginTensorDesc* outputDesc,
                           const void* const* inputs, void* const* outputs,
                           void* workspace, cudaStream_t stream) noexcept {
    int num_particles = inputDesc[0].dims.d[0];
    
    const float* positions = static_cast<const float*>(inputs[0]);
    const float* masses = static_cast<const float*>(inputs[1]);
    float* forces = static_cast<float*>(outputs[0]);

    size_t workspace_size = getWorkspaceSize(inputDesc, 2, outputDesc, 1);
    
    launch_tree_force_kernel(positions, masses, forces, num_particles, theta_,
                           workspace, workspace_size, stream);

    return 0;
}

nvinfer1::DataType TreeForcePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                     int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

const char* TreeForcePlugin::getPluginType() const noexcept {
    return "TreeForce";
}

const char* TreeForcePlugin::getPluginVersion() const noexcept {
    return "1";
}

int TreeForcePlugin::getNbOutputs() const noexcept {
    return 1; // forces only
}

int TreeForcePlugin::initialize() noexcept {
    return 0;
}

void TreeForcePlugin::terminate() noexcept {
}

size_t TreeForcePlugin::getSerializationSize() const noexcept {
    return sizeof(float) + sizeof(int);
}

void TreeForcePlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<float*>(d) = theta_;
    d += sizeof(float);
    *reinterpret_cast<int*>(d) = max_particles_;
}

void TreeForcePlugin::destroy() noexcept {
    delete this;
}

void TreeForcePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* TreeForcePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// TreeForcePluginCreator implementation
TreeForcePluginCreator::TreeForcePluginCreator() {
    field_collection_.clear();
    field_collection_.emplace_back("theta", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    field_collection_.emplace_back("max_particles", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    
    fc_.nbFields = field_collection_.size();
    fc_.fields = field_collection_.data();
}

const char* TreeForcePluginCreator::getPluginName() const noexcept {
    return "TreeForce";
}

const char* TreeForcePluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* TreeForcePluginCreator::getFieldNames() noexcept {
    return &fc_;
}

nvinfer1::IPluginV2* TreeForcePluginCreator::createPlugin(const char* name,
                                                         const nvinfer1::PluginFieldCollection* fc) noexcept {
    float theta = 0.5f;
    int max_particles = 100000;

    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fc->fields[i].name, "theta") == 0) {
            theta = *static_cast<const float*>(fc->fields[i].data);
        } else if (strcmp(fc->fields[i].name, "max_particles") == 0) {
            max_particles = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    return new TreeForcePlugin(theta, max_particles);
}

nvinfer1::IPluginV2* TreeForcePluginCreator::deserializePlugin(const char* name,
                                                              const void* serialData,
                                                              size_t serialLength) noexcept {
    return new TreeForcePlugin(serialData, serialLength);
}

void TreeForcePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* TreeForcePluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

}