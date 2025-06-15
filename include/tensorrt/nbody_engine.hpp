#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace tensorrt {

class NBodyEngine {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* bindings_[4];  // positions_in, masses_in, forces_out, potential_out
    cudaStream_t stream_;

    size_t max_particles_;
    float softening_length_;

public:
    NBodyEngine(size_t max_particles, float softening = 0.01f);
    ~NBodyEngine();

    bool build_engine();
    bool load_engine(const std::string& engine_path);
    bool save_engine(const std::string& engine_path);

    void compute_forces(const float* positions, const float* masses,
                       float* forces, float* potential, size_t num_particles);

    void set_softening_length(float softening) { softening_length_ = softening; }
    size_t get_max_particles() const { return max_particles_; }
};

class TreeCodeEngine {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* bindings_[3];  // positions_in, masses_in, forces_out
    cudaStream_t stream_;
    float theta_; // Opening angle for tree approximation

public:
    TreeCodeEngine(float theta = 0.5f);
    ~TreeCodeEngine();

    bool build_engine();
    void compute_forces_tree(const float* positions, const float* masses,
                           float* forces, size_t num_particles);
};

}