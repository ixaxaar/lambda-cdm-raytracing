#include "tensorrt/nbody_engine.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

namespace tensorrt {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

static Logger gLogger;

NBodyEngine::NBodyEngine(size_t max_particles, float softening)
    : max_particles_(max_particles), softening_length_(softening) {
    
    cudaStreamCreate(&stream_);
    
    for (int i = 0; i < 4; ++i) {
        bindings_[i] = nullptr;
    }
    
    // Allocate GPU memory for bindings
    cudaMalloc(&bindings_[0], max_particles_ * 3 * sizeof(float)); // positions
    cudaMalloc(&bindings_[1], max_particles_ * sizeof(float));     // masses
    cudaMalloc(&bindings_[2], max_particles_ * 3 * sizeof(float)); // forces
    cudaMalloc(&bindings_[3], max_particles_ * sizeof(float));     // potential
}

NBodyEngine::~NBodyEngine() {
    for (int i = 0; i < 4; ++i) {
        if (bindings_[i]) {
            cudaFree(bindings_[i]);
        }
    }
    cudaStreamDestroy(stream_);
}

bool NBodyEngine::build_engine() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger));
    if (!builder) return false;
    
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) return false;
    
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) return false;
    
    // Input tensors
    auto positions = network->addInput("positions", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims3{static_cast<int>(max_particles_), 3, 1});
    auto masses = network->addInput("masses", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims2{static_cast<int>(max_particles_), 1});
    
    if (!positions || !masses) return false;
    
    // Create custom N-body force computation layer
    // This would typically involve implementing a custom TensorRT plugin
    // For now, we'll create a simplified version using existing layers
    
    // Expand positions for pairwise computation
    auto pos_expanded = network->addShuffle(*positions);
    pos_expanded->setReshapeDimensions(
        nvinfer1::Dims4{static_cast<int>(max_particles_), 1, 3, 1});
    
    // Tile positions for all pairs
    auto pos_tiled = network->addShuffle(*pos_expanded->getOutput(0));
    pos_tiled->setReshapeDimensions(
        nvinfer1::Dims4{static_cast<int>(max_particles_), 
                       static_cast<int>(max_particles_), 3, 1});
    
    // Create force computation network (simplified)
    // In practice, this would use a custom plugin for efficiency
    
    // Output layers
    network->markOutput(*pos_tiled->getOutput(0)); // forces placeholder
    network->markOutput(*masses->getOutput(0));    // potential placeholder
    
    // Set optimization profiles
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{1, 3, 1});
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{static_cast<int>(max_particles_/2), 3, 1});
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{static_cast<int>(max_particles_), 3, 1});
    
    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims2{1, 1});
    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims2{static_cast<int>(max_particles_/2), 1});
    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims2{static_cast<int>(max_particles_), 1});
    
    config->addOptimizationProfile(profile);
    
    // Enable FP16 precision for performance
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    
    // Build engine
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config));
    
    if (!engine_) return false;
    
    // Create execution context
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    
    return context_ != nullptr;
}

bool NBodyEngine::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> trt_model_stream(size);
    file.read(trt_model_stream.data(), size);
    file.close();
    
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) return false;
    
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(trt_model_stream.data(), size));
    if (!engine_) return false;
    
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    
    return context_ != nullptr;
}

bool NBodyEngine::save_engine(const std::string& engine_path) {
    if (!engine_) return false;
    
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    if (!serialized_engine) return false;
    
    std::ofstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;
    
    file.write(static_cast<const char*>(serialized_engine->data()),
               serialized_engine->size());
    file.close();
    
    return true;
}

void NBodyEngine::compute_forces(const float* positions, const float* masses,
                                float* forces, float* potential, 
                                size_t num_particles) {
    if (!context_ || num_particles > max_particles_) return;
    
    // Copy input data to GPU
    cudaMemcpyAsync(bindings_[0], positions, 
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(bindings_[1], masses,
                   num_particles * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    
    // Set input binding dimensions
    context_->setBindingDimensions(0, 
        nvinfer1::Dims3{static_cast<int>(num_particles), 3, 1});
    context_->setBindingDimensions(1,
        nvinfer1::Dims2{static_cast<int>(num_particles), 1});
    
    // Execute inference
    bool success = context_->enqueueV2(bindings_, stream_, nullptr);
    assert(success);
    
    // Copy results back to host
    cudaMemcpyAsync(forces, bindings_[2],
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(potential, bindings_[3],
                   num_particles * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
}

TreeCodeEngine::TreeCodeEngine(float theta) : theta_(theta) {
    // Implementation for tree-based force computation
    // This would use a more sophisticated TensorRT network
    // with custom plugins for octree traversal
}

TreeCodeEngine::~TreeCodeEngine() = default;

bool TreeCodeEngine::build_engine() {
    // Implementation for building tree-based force computation engine
    // This is more complex and would require custom TensorRT plugins
    return false; // Placeholder
}

void TreeCodeEngine::compute_forces_tree(const float* positions, 
                                        const float* masses,
                                        float* forces, size_t num_particles) {
    // Implementation for tree-based force computation
    // Would use Barnes-Hut algorithm implemented as TensorRT custom layer
}

}