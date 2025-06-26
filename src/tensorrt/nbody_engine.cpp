#include "tensorrt/nbody_engine.hpp"
#include "tensorrt/nbody_plugins.hpp"
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
    // Register custom plugins
    static NBodyForcePluginCreator nbody_creator;
    nvinfer1::getPluginRegistry()->registerCreator(nbody_creator, "");

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

    // Create plugin field collection for N-body force plugin
    std::vector<nvinfer1::PluginField> plugin_fields;
    plugin_fields.emplace_back("softening", &softening_length_,
                              nvinfer1::PluginFieldType::kFLOAT32, 1);
    int max_particles = static_cast<int>(max_particles_);
    plugin_fields.emplace_back("max_particles", &max_particles,
                              nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::PluginFieldCollection plugin_data;
    plugin_data.nbFields = plugin_fields.size();
    plugin_data.fields = plugin_fields.data();

    // Create N-body force plugin
    auto nbody_plugin = nbody_creator.createPlugin("nbody_force", &plugin_data);
    if (!nbody_plugin) return false;

    // Add plugin layer to network
    std::vector<nvinfer1::ITensor*> plugin_inputs = {positions, masses};
    auto nbody_layer = network->addPluginV2(plugin_inputs.data(),
                                           plugin_inputs.size(), *nbody_plugin);
    if (!nbody_layer) return false;

    // Mark outputs
    network->markOutput(*nbody_layer->getOutput(0)); // forces
    network->markOutput(*nbody_layer->getOutput(1)); // potential

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
    for (int i = 0; i < 3; ++i) {
        bindings_[i] = nullptr;
    }
}

TreeCodeEngine::~TreeCodeEngine() {
    for (int i = 0; i < 3; ++i) {
        if (bindings_[i]) {
            cudaFree(bindings_[i]);
        }
    }
    if (bindings_[0]) { // Only destroy if we created it
        cudaStreamDestroy(stream_);
    }
}

bool TreeCodeEngine::build_engine() {
    // Register custom plugins
    static TreeForcePluginCreator tree_creator;
    nvinfer1::getPluginRegistry()->registerCreator(tree_creator, "");

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
        nvinfer1::Dims3{100000, 3, 1}); // Default max size
    auto masses = network->addInput("masses", nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims2{100000, 1});

    if (!positions || !masses) return false;

    // Create plugin field collection for tree force plugin
    std::vector<nvinfer1::PluginField> plugin_fields;
    plugin_fields.emplace_back("theta", &theta_,
                              nvinfer1::PluginFieldType::kFLOAT32, 1);
    int max_particles = 100000;
    plugin_fields.emplace_back("max_particles", &max_particles,
                              nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::PluginFieldCollection plugin_data;
    plugin_data.nbFields = plugin_fields.size();
    plugin_data.fields = plugin_fields.data();

    // Create tree force plugin
    auto tree_plugin = tree_creator.createPlugin("tree_force", &plugin_data);
    if (!tree_plugin) return false;

    // Add plugin layer to network
    std::vector<nvinfer1::ITensor*> plugin_inputs = {positions, masses};
    auto tree_layer = network->addPluginV2(plugin_inputs.data(),
                                          plugin_inputs.size(), *tree_plugin);
    if (!tree_layer) return false;

    // Mark output
    network->markOutput(*tree_layer->getOutput(0)); // forces

    // Set optimization profiles
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{1, 3, 1});
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{50000, 3, 1});
    profile->setDimensions("positions", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{100000, 3, 1});

    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims2{1, 1});
    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims2{50000, 1});
    profile->setDimensions("masses", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims2{100000, 1});

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

void TreeCodeEngine::compute_forces_tree(const float* positions,
                                        const float* masses,
                                        float* forces, size_t num_particles) {
    if (!context_ || num_particles > 100000) return;

    // Allocate GPU memory if not already done
    if (!bindings_[0]) {
        cudaMalloc(&bindings_[0], 100000 * 3 * sizeof(float)); // positions
        cudaMalloc(&bindings_[1], 100000 * sizeof(float));     // masses
        cudaMalloc(&bindings_[2], 100000 * 3 * sizeof(float)); // forces
        cudaStreamCreate(&stream_);
    }

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
    if (!success) return;

    // Copy results back to host
    cudaMemcpyAsync(forces, bindings_[2],
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

}