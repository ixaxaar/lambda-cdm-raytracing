#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>

namespace tensorrt {

class NBodyForcePlugin : public nvinfer1::IPluginV2DynamicExt {
private:
    float softening_length_;
    int max_particles_;
    std::string namespace_;

public:
    NBodyForcePlugin(float softening = 0.01f, int max_particles = 100000);
    NBodyForcePlugin(const void* data, size_t length);
    ~NBodyForcePlugin() override = default;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                           const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
               const void* const* inputs, void* const* outputs, void* workspace,
               cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                        int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

class NBodyForcePluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string namespace_;
    std::vector<nvinfer1::PluginField> field_collection_;
    nvinfer1::PluginFieldCollection fc_;

public:
    NBodyForcePluginCreator();
    ~NBodyForcePluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

class TreeForcePlugin : public nvinfer1::IPluginV2DynamicExt {
private:
    float theta_;
    int max_particles_;
    std::string namespace_;

public:
    TreeForcePlugin(float theta = 0.5f, int max_particles = 100000);
    TreeForcePlugin(const void* data, size_t length);
    ~TreeForcePlugin() override = default;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                           const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
               const void* const* inputs, void* const* outputs, void* workspace,
               cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                        int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

class TreeForcePluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string namespace_;
    std::vector<nvinfer1::PluginField> field_collection_;
    nvinfer1::PluginFieldCollection fc_;

public:
    TreeForcePluginCreator();
    ~TreeForcePluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
};

// CUDA kernel declarations
extern "C" {
    void launch_nbody_force_kernel(const float* positions, const float* masses,
                                 float* forces, float* potential,
                                 int num_particles, float softening,
                                 cudaStream_t stream);

    void launch_tree_force_kernel(const float* positions, const float* masses,
                                float* forces, int num_particles, float theta,
                                void* workspace, size_t workspace_size,
                                cudaStream_t stream);
}

}