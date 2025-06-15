#pragma once

#include "core/interfaces.hpp"
#include "core/math_types.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <cmath>

namespace forces {

// Forward declarations
class DirectForceComputer;
class TreeForceComputer;
class PMForceComputer;
class TensorRTForceComputer;
class FMMForceComputer;

enum class ForceComputeMethod {
    DIRECT,
    TREE,
    PARTICLE_MESH,
    TENSORRT,
    FAST_MULTIPOLE,
    HYBRID
};

struct ForceComputeParameters {
    ForceComputeMethod method = ForceComputeMethod::DIRECT;
    float softening_length = 0.01f;
    float theta = 0.5f;  // Opening angle for tree methods
    size_t grid_size = 64;  // For PM methods
    int tree_max_depth = 20;
    size_t leaf_capacity = 8;
    bool use_gpu = true;
    bool use_tensorrt = false;
    std::string tensorrt_engine_path;
    int cuda_device_id = 0;
    std::unordered_map<std::string, std::any> custom_params;
};

class IForceKernel {
public:
    virtual ~IForceKernel() = default;
    virtual void compute_pairwise_force(const float3& pos1, const float3& pos2,
                                       float mass1, float mass2,
                                       float3& force1, float3& force2,
                                       const std::any& params = {}) const = 0;
    virtual float compute_potential(const float3& pos1, const float3& pos2,
                                   float mass1, float mass2,
                                   const std::any& params = {}) const = 0;
    virtual std::string get_name() const = 0;
    virtual bool supports_gpu() const = 0;
};

class NewtonianGravityKernel : public IForceKernel {
public:
    void compute_pairwise_force(const float3& pos1, const float3& pos2,
                               float mass1, float mass2,
                               float3& force1, float3& force2,
                               const std::any& params = {}) const override;
    
    float compute_potential(const float3& pos1, const float3& pos2,
                           float mass1, float mass2,
                           const std::any& params = {}) const override;
    
    std::string get_name() const override { return "Newtonian"; }
    bool supports_gpu() const override { return true; }
};

class ModifiedGravityKernel : public IForceKernel {
private:
    float modification_parameter_;
    
public:
    explicit ModifiedGravityKernel(float mod_param = 1.0f) : modification_parameter_(mod_param) {}
    
    void compute_pairwise_force(const float3& pos1, const float3& pos2,
                               float mass1, float mass2,
                               float3& force1, float3& force2,
                               const std::any& params = {}) const override;
    
    float compute_potential(const float3& pos1, const float3& pos2,
                           float mass1, float mass2,
                           const std::any& params = {}) const override;
    
    std::string get_name() const override { return "ModifiedGravity"; }
    bool supports_gpu() const override { return true; }
    
    void set_modification_parameter(float param) { modification_parameter_ = param; }
};

class ForceComputerFactory {
private:
    static std::unordered_map<std::string, std::function<std::unique_ptr<core::IForceComputer>(const std::string&)>> factories_;
    static std::unordered_map<std::string, std::function<std::unique_ptr<IForceKernel>()>> kernel_factories_;
    
public:
    // Factory registration
    template<typename T>
    static void register_force_computer(const std::string& name);
    
    template<typename T>
    static void register_force_kernel(const std::string& name);
    
    // Component creation
    static std::unique_ptr<core::IForceComputer> create_force_computer(
        const std::string& type, const std::string& name,
        const ForceComputeParameters& params = {});
    
    static std::unique_ptr<IForceKernel> create_force_kernel(const std::string& type);
    
    // Convenience methods
    static std::unique_ptr<core::IForceComputer> create_direct_computer(
        const std::string& name, const ForceComputeParameters& params = {});
    
    static std::unique_ptr<core::IForceComputer> create_tree_computer(
        const std::string& name, const ForceComputeParameters& params = {});
    
    static std::unique_ptr<core::IForceComputer> create_pm_computer(
        const std::string& name, const ForceComputeParameters& params = {});
    
    static std::unique_ptr<core::IForceComputer> create_tensorrt_computer(
        const std::string& name, const ForceComputeParameters& params = {});
    
    static std::unique_ptr<core::IForceComputer> create_hybrid_computer(
        const std::string& name, const ForceComputeParameters& params = {});
    
    // Utility functions
    static std::vector<std::string> get_available_computers();
    static std::vector<std::string> get_available_kernels();
    static ForceComputeParameters get_recommended_parameters(ForceComputeMethod method, size_t num_particles);
    
    // Auto-selection based on problem size and hardware
    static ForceComputeMethod select_optimal_method(size_t num_particles, bool has_gpu, bool has_tensorrt);
    
    // Register all built-in components
    static void register_all_builtin_computers();
    static void register_all_builtin_kernels();
};

// Template implementations
template<typename T>
void ForceComputerFactory::register_force_computer(const std::string& name) {
    factories_[name] = [](const std::string& instance_name) -> std::unique_ptr<core::IForceComputer> {
        return std::make_unique<T>(instance_name);
    };
}

template<typename T>
void ForceComputerFactory::register_force_kernel(const std::string& name) {
    kernel_factories_[name] = []() -> std::unique_ptr<IForceKernel> {
        return std::make_unique<T>();
    };
}

}