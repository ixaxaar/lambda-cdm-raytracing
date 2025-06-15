#include "forces/force_computer_factory.hpp"
#include <iostream>

namespace forces {

// Static member definitions
std::unordered_map<std::string, std::function<std::unique_ptr<core::IForceComputer>(const std::string&)>> 
    ForceComputerFactory::factories_;

std::unordered_map<std::string, std::function<std::unique_ptr<IForceKernel>()>> 
    ForceComputerFactory::kernel_factories_;

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_force_computer(
    const std::string& type, const std::string& name, const ForceComputeParameters& params) {
    
    auto it = factories_.find(type);
    if (it != factories_.end()) {
        auto computer = it->second(name);
        if (computer) {
            std::cout << "Created force computer: " << type << " (" << name << ")" << std::endl;
        }
        return computer;
    }
    
    std::cerr << "Unknown force computer type: " << type << std::endl;
    return nullptr;
}

std::unique_ptr<IForceKernel> ForceComputerFactory::create_force_kernel(const std::string& type) {
    auto it = kernel_factories_.find(type);
    if (it != kernel_factories_.end()) {
        return it->second();
    }
    
    std::cerr << "Unknown force kernel type: " << type << std::endl;
    return nullptr;
}

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_direct_computer(
    const std::string& name, const ForceComputeParameters& params) {
    return create_force_computer("DirectForceComputer", name, params);
}

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_tree_computer(
    const std::string& name, const ForceComputeParameters& params) {
    return create_force_computer("TreeForceComputer", name, params);
}

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_pm_computer(
    const std::string& name, const ForceComputeParameters& params) {
    return create_force_computer("PMForceComputer", name, params);
}

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_tensorrt_computer(
    const std::string& name, const ForceComputeParameters& params) {
    return create_force_computer("TensorRTForceComputer", name, params);
}

std::unique_ptr<core::IForceComputer> ForceComputerFactory::create_hybrid_computer(
    const std::string& name, const ForceComputeParameters& params) {
    return create_force_computer("HybridForceComputer", name, params);
}

std::vector<std::string> ForceComputerFactory::get_available_computers() {
    std::vector<std::string> types;
    for (const auto& pair : factories_) {
        types.push_back(pair.first);
    }
    return types;
}

std::vector<std::string> ForceComputerFactory::get_available_kernels() {
    std::vector<std::string> types;
    for (const auto& pair : kernel_factories_) {
        types.push_back(pair.first);
    }
    return types;
}

ForceComputeParameters ForceComputerFactory::get_recommended_parameters(
    ForceComputeMethod method, size_t num_particles) {
    
    ForceComputeParameters params;
    params.method = method;
    
    switch (method) {
        case ForceComputeMethod::DIRECT:
            params.use_gpu = num_particles > 1000;
            break;
            
        case ForceComputeMethod::TREE:
            params.theta = (num_particles > 100000) ? 0.7f : 0.5f;
            params.leaf_capacity = 8;
            params.use_gpu = true;
            break;
            
        case ForceComputeMethod::PARTICLE_MESH:
            params.grid_size = static_cast<size_t>(std::cbrt(num_particles / 8.0));
            params.use_gpu = true;
            break;
            
        case ForceComputeMethod::TENSORRT:
            params.use_tensorrt = true;
            params.use_gpu = true;
            break;
            
        default:
            break;
    }
    
    return params;
}

ForceComputeMethod ForceComputerFactory::select_optimal_method(
    size_t num_particles, bool has_gpu, bool has_tensorrt) {
    
    if (num_particles < 1000) {
        return ForceComputeMethod::DIRECT;
    } else if (num_particles < 100000) {
        return has_gpu ? ForceComputeMethod::TREE : ForceComputeMethod::DIRECT;
    } else if (has_tensorrt) {
        return ForceComputeMethod::TENSORRT;
    } else {
        return ForceComputeMethod::TREE;
    }
}

void ForceComputerFactory::register_all_builtin_computers() {
    // Placeholder implementations - these would register actual classes
    std::cout << "Registering built-in force computers..." << std::endl;
    
    // TODO: Register actual implementations
    // register_force_computer<DirectForceComputer>("DirectForceComputer");
    // register_force_computer<TreeForceComputer>("TreeForceComputer");
    // register_force_computer<PMForceComputer>("PMForceComputer");
    // register_force_computer<TensorRTForceComputer>("TensorRTForceComputer");
}

void ForceComputerFactory::register_all_builtin_kernels() {
    std::cout << "Registering built-in force kernels..." << std::endl;
    
    // Register the Newtonian kernel
    register_force_kernel<NewtonianGravityKernel>("Newtonian");
    register_force_kernel<ModifiedGravityKernel>("ModifiedGravity");
}

// Force kernel implementations
void NewtonianGravityKernel::compute_pairwise_force(
    const float3& pos1, const float3& pos2,
    float mass1, float mass2,
    float3& force1, float3& force2,
    const std::any& params) const {
    
    // Simple Newtonian gravity calculation
    float3 dr;
    dr.x = pos2.x - pos1.x;
    dr.y = pos2.y - pos1.y;
    dr.z = pos2.z - pos1.z;
    
    float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    float softening = 0.01f; // Default softening
    r2 += softening * softening;
    
    float r = sqrtf(r2);
    float r3 = r2 * r;
    
    float force_magnitude = mass1 * mass2 / r3;
    
    force1.x = -force_magnitude * dr.x;
    force1.y = -force_magnitude * dr.y;
    force1.z = -force_magnitude * dr.z;
    
    force2.x = force_magnitude * dr.x;
    force2.y = force_magnitude * dr.y;
    force2.z = force_magnitude * dr.z;
}

float NewtonianGravityKernel::compute_potential(
    const float3& pos1, const float3& pos2,
    float mass1, float mass2,
    const std::any& params) const {
    
    float3 dr;
    dr.x = pos2.x - pos1.x;
    dr.y = pos2.y - pos1.y;
    dr.z = pos2.z - pos1.z;
    
    float r = sqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
    float softening = 0.01f;
    r = fmaxf(r, softening);
    
    return -mass1 * mass2 / r;
}

void ModifiedGravityKernel::compute_pairwise_force(
    const float3& pos1, const float3& pos2,
    float mass1, float mass2,
    float3& force1, float3& force2,
    const std::any& params) const {
    
    // Modified gravity with parameter
    float3 dr;
    dr.x = pos2.x - pos1.x;
    dr.y = pos2.y - pos1.y;
    dr.z = pos2.z - pos1.z;
    
    float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    float softening = 0.01f;
    r2 += softening * softening;
    
    float r = sqrtf(r2);
    float r3 = r2 * r;
    
    // Apply modification
    float force_magnitude = mass1 * mass2 / r3 * modification_parameter_;
    
    force1.x = -force_magnitude * dr.x;
    force1.y = -force_magnitude * dr.y;
    force1.z = -force_magnitude * dr.z;
    
    force2.x = force_magnitude * dr.x;
    force2.y = force_magnitude * dr.y;
    force2.z = force_magnitude * dr.z;
}

float ModifiedGravityKernel::compute_potential(
    const float3& pos1, const float3& pos2,
    float mass1, float mass2,
    const std::any& params) const {
    
    float3 dr;
    dr.x = pos2.x - pos1.x;
    dr.y = pos2.y - pos1.y;
    dr.z = pos2.z - pos1.z;
    
    float r = sqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
    float softening = 0.01f;
    r = fmaxf(r, softening);
    
    return -mass1 * mass2 / r * modification_parameter_;
}

}