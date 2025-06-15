#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <any>
#include <typeindex>
#include <cstdint>

namespace core {

// Forward declarations
class SimulationContext;
class ComponentRegistry;
class ConfigurationManager;

// Core interfaces for production-grade architecture

class IComponent {
public:
    virtual ~IComponent() = default;
    virtual bool initialize(const SimulationContext& context) = 0;
    virtual void finalize() = 0;
    virtual std::string get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
};

class IForceComputer : public IComponent {
public:
    virtual ~IForceComputer() = default;
    virtual void compute_forces(const float* positions, const float* masses,
                               float* forces, size_t num_particles,
                               const std::any& params = {}) = 0;
    virtual bool supports_gpu() const = 0;
    virtual bool supports_mpi() const = 0;
    virtual size_t get_max_particles() const = 0;
};

class IIntegrator : public IComponent {
public:
    virtual ~IIntegrator() = default;
    virtual void step(float* positions, float* velocities, const float* forces,
                     size_t num_particles, double dt, const std::any& params = {}) = 0;
    virtual double get_recommended_timestep() const = 0;
    virtual bool is_symplectic() const = 0;
};

class ICosmologyModel : public IComponent {
public:
    virtual ~ICosmologyModel() = default;
    virtual double hubble_function(double scale_factor) const = 0;
    virtual double growth_factor(double scale_factor) const = 0;
    virtual double omega_matter(double scale_factor) const = 0;
    virtual double omega_lambda(double scale_factor) const = 0;
    virtual void update_scale_factor(double& scale_factor, double dt) const = 0;
};

class IParticleGenerator : public IComponent {
public:
    virtual ~IParticleGenerator() = default;
    virtual void generate_particles(float* positions, float* velocities, float* masses,
                                   size_t num_particles, const std::any& params = {}) = 0;
    virtual bool supports_power_spectrum() const = 0;
    virtual void set_random_seed(uint64_t seed) = 0;
};

class IDataExporter : public IComponent {
public:
    virtual ~IDataExporter() = default;
    virtual bool export_snapshot(const std::string& filename,
                                const float* positions, const float* velocities,
                                const float* masses, size_t num_particles,
                                double time, const std::any& metadata = {}) = 0;
    virtual bool import_snapshot(const std::string& filename,
                                float* positions, float* velocities,
                                float* masses, size_t& num_particles,
                                double& time, std::any& metadata) = 0;
    virtual std::vector<std::string> supported_formats() const = 0;
};

class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void on_simulation_start(const SimulationContext& context) { (void)context; }
    virtual void on_simulation_end(const SimulationContext& context) { (void)context; }
    virtual void on_step_start(double time, size_t step) { (void)time; (void)step; }
    virtual void on_step_end(double time, size_t step) { (void)time; (void)step; }
    virtual void on_error(const std::string& error_message) { (void)error_message; }
    virtual void on_warning(const std::string& warning_message) { (void)warning_message; }
};

class IResourceManager {
public:
    virtual ~IResourceManager() = default;
    virtual void* allocate_gpu_memory(size_t size) = 0;
    virtual void* allocate_host_memory(size_t size) = 0;
    virtual void free_gpu_memory(void* ptr) = 0;
    virtual void free_host_memory(void* ptr) = 0;
    virtual size_t get_gpu_memory_usage() const = 0;
    virtual size_t get_host_memory_usage() const = 0;
    virtual bool has_sufficient_gpu_memory(size_t required) const = 0;
};

class IProfiler {
public:
    virtual ~IProfiler() = default;
    virtual void start_timer(const std::string& name) = 0;
    virtual void end_timer(const std::string& name) = 0;
    virtual double get_timer_value(const std::string& name) const = 0;
    virtual void reset_timers() = 0;
    virtual void print_summary() const = 0;
    virtual std::unordered_map<std::string, double> get_all_timers() const = 0;
};

// Plugin interface for dynamic loading
class IPlugin {
public:
    virtual ~IPlugin() = default;
    virtual bool initialize() = 0;
    virtual void finalize() = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::vector<std::string> get_component_types() const = 0;
    virtual std::unique_ptr<IComponent> create_component(const std::string& type,
                                                        const std::string& name) = 0;
};

// Factory function type for plugin creation
using PluginFactory = std::function<std::unique_ptr<IPlugin>()>;

// Component factory function type
template<typename T>
using ComponentFactory = std::function<std::unique_ptr<T>(const std::string&)>;

}