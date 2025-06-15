#pragma once

#include "interfaces.hpp"
#include "configuration_manager.hpp"
#include "component_registry.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace core {

class SimulationContext {
private:
    std::unique_ptr<ConfigurationManager> config_manager_;
    std::unique_ptr<ComponentRegistry> component_registry_;
    std::unique_ptr<IResourceManager> resource_manager_;
    std::unique_ptr<IProfiler> profiler_;
    
    std::vector<std::unique_ptr<IObserver>> observers_;
    std::unordered_map<std::string, std::any> runtime_parameters_;
    
    // Simulation state
    size_t num_particles_;
    double current_time_;
    double scale_factor_;
    size_t current_step_;
    std::string simulation_id_;
    
    // GPU and MPI context
    int cuda_device_id_;
    int mpi_rank_;
    int mpi_size_;
    
public:
    SimulationContext();
    ~SimulationContext();
    
    // Initialization and cleanup
    bool initialize(const std::string& config_file);
    void finalize();
    
    // Component management
    template<typename T>
    std::shared_ptr<T> get_component(const std::string& name) const;
    
    template<typename T>
    std::vector<std::shared_ptr<T>> get_components_of_type() const;
    
    bool register_component(const std::string& name, std::unique_ptr<IComponent> component);
    bool unregister_component(const std::string& name);
    
    // Observer management
    void add_observer(std::unique_ptr<IObserver> observer);
    void remove_observer(IObserver* observer);
    void notify_simulation_start();
    void notify_simulation_end();
    void notify_step_start(double time, size_t step);
    void notify_step_end(double time, size_t step);
    void notify_error(const std::string& error);
    void notify_warning(const std::string& warning);
    
    // Configuration access
    ConfigurationManager& get_config() { return *config_manager_; }
    const ConfigurationManager& get_config() const { return *config_manager_; }
    
    // Resource management
    IResourceManager& get_resource_manager() { return *resource_manager_; }
    const IResourceManager& get_resource_manager() const { return *resource_manager_; }
    
    // Profiling
    IProfiler& get_profiler() { return *profiler_; }
    const IProfiler& get_profiler() const { return *profiler_; }
    
    // Runtime parameters
    template<typename T>
    void set_parameter(const std::string& key, const T& value);
    
    template<typename T>
    T get_parameter(const std::string& key, const T& default_value = T{}) const;
    
    bool has_parameter(const std::string& key) const;
    void remove_parameter(const std::string& key);
    
    // Simulation state accessors
    size_t get_num_particles() const { return num_particles_; }
    void set_num_particles(size_t num) { num_particles_ = num; }
    
    double get_current_time() const { return current_time_; }
    void set_current_time(double time) { current_time_ = time; }
    
    double get_scale_factor() const { return scale_factor_; }
    void set_scale_factor(double scale) { scale_factor_ = scale; }
    
    size_t get_current_step() const { return current_step_; }
    void set_current_step(size_t step) { current_step_ = step; }
    
    const std::string& get_simulation_id() const { return simulation_id_; }
    void set_simulation_id(const std::string& id) { simulation_id_ = id; }
    
    // GPU/MPI context
    int get_cuda_device_id() const { return cuda_device_id_; }
    void set_cuda_device_id(int device_id) { cuda_device_id_ = device_id; }
    
    int get_mpi_rank() const { return mpi_rank_; }
    void set_mpi_rank(int rank) { mpi_rank_ = rank; }
    
    int get_mpi_size() const { return mpi_size_; }
    void set_mpi_size(int size) { mpi_size_ = size; }
    
    // Utility functions
    bool is_master_process() const { return mpi_rank_ == 0; }
    double get_redshift() const { return 1.0 / scale_factor_ - 1.0; }
    std::string get_timestamp() const;
    std::string get_output_directory() const;
    std::string generate_simulation_id() const;
    
    // Plugin management
    bool load_plugin(const std::string& plugin_path);
    bool unload_plugin(const std::string& plugin_name);
    std::vector<std::string> get_loaded_plugins() const;
};

// Template implementations
template<typename T>
std::shared_ptr<T> SimulationContext::get_component(const std::string& name) const {
    return component_registry_->get_component<T>(name);
}

template<typename T>
std::vector<std::shared_ptr<T>> SimulationContext::get_components_of_type() const {
    return component_registry_->get_components_of_type<T>();
}

template<typename T>
void SimulationContext::set_parameter(const std::string& key, const T& value) {
    runtime_parameters_[key] = value;
}

template<typename T>
T SimulationContext::get_parameter(const std::string& key, const T& default_value) const {
    auto it = runtime_parameters_.find(key);
    if (it != runtime_parameters_.end()) {
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return default_value;
        }
    }
    return default_value;
}

}