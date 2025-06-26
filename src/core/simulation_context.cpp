#include "core/simulation_context.hpp"
#include "core/resource_manager.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace core {

SimulationContext::SimulationContext()
    : config_manager_(std::make_unique<ConfigurationManager>()),
      component_registry_(std::make_unique<ComponentRegistry>()),
      num_particles_(0),
      current_time_(0.0),
      scale_factor_(1.0),
      current_step_(0),
      simulation_id_(""),
      cuda_device_id_(0),
      mpi_rank_(0),
      mpi_size_(1) {

#ifdef HAVE_CUDA
    // TODO: Initialize GPU resource manager when implementation is ready
    // resource_manager_ = std::make_unique<GPUResourceManager>();
    resource_manager_ = nullptr;
#else
    // TODO: CPU-only resource manager
    resource_manager_ = nullptr;
#endif

    // TODO: Initialize profiler
    profiler_ = nullptr;
}

SimulationContext::~SimulationContext() {
    finalize();
}

bool SimulationContext::initialize(const std::string& config_file) {
    try {
        // Load configuration
        if (!config_manager_->load_from_file(config_file)) {
            return false;
        }

        // Initialize resource manager
        if (resource_manager_) {
            // TODO: Initialize with configuration parameters
        }

        // Generate simulation ID
        simulation_id_ = generate_simulation_id();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize simulation context: " << e.what() << std::endl;
        return false;
    }
}

void SimulationContext::finalize() {
    // Notify observers
    notify_simulation_end();

    // Clean up components
    if (component_registry_) {
        component_registry_->finalize_all_components();
    }

    // Clean up resources
    if (resource_manager_) {
        // TODO: Cleanup resource manager
    }
}

void SimulationContext::add_observer(std::unique_ptr<IObserver> observer) {
    observers_.push_back(std::move(observer));
}

void SimulationContext::remove_observer(IObserver* observer) {
    observers_.erase(
        std::remove_if(observers_.begin(), observers_.end(),
                      [observer](const std::unique_ptr<IObserver>& ptr) {
                          return ptr.get() == observer;
                      }),
        observers_.end());
}

void SimulationContext::notify_simulation_start() {
    for (auto& observer : observers_) {
        observer->on_simulation_start(*this);
    }
}

void SimulationContext::notify_simulation_end() {
    for (auto& observer : observers_) {
        observer->on_simulation_end(*this);
    }
}

void SimulationContext::notify_step_start(double time, size_t step) {
    for (auto& observer : observers_) {
        observer->on_step_start(time, step);
    }
}

void SimulationContext::notify_step_end(double time, size_t step) {
    for (auto& observer : observers_) {
        observer->on_step_end(time, step);
    }
}

void SimulationContext::notify_error(const std::string& error) {
    for (auto& observer : observers_) {
        observer->on_error(error);
    }
}

void SimulationContext::notify_warning(const std::string& warning) {
    for (auto& observer : observers_) {
        observer->on_warning(warning);
    }
}

bool SimulationContext::has_parameter(const std::string& key) const {
    return runtime_parameters_.find(key) != runtime_parameters_.end();
}

void SimulationContext::remove_parameter(const std::string& key) {
    runtime_parameters_.erase(key);
}

std::string SimulationContext::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string SimulationContext::get_output_directory() const {
    if (config_manager_) {
        return config_manager_->get("simulation.output_directory", std::string("output"));
    }
    return "output";
}

std::string SimulationContext::generate_simulation_id() const {
    std::stringstream ss;
    ss << "lambda_cdm_" << get_timestamp() << "_" << num_particles_;
    return ss.str();
}

bool SimulationContext::register_component(const std::string& name, std::unique_ptr<IComponent> component) {
    if (component_registry_) {
        return component_registry_->register_component(name, std::move(component));
    }
    return false;
}

bool SimulationContext::unregister_component(const std::string& name) {
    if (component_registry_) {
        return component_registry_->unregister_component(name);
    }
    return false;
}

bool SimulationContext::load_plugin(const std::string& plugin_path) {
    if (component_registry_) {
        return component_registry_->load_plugin(plugin_path);
    }
    return false;
}

bool SimulationContext::unload_plugin(const std::string& plugin_name) {
    if (component_registry_) {
        return component_registry_->unload_plugin(plugin_name);
    }
    return false;
}

std::vector<std::string> SimulationContext::get_loaded_plugins() const {
    if (component_registry_) {
        return component_registry_->get_loaded_plugins();
    }
    return {};
}

}