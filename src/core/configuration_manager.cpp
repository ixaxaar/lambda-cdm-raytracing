#include "core/configuration_manager.hpp"
#include <iostream>
#include <sstream>

namespace core {

ConfigurationManager::ConfigurationManager()
    : root_(std::make_unique<ConfigurationNode>("root")) {
}

ConfigurationManager::~ConfigurationManager() = default;

bool ConfigurationManager::load_from_file(const std::string& filename) {
    config_file_path_ = filename;

    // For now, just return true - actual implementation would parse JSON/YAML
    std::cout << "Loading configuration from: " << filename << std::endl;

    // Set up default configuration
    setup_default_simulation_config();

    return true;
}

bool ConfigurationManager::save_to_file(const std::string& filename) const {
    std::cout << "Saving configuration to: " << filename << std::endl;
    return true;
}

bool ConfigurationManager::load_from_string(const std::string& config_string) {
    std::cout << "Loading configuration from string" << std::endl;
    return true;
}

std::string ConfigurationManager::save_to_string() const {
    return "{}"; // Placeholder
}

bool ConfigurationManager::has(const std::string& path) const {
    return root_->has_path(path);
}

void ConfigurationManager::setup_default_simulation_config() {
    // Set default simulation parameters
    root_->set_path("simulation.name", std::string("Lambda-CDM Simulation"));
    root_->set_path("simulation.output_directory", std::string("output"));
    root_->set_path("simulation.checkpoint_frequency", 100);

    // Default physics parameters
    root_->set_path("physics.cosmology.omega_m", 0.31);
    root_->set_path("physics.cosmology.omega_lambda", 0.69);
    root_->set_path("physics.cosmology.h", 0.67);

    // Default particle parameters
    root_->set_path("particles.num_particles", 10000);
    root_->set_path("particles.box_size", 100.0);

    // Default time parameters
    root_->set_path("time.initial_time", 0.0);
    root_->set_path("time.final_time", 10.0);
    root_->set_path("time.initial_timestep", 0.01);
}

void ConfigurationManager::setup_default_physics_config() {
    // Physics configuration defaults
}

void ConfigurationManager::setup_default_compute_config() {
    // Compute configuration defaults
}

void ConfigurationManager::setup_default_io_config() {
    // I/O configuration defaults
}

void ConfigurationManager::print_configuration() const {
    std::cout << "Configuration Summary:" << std::endl;
    std::cout << "=====================" << std::endl;

    if (root_->has_path("simulation.name")) {
        std::cout << "Simulation: " << root_->get_path<std::string>("simulation.name") << std::endl;
    }

    if (root_->has_path("particles.num_particles")) {
        std::cout << "Particles: " << root_->get_path<int>("particles.num_particles") << std::endl;
    }

    if (root_->has_path("particles.box_size")) {
        std::cout << "Box size: " << root_->get_path<double>("particles.box_size") << std::endl;
    }
}

// ConfigurationNode implementations
bool ConfigurationNode::has(const std::string& key) const {
    return values_.find(key) != values_.end();
}

void ConfigurationNode::remove(const std::string& key) {
    values_.erase(key);
}

ConfigurationNode* ConfigurationNode::get_child(const std::string& name) {
    auto it = children_.find(name);
    return (it != children_.end()) ? it->second.get() : nullptr;
}

const ConfigurationNode* ConfigurationNode::get_child(const std::string& name) const {
    auto it = children_.find(name);
    return (it != children_.end()) ? it->second.get() : nullptr;
}

ConfigurationNode* ConfigurationNode::create_child(const std::string& name) {
    children_[name] = std::make_unique<ConfigurationNode>(name);
    return children_[name].get();
}

void ConfigurationNode::remove_child(const std::string& name) {
    children_.erase(name);
}

bool ConfigurationNode::has_path(const std::string& path) const {
    // Simple implementation - in production this would handle nested paths
    return has(path);
}

std::vector<std::string> ConfigurationNode::get_keys() const {
    std::vector<std::string> keys;
    for (const auto& pair : values_) {
        keys.push_back(pair.first);
    }
    return keys;
}

std::vector<std::string> ConfigurationNode::get_child_names() const {
    std::vector<std::string> names;
    for (const auto& pair : children_) {
        names.push_back(pair.first);
    }
    return names;
}

}