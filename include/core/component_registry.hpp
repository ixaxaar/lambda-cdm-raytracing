#pragma once

#include "interfaces.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <typeindex>
#include <functional>
#include <mutex>

namespace core {

class ComponentRegistry {
private:
    // Component storage by name
    std::unordered_map<std::string, std::shared_ptr<IComponent>> components_;
    
    // Component storage by type
    std::unordered_map<std::type_index, std::vector<std::shared_ptr<IComponent>>> components_by_type_;
    
    // Factory functions for component creation
    std::unordered_map<std::string, std::function<std::unique_ptr<IComponent>(const std::string&)>> factories_;
    
    // Plugin management
    std::unordered_map<std::string, std::unique_ptr<IPlugin>> plugins_;
    std::unordered_map<std::string, void*> plugin_handles_; // For dynamic loading
    
    // Thread safety
    mutable std::mutex registry_mutex_;
    
    // Component dependencies
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    std::unordered_map<std::string, std::vector<std::string>> dependents_;
    
public:
    ComponentRegistry();
    ~ComponentRegistry();
    
    // Component registration
    template<typename T>
    bool register_component(const std::string& name, std::unique_ptr<T> component);
    
    bool register_component(const std::string& name, std::shared_ptr<IComponent> component);
    bool unregister_component(const std::string& name);
    
    // Component retrieval
    template<typename T>
    std::shared_ptr<T> get_component(const std::string& name) const;
    
    std::shared_ptr<IComponent> get_component(const std::string& name) const;
    
    template<typename T>
    std::vector<std::shared_ptr<T>> get_components_of_type() const;
    
    // Component existence checks
    bool has_component(const std::string& name) const;
    template<typename T>
    bool has_component_of_type() const;
    
    // Factory registration
    template<typename T>
    void register_factory(const std::string& type_name, 
                         std::function<std::unique_ptr<T>(const std::string&)> factory);
    
    // Component creation from factories
    template<typename T>
    std::unique_ptr<T> create_component(const std::string& type_name, const std::string& instance_name);
    
    // Plugin management
    bool load_plugin(const std::string& plugin_path);
    bool unload_plugin(const std::string& plugin_name);
    bool register_plugin(const std::string& name, std::unique_ptr<IPlugin> plugin);
    std::vector<std::string> get_loaded_plugins() const;
    
    // Dependency management
    void add_dependency(const std::string& component, const std::string& dependency);
    void remove_dependency(const std::string& component, const std::string& dependency);
    std::vector<std::string> get_dependencies(const std::string& component) const;
    std::vector<std::string> get_dependents(const std::string& component) const;
    std::vector<std::string> get_initialization_order() const;
    
    // Component lifecycle
    bool initialize_all_components(const SimulationContext& context);
    void finalize_all_components();
    bool initialize_component(const std::string& name, const SimulationContext& context);
    void finalize_component(const std::string& name);
    
    // Query and introspection
    std::vector<std::string> get_all_component_names() const;
    std::vector<std::string> get_component_names_by_type(const std::type_index& type) const;
    
    template<typename T>
    std::vector<std::string> get_component_names_by_type() const;
    
    std::string get_component_info(const std::string& name) const;
    
    // Configuration and validation
    bool validate_dependencies() const;
    void print_component_graph() const;
    void print_registry_status() const;
    
    // Bulk operations
    template<typename T>
    void initialize_components_of_type(const SimulationContext& context);
    
    template<typename T>
    void finalize_components_of_type();
    
private:
    // Helper methods
    bool has_circular_dependencies() const;
    void topological_sort(std::vector<std::string>& result) const;
    void* load_dynamic_library(const std::string& path);
    void unload_dynamic_library(void* handle);
    std::function<std::unique_ptr<IPlugin>()> get_plugin_factory(void* handle);
};

// Template implementations
template<typename T>
bool ComponentRegistry::register_component(const std::string& name, std::unique_ptr<T> component) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    if (components_.find(name) != components_.end()) {
        return false; // Component already exists
    }
    
    auto shared_component = std::shared_ptr<T>(component.release());
    components_[name] = shared_component;
    components_by_type_[std::type_index(typeid(T))].push_back(shared_component);
    
    return true;
}

template<typename T>
std::shared_ptr<T> ComponentRegistry::get_component(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = components_.find(name);
    if (it != components_.end()) {
        return std::dynamic_pointer_cast<T>(it->second);
    }
    return nullptr;
}

template<typename T>
std::vector<std::shared_ptr<T>> ComponentRegistry::get_components_of_type() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::shared_ptr<T>> result;
    auto it = components_by_type_.find(std::type_index(typeid(T)));
    
    if (it != components_by_type_.end()) {
        for (const auto& component : it->second) {
            if (auto typed_component = std::dynamic_pointer_cast<T>(component)) {
                result.push_back(typed_component);
            }
        }
    }
    
    return result;
}

template<typename T>
bool ComponentRegistry::has_component_of_type() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = components_by_type_.find(std::type_index(typeid(T)));
    return it != components_by_type_.end() && !it->second.empty();
}

template<typename T>
void ComponentRegistry::register_factory(const std::string& type_name,
                                        std::function<std::unique_ptr<T>(const std::string&)> factory) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    factories_[type_name] = [factory](const std::string& name) -> std::unique_ptr<IComponent> {
        return std::unique_ptr<IComponent>(factory(name).release());
    };
}

template<typename T>
std::unique_ptr<T> ComponentRegistry::create_component(const std::string& type_name, 
                                                      const std::string& instance_name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = factories_.find(type_name);
    if (it != factories_.end()) {
        auto component = it->second(instance_name);
        return std::unique_ptr<T>(dynamic_cast<T*>(component.release()));
    }
    return nullptr;
}

template<typename T>
std::vector<std::string> ComponentRegistry::get_component_names_by_type() const {
    return get_component_names_by_type(std::type_index(typeid(T)));
}

template<typename T>
void ComponentRegistry::initialize_components_of_type(const SimulationContext& context) {
    auto components = get_components_of_type<T>();
    for (auto& component : components) {
        component->initialize(context);
    }
}

template<typename T>
void ComponentRegistry::finalize_components_of_type() {
    auto components = get_components_of_type<T>();
    for (auto& component : components) {
        component->finalize();
    }
}

}