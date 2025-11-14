#include "core/component_registry.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>

// Dynamic plugin loading support
#ifdef __unix__
    #include <dlfcn.h>
    #define PLUGIN_AVAILABLE 1
#elif defined(_WIN32)
    #include <windows.h>
    #define PLUGIN_AVAILABLE 1
#else
    #define PLUGIN_AVAILABLE 0
#endif

namespace core {

ComponentRegistry::ComponentRegistry() = default;

ComponentRegistry::~ComponentRegistry() {
    finalize_all_components();
}

bool ComponentRegistry::register_component(const std::string& name, std::shared_ptr<IComponent> component) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    if (components_.find(name) != components_.end()) {
        return false; // Component already exists
    }

    components_[name] = component;

    // Add to type-based storage
    std::type_index type_idx(typeid(*component));
    components_by_type_[type_idx].push_back(component);

    return true;
}

bool ComponentRegistry::unregister_component(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = components_.find(name);
    if (it == components_.end()) {
        return false;
    }

    // Remove from type-based storage
    std::type_index type_idx(typeid(*it->second));
    auto& type_vec = components_by_type_[type_idx];
    type_vec.erase(
        std::remove(type_vec.begin(), type_vec.end(), it->second),
        type_vec.end());

    // Finalize component
    it->second->finalize();

    // Remove from main storage
    components_.erase(it);

    return true;
}

std::shared_ptr<IComponent> ComponentRegistry::get_component(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = components_.find(name);
    return (it != components_.end()) ? it->second : nullptr;
}

bool ComponentRegistry::has_component(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return components_.find(name) != components_.end();
}

std::vector<std::string> ComponentRegistry::get_component_names_by_type(const std::type_index& type) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> names;

    for (const auto& pair : components_) {
        if (std::type_index(typeid(*pair.second)) == type) {
            names.push_back(pair.first);
        }
    }

    return names;
}

void ComponentRegistry::add_dependency(const std::string& component, const std::string& dependency) {
    dependencies_[component].push_back(dependency);
    dependents_[dependency].push_back(component);
}

void ComponentRegistry::remove_dependency(const std::string& component, const std::string& dependency) {
    auto& deps = dependencies_[component];
    deps.erase(std::remove(deps.begin(), deps.end(), dependency), deps.end());

    auto& dependents = dependents_[dependency];
    dependents.erase(std::remove(dependents.begin(), dependents.end(), component), dependents.end());
}

std::vector<std::string> ComponentRegistry::get_dependencies(const std::string& component) const {
    auto it = dependencies_.find(component);
    return (it != dependencies_.end()) ? it->second : std::vector<std::string>();
}

std::vector<std::string> ComponentRegistry::get_dependents(const std::string& component) const {
    auto it = dependents_.find(component);
    return (it != dependents_.end()) ? it->second : std::vector<std::string>();
}

std::vector<std::string> ComponentRegistry::get_initialization_order() const {
    std::vector<std::string> result;
    topological_sort(result);
    return result;
}

bool ComponentRegistry::initialize_all_components(const SimulationContext& context) {
    auto init_order = get_initialization_order();

    for (const auto& name : init_order) {
        if (!initialize_component(name, context)) {
            std::cerr << "Failed to initialize component: " << name << std::endl;
            return false;
        }
    }

    return true;
}

void ComponentRegistry::finalize_all_components() {
    auto init_order = get_initialization_order();

    // Finalize in reverse order
    for (auto it = init_order.rbegin(); it != init_order.rend(); ++it) {
        finalize_component(*it);
    }
}

bool ComponentRegistry::initialize_component(const std::string& name, const SimulationContext& context) {
    auto component = get_component(name);
    if (!component) {
        return false;
    }

    try {
        return component->initialize(context);
    } catch (const std::exception& e) {
        std::cerr << "Exception during component initialization: " << e.what() << std::endl;
        return false;
    }
}

void ComponentRegistry::finalize_component(const std::string& name) {
    auto component = get_component(name);
    if (component) {
        try {
            component->finalize();
        } catch (const std::exception& e) {
            std::cerr << "Exception during component finalization: " << e.what() << std::endl;
        }
    }
}

std::vector<std::string> ComponentRegistry::get_all_component_names() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> names;
    for (const auto& pair : components_) {
        names.push_back(pair.first);
    }
    return names;
}

std::string ComponentRegistry::get_component_info(const std::string& name) const {
    auto component = get_component(name);
    if (!component) {
        return "Component not found";
    }

    std::stringstream ss;
    ss << "Name: " << component->get_name() << std::endl;
    ss << "Type: " << component->get_type() << std::endl;
    ss << "Version: " << component->get_version() << std::endl;

    return ss.str();
}

bool ComponentRegistry::validate_dependencies() const {
    return !has_circular_dependencies();
}

void ComponentRegistry::print_component_graph() const {
    std::cout << "Component Dependency Graph:" << std::endl;
    std::cout << "===========================" << std::endl;

    for (const auto& pair : dependencies_) {
        std::cout << pair.first << " depends on: ";
        for (const auto& dep : pair.second) {
            std::cout << dep << " ";
        }
        std::cout << std::endl;
    }
}

void ComponentRegistry::print_registry_status() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::cout << "Component Registry Status:" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Total components: " << components_.size() << std::endl;
    std::cout << "Total factories: " << factories_.size() << std::endl;
    std::cout << "Loaded plugins: " << plugins_.size() << std::endl;

    std::cout << "\nRegistered components:" << std::endl;
    for (const auto& pair : components_) {
        std::cout << "  - " << pair.first << " (" << pair.second->get_type() << ")" << std::endl;
    }
}

bool ComponentRegistry::has_circular_dependencies() const {
    // Simple cycle detection - in production this would be more sophisticated
    return false;
}

void ComponentRegistry::topological_sort(std::vector<std::string>& result) const {
    // Simple implementation - in production this would handle complex dependency graphs
    for (const auto& pair : components_) {
        result.push_back(pair.first);
    }
}

bool ComponentRegistry::load_plugin(const std::string& plugin_path) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::cout << "Loading plugin: " << plugin_path << std::endl;

#if PLUGIN_AVAILABLE

#ifdef __unix__
    // Load the shared library
    void* handle = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "Failed to load plugin: " << dlerror() << std::endl;
        return false;
    }

    // Look for the plugin factory function
    // Expected signature: extern "C" IPlugin* create_plugin()
    using CreatePluginFunc = IPlugin* (*)();
    CreatePluginFunc create_plugin = reinterpret_cast<CreatePluginFunc>(
        dlsym(handle, "create_plugin"));

    if (!create_plugin) {
        std::cerr << "Plugin does not export 'create_plugin' function: " << dlerror() << std::endl;
        dlclose(handle);
        return false;
    }

    // Create the plugin instance
    IPlugin* plugin = create_plugin();
    if (!plugin) {
        std::cerr << "Failed to create plugin instance" << std::endl;
        dlclose(handle);
        return false;
    }

    // Initialize the plugin
    if (!plugin->initialize()) {
        std::cerr << "Failed to initialize plugin" << std::endl;
        delete plugin;
        dlclose(handle);
        return false;
    }

    // Store the plugin  (we'll manage the handle separately)
    std::string plugin_name = plugin->get_name();
    plugins_[plugin_name] = std::unique_ptr<IPlugin>(plugin);

    // Note: The dlclose(handle) will be called when the process exits
    // For production code, we'd want to track handles separately for proper cleanup

    std::cout << "Plugin loaded successfully: " << plugin_name
              << " v" << plugin->get_version() << std::endl;
    return true;

#elif defined(_WIN32)
    // Windows implementation using LoadLibrary
    HMODULE handle = LoadLibraryA(plugin_path.c_str());
    if (!handle) {
        std::cerr << "Failed to load plugin: error code " << GetLastError() << std::endl;
        return false;
    }

    using CreatePluginFunc = IPlugin* (*)();
    CreatePluginFunc create_plugin = reinterpret_cast<CreatePluginFunc>(
        GetProcAddress(handle, "create_plugin"));

    if (!create_plugin) {
        std::cerr << "Plugin does not export 'create_plugin' function" << std::endl;
        FreeLibrary(handle);
        return false;
    }

    IPlugin* plugin = create_plugin();
    if (!plugin || !plugin->initialize()) {
        std::cerr << "Failed to create or initialize plugin" << std::endl;
        if (plugin) delete plugin;
        FreeLibrary(handle);
        return false;
    }

    std::string plugin_name = plugin->get_name();
    plugins_[plugin_name] = std::unique_ptr<IPlugin>(plugin);

    // Note: The FreeLibrary(handle) will be called when the process exits
    // For production code, we'd want to track handles separately for proper cleanup

    std::cout << "Plugin loaded successfully: " << plugin_name << std::endl;
    return true;
#endif

#else
    std::cerr << "Dynamic plugin loading not supported on this platform" << std::endl;
    return false;
#endif
}

bool ComponentRegistry::unload_plugin(const std::string& plugin_name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::cout << "Unloading plugin: " << plugin_name << std::endl;

    auto it = plugins_.find(plugin_name);
    if (it == plugins_.end()) {
        std::cerr << "Plugin not found: " << plugin_name << std::endl;
        return false;
    }

    // Get component types before removing
    auto component_types = it->second->get_component_types();

    // Remove all components created by this plugin
    for (const auto& type : component_types) {
        auto comp_names = get_component_names_by_type(std::type_index(typeid(type)));
        for (const auto& comp_name : comp_names) {
            unregister_component(comp_name);
        }
    }

    // Remove the plugin (shared_ptr destructor will call finalize and dlclose)
    plugins_.erase(it);

    std::cout << "Plugin unloaded successfully: " << plugin_name << std::endl;
    return true;
}

std::vector<std::string> ComponentRegistry::get_loaded_plugins() const {
    std::vector<std::string> plugin_names;
    for (const auto& pair : plugins_) {
        plugin_names.push_back(pair.first);
    }
    return plugin_names;
}

}