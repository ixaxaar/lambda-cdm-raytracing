#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <any>
#include <memory>
#include <functional>

namespace core {

class ConfigurationNode {
private:
    std::unordered_map<std::string, std::any> values_;
    std::unordered_map<std::string, std::unique_ptr<ConfigurationNode>> children_;
    std::string name_;
    
public:
    explicit ConfigurationNode(const std::string& name = "") : name_(name) {}
    
    // Value accessors with type safety
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const;
    
    template<typename T>
    void set(const std::string& key, const T& value);
    
    bool has(const std::string& key) const;
    void remove(const std::string& key);
    
    // Nested node access
    ConfigurationNode* get_child(const std::string& name);
    const ConfigurationNode* get_child(const std::string& name) const;
    ConfigurationNode* create_child(const std::string& name);
    void remove_child(const std::string& name);
    
    // Path-based access (e.g., "physics.cosmology.omega_m")
    template<typename T>
    T get_path(const std::string& path, const T& default_value = T{}) const;
    
    template<typename T>
    void set_path(const std::string& path, const T& value);
    
    bool has_path(const std::string& path) const;
    
    // Iteration
    std::vector<std::string> get_keys() const;
    std::vector<std::string> get_child_names() const;
    
    // Validation
    using Validator = std::function<bool(const std::any&)>;
    void add_validator(const std::string& key, Validator validator);
    bool validate() const;
    
    const std::string& get_name() const { return name_; }
};

class ConfigurationManager {
private:
    std::unique_ptr<ConfigurationNode> root_;
    std::string config_file_path_;
    std::unordered_map<std::string, std::string> environment_overrides_;
    std::unordered_map<std::string, std::string> command_line_overrides_;
    
    // Configuration schemas for validation
    std::unordered_map<std::string, std::function<void(ConfigurationNode*)>> schemas_;
    
public:
    ConfigurationManager();
    ~ConfigurationManager();
    
    // Loading and saving
    bool load_from_file(const std::string& filename);
    bool save_to_file(const std::string& filename) const;
    bool load_from_string(const std::string& config_string);
    std::string save_to_string() const;
    
    // Format support
    bool load_json(const std::string& filename);
    bool load_yaml(const std::string& filename);
    bool load_toml(const std::string& filename);
    bool save_json(const std::string& filename) const;
    bool save_yaml(const std::string& filename) const;
    
    // Root node access
    ConfigurationNode& get_root() { return *root_; }
    const ConfigurationNode& get_root() const { return *root_; }
    
    // Convenience accessors
    template<typename T>
    T get(const std::string& path, const T& default_value = T{}) const;
    
    template<typename T>
    void set(const std::string& path, const T& value);
    
    bool has(const std::string& path) const;
    
    // Override management
    void add_environment_override(const std::string& env_var, const std::string& config_path);
    void add_command_line_override(const std::string& arg, const std::string& config_path);
    void apply_overrides();
    
    // Schema registration and validation
    void register_schema(const std::string& name, std::function<void(ConfigurationNode*)> schema);
    bool validate_against_schema(const std::string& schema_name) const;
    bool validate_all() const;
    
    // Configuration merging
    void merge_from(const ConfigurationManager& other);
    void merge_from_file(const std::string& filename);
    
    // Default configurations
    void setup_default_simulation_config();
    void setup_default_physics_config();
    void setup_default_compute_config();
    void setup_default_io_config();
    
    // Utilities
    std::vector<std::string> get_all_paths() const;
    void print_configuration() const;
    void print_configuration_tree() const;
    
private:
    std::vector<std::string> split_path(const std::string& path) const;
    ConfigurationNode* navigate_to_path(const std::string& path, bool create_if_missing = false);
    const ConfigurationNode* navigate_to_path(const std::string& path) const;
    void apply_environment_overrides();
    void apply_command_line_overrides();
    void collect_all_paths(const ConfigurationNode* node, const std::string& prefix,
                          std::vector<std::string>& paths) const;
};

// Template implementations
template<typename T>
T ConfigurationNode::get(const std::string& key, const T& default_value) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return default_value;
        }
    }
    return default_value;
}

template<typename T>
void ConfigurationNode::set(const std::string& key, const T& value) {
    values_[key] = value;
}

template<typename T>
T ConfigurationNode::get_path(const std::string& path, const T& default_value) const {
    // Implementation would navigate through nested nodes
    // For brevity, simplified version
    return get<T>(path, default_value);
}

template<typename T>
void ConfigurationNode::set_path(const std::string& path, const T& value) {
    // Implementation would navigate/create nested nodes
    // For brevity, simplified version
    set<T>(path, value);
}

template<typename T>
T ConfigurationManager::get(const std::string& path, const T& default_value) const {
    return root_->get_path<T>(path, default_value);
}

template<typename T>
void ConfigurationManager::set(const std::string& path, const T& value) {
    root_->set_path<T>(path, value);
}

}