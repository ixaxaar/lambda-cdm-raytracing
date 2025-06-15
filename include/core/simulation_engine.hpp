#pragma once

#include "interfaces.hpp"
#include "simulation_context.hpp"
#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace core {

enum class SimulationState {
    UNINITIALIZED,
    INITIALIZED,
    RUNNING,
    PAUSED,
    FINISHED,
    ERROR
};

struct SimulationStatistics {
    size_t total_steps;
    double total_time;
    double current_time;
    double scale_factor;
    double redshift;
    size_t num_particles;
    
    // Performance metrics
    double steps_per_second;
    double particles_per_second;
    double force_computation_time;
    double integration_time;
    double communication_time;
    double io_time;
    
    // Memory usage
    size_t gpu_memory_usage;
    size_t host_memory_usage;
    
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point current_step_start;
};

class SimulationEngine {
private:
    std::unique_ptr<SimulationContext> context_;
    SimulationState state_;
    SimulationStatistics statistics_;
    
    // Core components
    std::shared_ptr<IForceComputer> force_computer_;
    std::shared_ptr<IIntegrator> integrator_;
    std::shared_ptr<ICosmologyModel> cosmology_model_;
    std::vector<std::shared_ptr<IDataExporter>> exporters_;
    
    // Simulation data
    std::unique_ptr<float[]> positions_;
    std::unique_ptr<float[]> velocities_;
    std::unique_ptr<float[]> masses_;
    std::unique_ptr<float[]> forces_;
    
    // Configuration
    double time_step_;
    double max_time_;
    size_t max_steps_;
    size_t output_frequency_;
    std::string output_directory_;
    
    // State management
    bool should_stop_;
    bool is_initialized_;
    
public:
    SimulationEngine();
    ~SimulationEngine();
    
    // Initialization and cleanup
    bool initialize(const std::string& config_file);
    bool initialize(std::unique_ptr<SimulationContext> context);
    void finalize();
    
    // Simulation control
    bool run();
    bool run_steps(size_t num_steps);
    bool run_until_time(double target_time);
    
    void pause();
    void resume();
    void stop();
    bool reset();
    
    // Single step execution
    bool step();
    
    // Configuration
    void set_time_step(double dt) { time_step_ = dt; }
    void set_max_time(double max_time) { max_time_ = max_time; }
    void set_max_steps(size_t max_steps) { max_steps_ = max_steps; }
    void set_output_frequency(size_t frequency) { output_frequency_ = frequency; }
    void set_output_directory(const std::string& dir) { output_directory_ = dir; }
    
    // Component management
    void set_force_computer(std::shared_ptr<IForceComputer> computer) { force_computer_ = computer; }
    void set_integrator(std::shared_ptr<IIntegrator> integrator) { integrator_ = integrator; }
    void set_cosmology_model(std::shared_ptr<ICosmologyModel> model) { cosmology_model_ = model; }
    void add_data_exporter(std::shared_ptr<IDataExporter> exporter) { exporters_.push_back(exporter); }
    
    // State access
    SimulationState get_state() const { return state_; }
    const SimulationStatistics& get_statistics() const { return statistics_; }
    SimulationContext& get_context() { return *context_; }
    const SimulationContext& get_context() const { return *context_; }
    
    // Data access
    const float* get_positions() const { return positions_.get(); }
    const float* get_velocities() const { return velocities_.get(); }
    const float* get_masses() const { return masses_.get(); }
    const float* get_forces() const { return forces_.get(); }
    
    // Snapshot management
    bool save_snapshot(const std::string& filename) const;
    bool load_snapshot(const std::string& filename);
    bool export_data(const std::string& filename) const;
    
    // Checkpoint system
    bool create_checkpoint(const std::string& checkpoint_dir) const;
    bool restore_from_checkpoint(const std::string& checkpoint_dir);
    void set_checkpoint_frequency(size_t frequency);
    
    // Analysis and diagnostics
    double compute_total_energy() const;
    double compute_kinetic_energy() const;
    double compute_potential_energy() const;
    float3 compute_center_of_mass() const;
    float3 compute_angular_momentum() const;
    
    // Performance monitoring
    void update_performance_statistics();
    void print_performance_summary() const;
    void reset_performance_counters();
    
    // Event handling
    void on_simulation_start();
    void on_simulation_end();
    void on_step_start();
    void on_step_end();
    void on_checkpoint_created(const std::string& checkpoint_path);
    void on_data_exported(const std::string& filename);
    
private:
    // Initialization helpers
    bool initialize_components();
    bool initialize_simulation_data();
    bool validate_configuration();
    
    // Main simulation loop helpers
    bool should_continue() const;
    bool should_output() const;
    bool should_checkpoint() const;
    
    void update_simulation_state();
    void update_statistics();
    
    // Force computation and integration
    void compute_forces();
    void integrate_step();
    void update_cosmology();
    
    // I/O operations
    bool output_snapshot();
    bool create_periodic_checkpoint();
    
    // Error handling
    void handle_error(const std::string& error_message);
    void set_error_state(const std::string& error_message);
    
    // Utility methods
    std::string generate_output_filename() const;
    std::string generate_checkpoint_name() const;
    void log_step_info() const;
    
    // Memory management
    bool allocate_simulation_arrays();
    void deallocate_simulation_arrays();
    
    // Validation
    bool validate_particle_data() const;
    bool validate_forces() const;
    void check_numerical_stability() const;
};

// Utility class for simulation configuration
class SimulationBuilder {
private:
    std::unique_ptr<SimulationContext> context_;
    std::string config_file_;
    
public:
    SimulationBuilder();
    
    SimulationBuilder& with_config_file(const std::string& config_file);
    SimulationBuilder& with_num_particles(size_t num_particles);
    SimulationBuilder& with_box_size(float box_size);
    SimulationBuilder& with_time_step(double dt);
    SimulationBuilder& with_max_time(double max_time);
    SimulationBuilder& with_output_directory(const std::string& dir);
    
    SimulationBuilder& with_force_computer(const std::string& type);
    SimulationBuilder& with_integrator(const std::string& type);
    SimulationBuilder& with_cosmology_model(const std::string& type);
    
    SimulationBuilder& enable_gpu(int device_id = 0);
    SimulationBuilder& enable_mpi();
    SimulationBuilder& enable_tensorrt(const std::string& engine_path = "");
    
    std::unique_ptr<SimulationEngine> build();
};

}