#include "core/simulation_engine.hpp"
#include <iostream>

namespace core {

SimulationEngine::SimulationEngine()
    : context_(nullptr),
      state_(SimulationState::UNINITIALIZED),
      time_step_(0.01),
      max_time_(10.0),
      max_steps_(1000000),
      output_frequency_(10),
      output_directory_("output"),
      should_stop_(false),
      is_initialized_(false) {
    
    // Initialize statistics
    statistics_ = {};
}

SimulationEngine::~SimulationEngine() {
    finalize();
}

bool SimulationEngine::initialize(const std::string& config_file) {
    try {
        context_ = std::make_unique<SimulationContext>();
        return initialize(std::move(context_));
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize simulation: " << e.what() << std::endl;
        return false;
    }
}

bool SimulationEngine::initialize(std::unique_ptr<SimulationContext> context) {
    if (!context) {
        return false;
    }
    
    context_ = std::move(context);
    
    try {
        // Initialize context
        if (!validate_configuration()) {
            state_ = SimulationState::ERROR;
            return false;
        }
        
        if (!initialize_components()) {
            state_ = SimulationState::ERROR;
            return false;
        }
        
        if (!initialize_simulation_data()) {
            state_ = SimulationState::ERROR;
            return false;
        }
        
        state_ = SimulationState::INITIALIZED;
        is_initialized_ = true;
        
        std::cout << "Simulation initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        state_ = SimulationState::ERROR;
        return false;
    }
}

void SimulationEngine::finalize() {
    if (context_) {
        context_->finalize();
    }
    
    deallocate_simulation_arrays();
    state_ = SimulationState::UNINITIALIZED;
    is_initialized_ = false;
}

bool SimulationEngine::run() {
    if (state_ != SimulationState::INITIALIZED) {
        std::cerr << "Simulation not properly initialized" << std::endl;
        return false;
    }
    
    state_ = SimulationState::RUNNING;
    should_stop_ = false;
    
    // Notify observers
    on_simulation_start();
    
    std::cout << "Starting simulation..." << std::endl;
    
    // Main simulation loop
    while (should_continue()) {
        if (!step()) {
            state_ = SimulationState::ERROR;
            break;
        }
        
        if (should_stop_) {
            break;
        }
    }
    
    // Finalize
    if (state_ == SimulationState::RUNNING) {
        state_ = SimulationState::FINISHED;
    }
    
    on_simulation_end();
    
    return state_ == SimulationState::FINISHED;
}

bool SimulationEngine::step() {
    if (state_ != SimulationState::RUNNING) {
        return false;
    }
    
    on_step_start();
    
    try {
        // Update simulation state
        update_simulation_state();
        
        // Perform simulation step
        compute_forces();
        integrate_step();
        update_cosmology();
        
        // Update statistics
        update_statistics();
        
        // Output if needed
        if (should_output()) {
            output_snapshot();
        }
        
        // Checkpoint if needed
        if (should_checkpoint()) {
            create_periodic_checkpoint();
        }
        
        on_step_end();
        return true;
        
    } catch (const std::exception& e) {
        handle_error("Exception during simulation step: " + std::string(e.what()));
        return false;
    }
}

void SimulationEngine::pause() {
    if (state_ == SimulationState::RUNNING) {
        state_ = SimulationState::PAUSED;
        std::cout << "Simulation paused" << std::endl;
    }
}

void SimulationEngine::resume() {
    if (state_ == SimulationState::PAUSED) {
        state_ = SimulationState::RUNNING;
        std::cout << "Simulation resumed" << std::endl;
    }
}

void SimulationEngine::stop() {
    should_stop_ = true;
    std::cout << "Simulation stop requested" << std::endl;
}

bool SimulationEngine::reset() {
    finalize();
    statistics_ = {};
    return true;
}

bool SimulationEngine::validate_configuration() {
    if (!context_) {
        return false;
    }
    
    // Basic validation
    auto& config = context_->get_config();
    
    // Check required parameters
    if (!config.has("particles.num_particles")) {
        std::cerr << "Missing required parameter: particles.num_particles" << std::endl;
        return false;
    }
    
    context_->set_num_particles(config.get<size_t>("particles.num_particles", 10000));
    
    std::cout << "Configuration validated" << std::endl;
    return true;
}

bool SimulationEngine::initialize_components() {
    if (!context_) {
        return false;
    }
    
    // TODO: Initialize all registered components
    // This would be implemented when component registry is fully integrated
    
    std::cout << "Components initialized" << std::endl;
    return true;
}

bool SimulationEngine::initialize_simulation_data() {
    if (!context_) {
        return false;
    }
    
    size_t num_particles = context_->get_num_particles();
    
    if (!allocate_simulation_arrays()) {
        return false;
    }
    
    // Initialize particles (placeholder)
    for (size_t i = 0; i < num_particles; ++i) {
        positions_[i * 3 + 0] = 0.0f;  // x
        positions_[i * 3 + 1] = 0.0f;  // y
        positions_[i * 3 + 2] = 0.0f;  // z
        
        velocities_[i * 3 + 0] = 0.0f;  // vx
        velocities_[i * 3 + 1] = 0.0f;  // vy
        velocities_[i * 3 + 2] = 0.0f;  // vz
        
        masses_[i] = 1.0f;
        
        forces_[i * 3 + 0] = 0.0f;  // fx
        forces_[i * 3 + 1] = 0.0f;  // fy
        forces_[i * 3 + 2] = 0.0f;  // fz
    }
    
    std::cout << "Simulation data initialized for " << num_particles << " particles" << std::endl;
    return true;
}

bool SimulationEngine::should_continue() const {
    return statistics_.current_step < max_steps_ && 
           statistics_.current_time < max_time_ &&
           !should_stop_;
}

bool SimulationEngine::should_output() const {
    return (statistics_.current_step % output_frequency_) == 0;
}

bool SimulationEngine::should_checkpoint() const {
    // TODO: Implement checkpoint logic
    return false;
}

void SimulationEngine::update_simulation_state() {
    statistics_.current_step++;
    statistics_.current_time += time_step_;
    context_->set_current_step(statistics_.current_step);
    context_->set_current_time(statistics_.current_time);
}

void SimulationEngine::update_statistics() {
    // Update performance statistics
    statistics_.total_steps = statistics_.current_step;
    statistics_.total_time = statistics_.current_time;
    
    // TODO: Update detailed performance metrics
}

void SimulationEngine::compute_forces() {
    // TODO: Use registered force computer
    // For now, just placeholder
}

void SimulationEngine::integrate_step() {
    // TODO: Use registered integrator
    // For now, just placeholder
}

void SimulationEngine::update_cosmology() {
    // TODO: Use registered cosmology model
    // For now, just placeholder
}

bool SimulationEngine::output_snapshot() {
    std::cout << "Output snapshot at step " << statistics_.current_step << std::endl;
    return true;
}

bool SimulationEngine::create_periodic_checkpoint() {
    std::cout << "Creating checkpoint at step " << statistics_.current_step << std::endl;
    return true;
}

void SimulationEngine::on_simulation_start() {
    statistics_.start_time = std::chrono::steady_clock::now();
    context_->notify_simulation_start();
}

void SimulationEngine::on_simulation_end() {
    context_->notify_simulation_end();
}

void SimulationEngine::on_step_start() {
    statistics_.current_step_start = std::chrono::steady_clock::now();
    context_->notify_step_start(statistics_.current_time, statistics_.current_step);
}

void SimulationEngine::on_step_end() {
    context_->notify_step_end(statistics_.current_time, statistics_.current_step);
}

void SimulationEngine::handle_error(const std::string& error_message) {
    std::cerr << "Simulation error: " << error_message << std::endl;
    context_->notify_error(error_message);
    state_ = SimulationState::ERROR;
}

bool SimulationEngine::allocate_simulation_arrays() {
    size_t num_particles = context_->get_num_particles();
    
    try {
        positions_ = std::make_unique<float[]>(num_particles * 3);
        velocities_ = std::make_unique<float[]>(num_particles * 3);
        masses_ = std::make_unique<float[]>(num_particles);
        forces_ = std::make_unique<float[]>(num_particles * 3);
        
        return true;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate simulation arrays: " << e.what() << std::endl;
        return false;
    }
}

void SimulationEngine::deallocate_simulation_arrays() {
    positions_.reset();
    velocities_.reset();
    masses_.reset();
    forces_.reset();
}

void SimulationEngine::print_performance_summary() const {
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Total steps: " << statistics_.total_steps << std::endl;
    std::cout << "Total time: " << statistics_.total_time << std::endl;
    std::cout << "Particles: " << context_->get_num_particles() << std::endl;
}

double SimulationEngine::compute_total_energy() const {
    // TODO: Implement energy computation
    return 0.0;
}

double SimulationEngine::compute_kinetic_energy() const {
    // TODO: Implement kinetic energy computation
    return 0.0;
}

double SimulationEngine::compute_potential_energy() const {
    // TODO: Implement potential energy computation
    return 0.0;
}

float3 SimulationEngine::compute_center_of_mass() const {
    // TODO: Implement center of mass computation
    return make_float3(0.0f, 0.0f, 0.0f);
}

float3 SimulationEngine::compute_angular_momentum() const {
    // TODO: Implement angular momentum computation
    return make_float3(0.0f, 0.0f, 0.0f);
}

// SimulationBuilder implementation
SimulationBuilder::SimulationBuilder() 
    : context_(std::make_unique<SimulationContext>()) {
}

SimulationBuilder& SimulationBuilder::with_config_file(const std::string& config_file) {
    config_file_ = config_file;
    return *this;
}

SimulationBuilder& SimulationBuilder::with_num_particles(size_t num_particles) {
    context_->set_num_particles(num_particles);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_box_size(float box_size) {
    context_->set_parameter<float>("box_size", box_size);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_time_step(double dt) {
    context_->set_parameter<double>("time_step", dt);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_max_time(double max_time) {
    context_->set_parameter<double>("max_time", max_time);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_output_directory(const std::string& dir) {
    context_->set_parameter<std::string>("output_directory", dir);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_force_computer(const std::string& type) {
    context_->set_parameter<std::string>("force_computer_type", type);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_integrator(const std::string& type) {
    context_->set_parameter<std::string>("integrator_type", type);
    return *this;
}

SimulationBuilder& SimulationBuilder::with_cosmology_model(const std::string& type) {
    context_->set_parameter<std::string>("cosmology_model_type", type);
    return *this;
}

SimulationBuilder& SimulationBuilder::enable_gpu(int device_id) {
    context_->set_parameter<int>("gpu_device_id", device_id);
    context_->set_parameter<bool>("use_gpu", true);
    return *this;
}

SimulationBuilder& SimulationBuilder::enable_mpi() {
    context_->set_parameter<bool>("use_mpi", true);
    return *this;
}

SimulationBuilder& SimulationBuilder::enable_tensorrt(const std::string& engine_path) {
    context_->set_parameter<std::string>("tensorrt_engine_path", engine_path);
    context_->set_parameter<bool>("use_tensorrt", true);
    return *this;
}

std::unique_ptr<SimulationEngine> SimulationBuilder::build() {
    auto engine = std::make_unique<SimulationEngine>();
    
    if (!config_file_.empty()) {
        context_->initialize(config_file_);
    }
    
    if (engine->initialize(std::move(context_))) {
        return engine;
    }
    
    return nullptr;
}

}