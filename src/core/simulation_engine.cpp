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

    // Initialize all registered components
    if (!context_->get_component_registry().initialize_all_components(*context_)) {
        std::cerr << "Failed to initialize components" << std::endl;
        return false;
    }

    // Get force computer from context configuration
    auto& config = context_->get_config();
    std::string force_type = config.get<std::string>("simulation.force_computer", "TreeForceComputer");

    // For now, create a default TreeForceComputer if not set
    // TODO: This will be replaced with proper component retrieval from registry
    if (!force_computer_) {
        std::cout << "No force computer registered, components initialization deferred" << std::endl;
    }

    std::cout << "Components initialized successfully" << std::endl;
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
    // Checkpoint based on step frequency if configured
    // For now, return false (checkpointing will be fully implemented with I/O system)
    // When checkpoint_frequency_ is set via set_checkpoint_frequency(), this will check against it
    return false;  // Will be implemented with CheckpointManager
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
    statistics_.num_particles = context_->get_num_particles();

    // Calculate elapsed time for current step
    auto current_time = std::chrono::steady_clock::now();
    auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        current_time - statistics_.current_step_start).count();

    if (step_duration > 0) {
        statistics_.steps_per_second = 1.0e6 / step_duration;
        statistics_.particles_per_second = (statistics_.num_particles * 1.0e6) / step_duration;
    }

    // Update total elapsed time
    if (statistics_.current_step > 0) {
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - statistics_.start_time).count();
        if (total_duration > 0) {
            statistics_.steps_per_second = static_cast<double>(statistics_.current_step) / total_duration;
        }
    }

    // Note: Detailed timing breakdowns (force_computation_time, integration_time, etc.)
    // will be implemented when IProfiler is integrated
}

void SimulationEngine::compute_forces() {
    if (force_computer_) {
        size_t num_particles = context_->get_num_particles();
        force_computer_->compute_forces(
            positions_.get(),
            masses_.get(),
            forces_.get(),
            num_particles
        );
    } else {
        // No force computer registered - forces remain zero
        // This is acceptable for initialization/testing
    }
}

void SimulationEngine::integrate_step() {
    if (integrator_) {
        size_t num_particles = context_->get_num_particles();
        integrator_->step(
            positions_.get(),
            velocities_.get(),
            forces_.get(),
            num_particles,
            time_step_
        );
    } else {
        // No integrator registered - use simple Euler integration as fallback
        size_t num_particles = context_->get_num_particles();
        for (size_t i = 0; i < num_particles; ++i) {
            // Update velocities: v = v + a*dt
            velocities_[i * 3 + 0] += forces_[i * 3 + 0] * time_step_;
            velocities_[i * 3 + 1] += forces_[i * 3 + 1] * time_step_;
            velocities_[i * 3 + 2] += forces_[i * 3 + 2] * time_step_;

            // Update positions: x = x + v*dt
            positions_[i * 3 + 0] += velocities_[i * 3 + 0] * time_step_;
            positions_[i * 3 + 1] += velocities_[i * 3 + 1] * time_step_;
            positions_[i * 3 + 2] += velocities_[i * 3 + 2] * time_step_;
        }
    }
}

void SimulationEngine::update_cosmology() {
    if (cosmology_model_) {
        // Update scale factor using cosmology model
        double scale_factor = context_->get_scale_factor();
        cosmology_model_->update_scale_factor(scale_factor, time_step_);
        context_->set_scale_factor(scale_factor);

        // Update statistics
        statistics_.scale_factor = scale_factor;
        statistics_.redshift = (1.0 / scale_factor) - 1.0;
    } else {
        // No cosmology model - this is acceptable for non-cosmological simulations
    }
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
    return compute_kinetic_energy() + compute_potential_energy();
}

double SimulationEngine::compute_kinetic_energy() const {
    if (!velocities_ || !masses_) {
        return 0.0;
    }

    double kinetic = 0.0;
    size_t num_particles = context_->get_num_particles();

    for (size_t i = 0; i < num_particles; ++i) {
        float vx = velocities_[i * 3 + 0];
        float vy = velocities_[i * 3 + 1];
        float vz = velocities_[i * 3 + 2];
        float mass = masses_[i];

        double v_squared = vx * vx + vy * vy + vz * vz;
        kinetic += 0.5 * mass * v_squared;
    }

    return kinetic;
}

double SimulationEngine::compute_potential_energy() const {
    if (!positions_ || !masses_) {
        return 0.0;
    }

    double potential = 0.0;
    size_t num_particles = context_->get_num_particles();
    float softening = 0.01f;  // Gravitational softening length

    // Compute pairwise potential energy
    for (size_t i = 0; i < num_particles; ++i) {
        float xi = positions_[i * 3 + 0];
        float yi = positions_[i * 3 + 1];
        float zi = positions_[i * 3 + 2];
        float mi = masses_[i];

        for (size_t j = i + 1; j < num_particles; ++j) {
            float xj = positions_[j * 3 + 0];
            float yj = positions_[j * 3 + 1];
            float zj = positions_[j * 3 + 2];
            float mj = masses_[j];

            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;

            float r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            float r = sqrtf(r2);

            // Gravitational potential: -G*m1*m2/r (G=1 in our units)
            potential -= mi * mj / r;
        }
    }

    return potential;
}

float3 SimulationEngine::compute_center_of_mass() const {
    if (!positions_ || !masses_) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float total_mass = 0.0f;
    float3 com = make_float3(0.0f, 0.0f, 0.0f);
    size_t num_particles = context_->get_num_particles();

    for (size_t i = 0; i < num_particles; ++i) {
        float mass = masses_[i];
        total_mass += mass;

        com.x += positions_[i * 3 + 0] * mass;
        com.y += positions_[i * 3 + 1] * mass;
        com.z += positions_[i * 3 + 2] * mass;
    }

    if (total_mass > 0.0f) {
        com.x /= total_mass;
        com.y /= total_mass;
        com.z /= total_mass;
    }

    return com;
}

float3 SimulationEngine::compute_angular_momentum() const {
    if (!positions_ || !velocities_ || !masses_) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // First compute center of mass
    float3 com = compute_center_of_mass();

    float3 L = make_float3(0.0f, 0.0f, 0.0f);
    size_t num_particles = context_->get_num_particles();

    for (size_t i = 0; i < num_particles; ++i) {
        float mass = masses_[i];

        // Position relative to center of mass
        float3 r;
        r.x = positions_[i * 3 + 0] - com.x;
        r.y = positions_[i * 3 + 1] - com.y;
        r.z = positions_[i * 3 + 2] - com.z;

        // Velocity
        float3 v;
        v.x = velocities_[i * 3 + 0];
        v.y = velocities_[i * 3 + 1];
        v.z = velocities_[i * 3 + 2];

        // Angular momentum L = r × (m*v)
        L.x += mass * (r.y * v.z - r.z * v.y);
        L.y += mass * (r.z * v.x - r.x * v.z);
        L.z += mass * (r.x * v.y - r.y * v.x);
    }

    return L;
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