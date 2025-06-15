#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>
#include "core/math_types.hpp"
#include "physics/cosmology_model.hpp"

namespace physics {

// Particle structure (kept for compatibility, but GPU uses SoA)
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    uint64_t id;
};

// Forward declaration to hide CUDA implementation details
class LambdaCDMSimulationImpl;

class LambdaCDMSimulation {
private:
    std::unique_ptr<LambdaCDMSimulationImpl> pImpl;
    
public:
    LambdaCDMSimulation(size_t num_particles, float box_size, 
                       const CosmologyParams& params = CosmologyParams());
    ~LambdaCDMSimulation();
    
    // Disable copy constructor and assignment
    LambdaCDMSimulation(const LambdaCDMSimulation&) = delete;
    LambdaCDMSimulation& operator=(const LambdaCDMSimulation&) = delete;
    
    // Enable move semantics
    LambdaCDMSimulation(LambdaCDMSimulation&&) noexcept;
    LambdaCDMSimulation& operator=(LambdaCDMSimulation&&) noexcept;
    
    // Initialization
    void initialize_particles();
    void set_initial_conditions_from_power_spectrum();
    void set_softening(float softening);
    
    // Simulation methods
    void step(double dt);
    void compute_forces();
    void compute_energy();
    void update_scale_factor(double dt);
    
    // Cosmology functions
    double hubble_function(double a) const;
    double growth_factor(double a) const;
    double power_spectrum(double k) const;
    
    // Data access (copy from GPU)
    void copy_particles_to_host(std::vector<Particle>& particles) const;
    void copy_positions_to_host(float* positions) const;
    void copy_velocities_to_host(float* velocities) const;
    
    // Accessors
    double get_scale_factor() const;
    double get_redshift() const;
    size_t get_num_particles() const;
    float get_box_size() const;
    float get_kinetic_energy() const;
    float get_potential_energy() const;
    float get_total_energy() const;
    size_t get_current_step() const;
    
    // Performance monitoring
    void enable_profiling();
    void disable_profiling();
    void print_performance_stats() const;
};

}