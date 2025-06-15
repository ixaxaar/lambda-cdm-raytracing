#include "physics/initial_conditions.hpp"
#include "physics/cosmology_model.hpp"
#include <iostream>
#include <chrono>

using namespace physics;

int main() {
    std::cout << "=== Zel'dovich Approximation Test ===" << std::endl;
    
    // Set up cosmology
    CosmologyParams cosmo_params;
    cosmo_params.omega_m = 0.31;
    cosmo_params.omega_lambda = 0.69;
    cosmo_params.h = 0.67;
    cosmo_params.sigma_8 = 0.81;
    cosmo_params.n_s = 0.965;
    
    CosmologyModel cosmology(cosmo_params);
    
    // Set up initial conditions parameters
    InitialConditionsParams ic_params;
    ic_params.grid_size = 64;                 // Smaller grid for testing
    ic_params.box_size = 100.0f;              // 100 Mpc/h
    ic_params.z_initial = 49.0;               // Initial redshift
    ic_params.ps_type = PowerSpectrumType::EISENSTEIN_HU;
    ic_params.random_seed = 12345;
    ic_params.normalize_at_z0 = true;
    
    std::cout << "Cosmology parameters:" << std::endl;
    std::cout << "  Omega_m = " << cosmo_params.omega_m << std::endl;
    std::cout << "  Omega_Lambda = " << cosmo_params.omega_lambda << std::endl;
    std::cout << "  h = " << cosmo_params.h << std::endl;
    std::cout << "  sigma_8 = " << cosmo_params.sigma_8 << std::endl;
    std::cout << "  n_s = " << cosmo_params.n_s << std::endl;
    
    std::cout << "\nInitial conditions parameters:" << std::endl;
    std::cout << "  Grid size: " << ic_params.grid_size << "³" << std::endl;
    std::cout << "  Box size: " << ic_params.box_size << " Mpc/h" << std::endl;
    std::cout << "  Initial redshift: " << ic_params.z_initial << std::endl;
    
    try {
        // Create initial conditions generator
        auto start_time = std::chrono::high_resolution_clock::now();
        
        InitialConditionsGenerator ic_gen(ic_params, cosmology);
        
        auto init_time = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_time - start_time);
        std::cout << "\nInitialization time: " << init_duration.count() << " ms" << std::endl;
        
        // Validate power spectrum
        std::cout << "\nValidating power spectrum..." << std::endl;
        ic_gen.validate_power_spectrum();
        
        // Generate initial conditions
        const size_t num_particles = 10000;
        std::vector<float3> positions;
        std::vector<float3> velocities;
        std::vector<float> masses;
        
        std::cout << "\nGenerating initial conditions for " << num_particles << " particles..." << std::endl;
        
        auto gen_start = std::chrono::high_resolution_clock::now();
        ic_gen.generate_particles(num_particles, positions, velocities, masses);
        auto gen_end = std::chrono::high_resolution_clock::now();
        
        auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);
        std::cout << "Generation time: " << gen_duration.count() << " ms" << std::endl;
        
        // Validate initial conditions
        std::cout << "\nValidating generated initial conditions..." << std::endl;
        bool valid = initial_conditions_utils::validate_initial_conditions(
            positions, velocities, masses, ic_params.box_size);
        
        if (valid) {
            std::cout << "✓ Initial conditions are valid!" << std::endl;
        } else {
            std::cerr << "✗ Initial conditions validation failed!" << std::endl;
            return 1;
        }
        
        // Compute statistics
        std::cout << "\nComputing statistics..." << std::endl;
        
        // Center of mass
        float3 center_of_mass = {0.0f, 0.0f, 0.0f};
        float total_mass = 0.0f;
        
        for (size_t i = 0; i < num_particles; i++) {
            center_of_mass.x += positions[i].x * masses[i];
            center_of_mass.y += positions[i].y * masses[i];
            center_of_mass.z += positions[i].z * masses[i];
            total_mass += masses[i];
        }
        
        center_of_mass.x /= total_mass;
        center_of_mass.y /= total_mass;
        center_of_mass.z /= total_mass;
        
        // RMS velocities
        float rms_vx = 0.0f, rms_vy = 0.0f, rms_vz = 0.0f;
        for (const auto& vel : velocities) {
            rms_vx += vel.x * vel.x;
            rms_vy += vel.y * vel.y;
            rms_vz += vel.z * vel.z;
        }
        
        rms_vx = std::sqrt(rms_vx / num_particles);
        rms_vy = std::sqrt(rms_vy / num_particles);
        rms_vz = std::sqrt(rms_vz / num_particles);
        
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Total particles: " << num_particles << std::endl;
        std::cout << "  Total mass: " << total_mass << std::endl;
        std::cout << "  Center of mass: (" << center_of_mass.x << ", " << center_of_mass.y << ", " << center_of_mass.z << ")" << std::endl;
        std::cout << "  RMS velocities: vx=" << rms_vx << ", vy=" << rms_vy << ", vz=" << rms_vz << " km/s" << std::endl;
        
        // Position range
        float3 pos_min = {1e30f, 1e30f, 1e30f};
        float3 pos_max = {-1e30f, -1e30f, -1e30f};
        
        for (const auto& pos : positions) {
            pos_min.x = std::min(pos_min.x, pos.x);
            pos_min.y = std::min(pos_min.y, pos.y);
            pos_min.z = std::min(pos_min.z, pos.z);
            pos_max.x = std::max(pos_max.x, pos.x);
            pos_max.y = std::max(pos_max.y, pos.y);
            pos_max.z = std::max(pos_max.z, pos.z);
        }
        
        std::cout << "  Position range: [" << pos_min.x << ", " << pos_max.x << "] Mpc/h" << std::endl;
        std::cout << "                  [" << pos_min.y << ", " << pos_max.y << "] Mpc/h" << std::endl;
        std::cout << "                  [" << pos_min.z << ", " << pos_max.z << "] Mpc/h" << std::endl;
        
        // Sample a few particles
        std::cout << "\nSample particles:" << std::endl;
        for (int i = 0; i < std::min(5, (int)num_particles); i++) {
            std::cout << "  Particle " << i << ": "
                      << "pos=(" << positions[i].x << ", " << positions[i].y << ", " << positions[i].z << ") "
                      << "vel=(" << velocities[i].x << ", " << velocities[i].y << ", " << velocities[i].z << ") "
                      << "mass=" << masses[i] << std::endl;
        }
        
        std::cout << "\n=== Test completed successfully! ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}