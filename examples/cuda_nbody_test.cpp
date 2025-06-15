#include <iostream>
#include <chrono>
#include <iomanip>
#include "physics/lambda_cdm.hpp"

using namespace physics;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        size_t num_particles = 10000;
        int num_steps = 100;
        
        if (argc > 1) {
            num_particles = std::stoul(argv[1]);
        }
        if (argc > 2) {
            num_steps = std::stoi(argv[2]);
        }
        
        std::cout << "\n=== Lambda-CDM N-body CUDA Test ===" << std::endl;
        std::cout << "Number of particles: " << num_particles << std::endl;
        std::cout << "Number of steps: " << num_steps << std::endl;
        
        // Set up cosmological parameters
        CosmologyParams params;
        params.omega_m = 0.31;
        params.omega_lambda = 0.69;
        params.h = 0.67;
        
        // Create simulation
        float box_size = 100.0f;  // Mpc/h
        LambdaCDMSimulation sim(num_particles, box_size, params);
        
        // Set simulation parameters
        sim.set_softening(box_size / 1000.0f);  // 0.1% of box size
        
        // Initialize particles
        std::cout << "\nInitializing particles..." << std::endl;
        sim.initialize_particles();
        
        // Initial energy computation
        std::cout << "\nComputing initial energy..." << std::endl;
        sim.compute_energy();
        float initial_energy = sim.get_total_energy();
        std::cout << "Initial energy: " << initial_energy << std::endl;
        std::cout << "  Kinetic: " << sim.get_kinetic_energy() << std::endl;
        std::cout << "  Potential: " << sim.get_potential_energy() << std::endl;
        
        // Run simulation
        std::cout << "\nRunning simulation..." << std::endl;
        double dt = 0.001;  // Small timestep for accuracy
        
        auto start_time = high_resolution_clock::now();
        
        for (int i = 0; i < num_steps; i++) {
            sim.step(dt);
            
            if ((i + 1) % 10 == 0) {
                sim.compute_energy();
                float current_energy = sim.get_total_energy();
                float energy_error = std::abs((current_energy - initial_energy) / initial_energy);
                
                std::cout << "Step " << std::setw(4) << (i + 1) 
                          << " | z = " << std::fixed << std::setprecision(3) 
                          << sim.get_redshift()
                          << " | Energy error: " << std::scientific 
                          << energy_error << std::endl;
            }
        }
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        // Final statistics
        sim.compute_energy();
        float final_energy = sim.get_total_energy();
        float energy_error = std::abs((final_energy - initial_energy) / initial_energy);
        
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "Total time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Time per step: " << duration.count() / (double)num_steps << " ms" << std::endl;
        std::cout << "Final redshift: " << sim.get_redshift() << std::endl;
        std::cout << "Final energy: " << final_energy << std::endl;
        std::cout << "  Kinetic: " << sim.get_kinetic_energy() << std::endl;
        std::cout << "  Potential: " << sim.get_potential_energy() << std::endl;
        std::cout << "Energy conservation error: " << energy_error << std::endl;
        
        // Performance metrics
        double particles_per_second = (double)num_particles * num_steps / (duration.count() / 1000.0);
        std::cout << "\nPerformance: " << std::scientific << particles_per_second 
                  << " particle-updates/second" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}