#include "core/simulation_engine.hpp"
#include "forces/force_computer_factory.hpp"
#include <iostream>
#include <memory>

using namespace core;
using namespace forces;

int main(int argc, char* argv[]) {
    try {
        // Register all built-in components
        ForceComputerFactory::register_all_builtin_computers();
        ForceComputerFactory::register_all_builtin_kernels();
        
        // Build simulation using the builder pattern
        auto simulation = SimulationBuilder()
            .with_config_file("examples/configs/basic_lambda_cdm.json")
            .with_num_particles(100000)
            .with_box_size(100.0f)
            .with_time_step(0.01)
            .with_max_time(10.0)
            .with_output_directory("output/basic_sim")
            .with_force_computer("TreeForceComputer")
            .with_integrator("LeapfrogIntegrator")
            .with_cosmology_model("LambdaCDMModel")
            .enable_gpu(0)
            .enable_tensorrt("models/nbody_engine.trt")
            .build();
        
        if (!simulation) {
            std::cerr << "Failed to create simulation!" << std::endl;
            return 1;
        }
        
        std::cout << "Starting Lambda-CDM simulation..." << std::endl;
        std::cout << "Particles: " << simulation->get_context().get_num_particles() << std::endl;
        std::cout << "Time step: " << simulation->get_context().get_parameter<double>("time_step") << std::endl;
        
        // Run the simulation
        bool success = simulation->run();
        
        if (success) {
            std::cout << "Simulation completed successfully!" << std::endl;
            
            // Print performance summary
            simulation->print_performance_summary();
            
            // Final statistics
            const auto& stats = simulation->get_statistics();
            std::cout << "\nFinal Statistics:" << std::endl;
            std::cout << "Total steps: " << stats.total_steps << std::endl;
            std::cout << "Final time: " << stats.current_time << std::endl;
            std::cout << "Final redshift: " << stats.redshift << std::endl;
            std::cout << "Average steps/sec: " << stats.steps_per_second << std::endl;
            
            // Energy conservation check
            double total_energy = simulation->compute_total_energy();
            std::cout << "Total energy: " << total_energy << std::endl;
            
        } else {
            std::cerr << "Simulation failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}