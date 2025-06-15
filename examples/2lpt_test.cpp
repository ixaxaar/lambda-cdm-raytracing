#include "physics/initial_conditions.hpp"
#include "physics/cosmology_model.hpp"
#include <iostream>
#include <chrono>

using namespace physics;

int main() {
    std::cout << "=== 2LPT (Second-order Lagrangian Perturbation Theory) Test ===" << std::endl;
    
    // Set up cosmology
    CosmologyParams cosmo_params;
    cosmo_params.omega_m = 0.31;
    cosmo_params.omega_lambda = 0.69;
    cosmo_params.h = 0.67;
    cosmo_params.sigma_8 = 0.81;
    cosmo_params.n_s = 0.965;
    
    CosmologyModel cosmology(cosmo_params);
    
    std::cout << "Cosmology parameters:" << std::endl;
    std::cout << "  Omega_m = " << cosmo_params.omega_m << std::endl;
    std::cout << "  Omega_Lambda = " << cosmo_params.omega_lambda << std::endl;
    std::cout << "  h = " << cosmo_params.h << std::endl;
    std::cout << "  sigma_8 = " << cosmo_params.sigma_8 << std::endl;
    std::cout << "  n_s = " << cosmo_params.n_s << std::endl;
    
    // Test both 1LPT (Zel'dovich) and 2LPT
    const size_t num_particles = 5000;
    
    // Test 1: Standard Zel'dovich approximation
    std::cout << "\n=== Test 1: Zel'dovich (1LPT) Approximation ===" << std::endl;
    
    InitialConditionsParams ic_params_1lpt;
    ic_params_1lpt.grid_size = 32;          // Smaller grid for faster computation
    ic_params_1lpt.box_size = 50.0f;        // 50 Mpc/h
    ic_params_1lpt.z_initial = 49.0;
    ic_params_1lpt.ps_type = PowerSpectrumType::EISENSTEIN_HU;
    ic_params_1lpt.random_seed = 12345;
    ic_params_1lpt.use_2lpt = false;        // Use 1LPT
    ic_params_1lpt.normalize_at_z0 = true;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        InitialConditionsGenerator ic_gen_1lpt(ic_params_1lpt, cosmology);
        
        std::vector<float3> positions_1lpt;
        std::vector<float3> velocities_1lpt;
        std::vector<float> masses_1lpt;
        
        ic_gen_1lpt.generate_particles(num_particles, positions_1lpt, velocities_1lpt, masses_1lpt);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Validate
        bool valid_1lpt = initial_conditions_utils::validate_initial_conditions(
            positions_1lpt, velocities_1lpt, masses_1lpt, ic_params_1lpt.box_size);
        
        std::cout << "1LPT Results:" << std::endl;
        std::cout << "  Generation time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Validation: " << (valid_1lpt ? "✓ PASSED" : "✗ FAILED") << std::endl;
        
        // Compute RMS displacement
        double rms_disp_1lpt = ic_gen_1lpt.compute_rms_displacement(positions_1lpt);
        double rms_vel_1lpt = ic_gen_1lpt.compute_rms_velocity(velocities_1lpt);
        
        std::cout << "  RMS displacement: " << rms_disp_1lpt << " Mpc/h" << std::endl;
        std::cout << "  RMS velocity: " << rms_vel_1lpt << " km/s" << std::endl;
        
        // Test 2: 2LPT approximation
        std::cout << "\n=== Test 2: 2LPT Approximation ===" << std::endl;
        
        InitialConditionsParams ic_params_2lpt = ic_params_1lpt;
        ic_params_2lpt.use_2lpt = true;         // Use 2LPT
        ic_params_2lpt.random_seed = 12345;     // Same seed for comparison
        
        start_time = std::chrono::high_resolution_clock::now();
        
        InitialConditionsGenerator ic_gen_2lpt(ic_params_2lpt, cosmology);
        
        std::vector<float3> positions_2lpt;
        std::vector<float3> velocities_2lpt;
        std::vector<float> masses_2lpt;
        
        ic_gen_2lpt.generate_particles(num_particles, positions_2lpt, velocities_2lpt, masses_2lpt);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Validate
        bool valid_2lpt = initial_conditions_utils::validate_initial_conditions(
            positions_2lpt, velocities_2lpt, masses_2lpt, ic_params_2lpt.box_size);
        
        std::cout << "2LPT Results:" << std::endl;
        std::cout << "  Generation time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Validation: " << (valid_2lpt ? "✓ PASSED" : "✗ FAILED") << std::endl;
        
        // Compute RMS displacement
        double rms_disp_2lpt = ic_gen_2lpt.compute_rms_displacement(positions_2lpt);
        double rms_vel_2lpt = ic_gen_2lpt.compute_rms_velocity(velocities_2lpt);
        
        std::cout << "  RMS displacement: " << rms_disp_2lpt << " Mpc/h" << std::endl;
        std::cout << "  RMS velocity: " << rms_vel_2lpt << " km/s" << std::endl;
        
        // Compare results
        std::cout << "\n=== Comparison: 1LPT vs 2LPT ===" << std::endl;
        
        double disp_ratio = rms_disp_2lpt / rms_disp_1lpt;
        double vel_ratio = rms_vel_2lpt / rms_vel_1lpt;
        
        std::cout << "  RMS displacement ratio (2LPT/1LPT): " << disp_ratio << std::endl;
        std::cout << "  RMS velocity ratio (2LPT/1LPT): " << vel_ratio << std::endl;
        
        // Compute cross-correlation (simple measure of similarity)
        double cross_corr = 0.0;
        double norm_1lpt = 0.0;
        double norm_2lpt = 0.0;
        
        size_t min_particles = std::min(positions_1lpt.size(), positions_2lpt.size());
        for (size_t i = 0; i < min_particles; i++) {
            // Use displacement from grid center for correlation
            float3 center = make_float3(ic_params_1lpt.box_size/2, ic_params_1lpt.box_size/2, ic_params_1lpt.box_size/2);
            
            float3 disp_1lpt = make_float3(
                positions_1lpt[i].x - center.x,
                positions_1lpt[i].y - center.y,
                positions_1lpt[i].z - center.z
            );
            
            float3 disp_2lpt = make_float3(
                positions_2lpt[i].x - center.x,
                positions_2lpt[i].y - center.y,
                positions_2lpt[i].z - center.z
            );
            
            cross_corr += disp_1lpt.x * disp_2lpt.x + disp_1lpt.y * disp_2lpt.y + disp_1lpt.z * disp_2lpt.z;
            norm_1lpt += disp_1lpt.x * disp_1lpt.x + disp_1lpt.y * disp_1lpt.y + disp_1lpt.z * disp_1lpt.z;
            norm_2lpt += disp_2lpt.x * disp_2lpt.x + disp_2lpt.y * disp_2lpt.y + disp_2lpt.z * disp_2lpt.z;
        }
        
        double correlation = cross_corr / std::sqrt(norm_1lpt * norm_2lpt);
        std::cout << "  Cross-correlation coefficient: " << correlation << std::endl;
        
        // Sample comparison
        std::cout << "\nSample particle comparison (first 3 particles):" << std::endl;
        for (int i = 0; i < std::min(3, (int)min_particles); i++) {
            std::cout << "  Particle " << i << ":" << std::endl;
            std::cout << "    1LPT pos: (" << positions_1lpt[i].x << ", " << positions_1lpt[i].y << ", " << positions_1lpt[i].z << ")" << std::endl;
            std::cout << "    2LPT pos: (" << positions_2lpt[i].x << ", " << positions_2lpt[i].y << ", " << positions_2lpt[i].z << ")" << std::endl;
            std::cout << "    1LPT vel: (" << velocities_1lpt[i].x << ", " << velocities_1lpt[i].y << ", " << velocities_1lpt[i].z << ")" << std::endl;
            std::cout << "    2LPT vel: (" << velocities_2lpt[i].x << ", " << velocities_2lpt[i].y << ", " << velocities_2lpt[i].z << ")" << std::endl;
        }
        
        // Physical interpretation
        std::cout << "\n=== Physical Interpretation ===" << std::endl;
        std::cout << "2LPT provides more accurate initial conditions by including:" << std::endl;
        std::cout << "  - Second-order gravitational effects" << std::endl;
        std::cout << "  - Non-linear mode coupling" << std::endl;
        std::cout << "  - Improved clustering at small scales" << std::endl;
        std::cout << "  - Better representation of voids and filaments" << std::endl;
        
        if (correlation > 0.8) {
            std::cout << "\nHigh correlation indicates 2LPT provides corrections to 1LPT" << std::endl;
            std::cout << "while preserving the large-scale structure." << std::endl;
        }
        
        if (disp_ratio > 1.0) {
            std::cout << "\n2LPT produces larger displacements, indicating enhanced" << std::endl;
            std::cout << "small-scale clustering due to second-order effects." << std::endl;
        }
        
        std::cout << "\n=== Test completed successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}