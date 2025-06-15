#include <iostream>
#include <iomanip>
#include <vector>
#include "physics/cosmology_model.hpp"

using namespace physics;

int main() {
    std::cout << "\n=== Lambda-CDM Cosmology Test ===" << std::endl;
    
    // Set up standard Lambda-CDM parameters
    CosmologyParams params;
    params.omega_m = 0.31;
    params.omega_lambda = 0.69;
    params.omega_k = 0.0;  // Flat universe
    params.h = 0.67;
    params.sigma_8 = 0.81;
    params.n_s = 0.965;
    
    // Create cosmology model
    CosmologyModel cosmo(params);
    
    // Print summary
    cosmo.print_summary();
    
    // Test scale factor evolution
    std::cout << "\n=== Scale Factor Evolution ===" << std::endl;
    std::cout << std::setw(12) << "z" << std::setw(12) << "a" 
              << std::setw(15) << "H(z) [km/s/Mpc]" 
              << std::setw(15) << "Age [Gyr]"
              << std::setw(15) << "Growth D(z)" << std::endl;
    std::cout << std::string(69, '-') << std::endl;
    
    std::vector<double> redshifts = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
    for (double z : redshifts) {
        double a = CosmologyModel::z_to_a(z);
        double H = cosmo.hubble_parameter(z);
        double age = cosmo.age_at_redshift(z);
        double D = cosmo.growth_factor(a);
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(12) << z
                  << std::setprecision(4)
                  << std::setw(12) << a
                  << std::setprecision(1)
                  << std::setw(15) << H
                  << std::setprecision(2)
                  << std::setw(15) << age
                  << std::setprecision(4)
                  << std::setw(15) << D
                  << std::endl;
    }
    
    // Test distance measures
    std::cout << "\n=== Cosmological Distances ===" << std::endl;
    std::cout << std::setw(12) << "z" 
              << std::setw(20) << "Comoving [Mpc]"
              << std::setw(20) << "Angular Diam [Mpc]"
              << std::setw(20) << "Luminosity [Mpc]" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    for (double z : {0.1, 0.5, 1.0, 2.0, 3.0}) {
        double d_c = cosmo.comoving_distance(z);
        double d_a = cosmo.angular_diameter_distance(z);
        double d_l = cosmo.luminosity_distance(z);
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(12) << z
                  << std::setprecision(1)
                  << std::setw(20) << d_c
                  << std::setw(20) << d_a
                  << std::setw(20) << d_l
                  << std::endl;
    }
    
    // Test power spectrum
    std::cout << "\n=== Matter Power Spectrum at z=0 ===" << std::endl;
    std::cout << std::setw(15) << "k [h/Mpc]" 
              << std::setw(20) << "P(k) [(Mpc/h)^3]" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<double> k_values = {0.001, 0.01, 0.1, 1.0, 10.0};
    for (double k : k_values) {
        double P_k = cosmo.power_spectrum(k, 0.0);
        
        std::cout << std::scientific << std::setprecision(3)
                  << std::setw(15) << k
                  << std::setw(20) << P_k
                  << std::endl;
    }
    
    // Test growth history
    std::cout << "\n=== Growth History ===" << std::endl;
    std::cout << std::setw(12) << "z" 
              << std::setw(15) << "Growth D(z)"
              << std::setw(15) << "Growth rate f"
              << std::setw(15) << "Omega_m(z)" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    for (double z : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        double a = CosmologyModel::z_to_a(z);
        double D = cosmo.growth_factor(a);
        double f = cosmo.growth_rate(a);
        double omega_m = cosmo.omega_matter_a(a);
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(12) << z
                  << std::setprecision(4)
                  << std::setw(15) << D
                  << std::setw(15) << f
                  << std::setw(15) << omega_m
                  << std::endl;
    }
    
    return 0;
}