#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include "core/math_types.hpp"

namespace physics {

// Cosmological parameters for Lambda-CDM model
struct CosmologyParams {
    double omega_m = 0.31;        // Matter density parameter
    double omega_lambda = 0.69;   // Dark energy density parameter
    double omega_k = 0.0;         // Curvature density parameter (flat universe)
    double h = 0.67;              // Hubble parameter (H0 = 100h km/s/Mpc)
    double sigma_8 = 0.81;        // Power spectrum normalization
    double n_s = 0.965;           // Spectral index
    
    // Derived parameters
    double H0() const { return 100.0 * h; }  // Hubble constant in km/s/Mpc
    
    // Validate parameters
    void validate() const {
        double omega_total = omega_m + omega_lambda + omega_k;
        if (std::abs(omega_total - 1.0) > 1e-6) {
            throw std::runtime_error("Cosmological parameters must sum to 1.0");
        }
        if (omega_m < 0 || omega_lambda < 0) {
            throw std::runtime_error("Density parameters must be non-negative");
        }
    }
};

// Lambda-CDM cosmology model
class CosmologyModel {
private:
    CosmologyParams params_;
    
    // Constants
    static constexpr double c_light = 299792.458;  // Speed of light in km/s
    static constexpr double G = 4.3009e-9;         // G in (km/s)^2 Mpc/M_sun
    
public:
    explicit CosmologyModel(const CosmologyParams& params) : params_(params) {
        params_.validate();
    }
    
    // Friedmann equation: compute Hubble parameter at redshift z
    double hubble_parameter(double z) const {
        double a = 1.0 / (1.0 + z);  // Scale factor
        double E_squared = params_.omega_m * std::pow(a, -3) + 
                          params_.omega_k * std::pow(a, -2) + 
                          params_.omega_lambda;
        return params_.H0() * std::sqrt(E_squared);
    }
    
    // Hubble parameter at scale factor a
    double hubble_parameter_a(double a) const {
        double z = 1.0 / a - 1.0;
        return hubble_parameter(z);
    }
    
    // Time derivative of scale factor: da/dt = a * H(a)
    double scale_factor_derivative(double a) const {
        return a * hubble_parameter_a(a);
    }
    
    // Second derivative of scale factor (for acceleration)
    double scale_factor_second_derivative(double a) const {
        // d²a/dt² = -4πG/3 * a * (ρ + 3p/c²)
        // For Lambda-CDM: ρ = ρ_m + ρ_Λ, p = p_Λ = -ρ_Λ
        double rho_crit_0 = 3.0 * params_.H0() * params_.H0() / (8.0 * M_PI * G);
        double rho_m = params_.omega_m * rho_crit_0 * std::pow(a, -3);
        double rho_lambda = params_.omega_lambda * rho_crit_0;
        
        return -4.0 * M_PI * G / 3.0 * a * (rho_m - 2.0 * rho_lambda);
    }
    
    // Growth factor D(a) for linear perturbations
    double growth_factor(double a) const {
        // Approximate solution for flat Lambda-CDM
        double omega_m_z = omega_matter_a(a);
        double omega_l_z = omega_lambda_a(a);
        
        // Carroll et al. (1992) approximation
        double D = a * std::pow(omega_m_z, 0.6) / 
                  (std::pow(omega_m_z, 0.6) + 
                   omega_l_z * (1.0 + omega_m_z / 70.0));
        
        return D;
    }
    
    // Growth rate f = d log D / d log a
    double growth_rate(double a) const {
        // Approximation: f ≈ Ω_m(a)^0.55
        return std::pow(omega_matter_a(a), 0.55);
    }
    
    // Matter density parameter at scale factor a
    double omega_matter_a(double a) const {
        double E2 = std::pow(hubble_parameter_a(a) / params_.H0(), 2);
        return params_.omega_m * std::pow(a, -3) / E2;
    }
    
    // Dark energy density parameter at scale factor a
    double omega_lambda_a(double a) const {
        double E2 = std::pow(hubble_parameter_a(a) / params_.H0(), 2);
        return params_.omega_lambda / E2;
    }
    
    // Comoving distance to redshift z (in Mpc)
    double comoving_distance(double z) const {
        // Integrate 1/H(z') from 0 to z
        const int n_steps = 1000;
        double dz = z / n_steps;
        double chi = 0.0;
        
        for (int i = 0; i < n_steps; i++) {
            double z_mid = (i + 0.5) * dz;
            chi += c_light / hubble_parameter(z_mid) * dz;
        }
        
        return chi;
    }
    
    // Angular diameter distance (in Mpc)
    double angular_diameter_distance(double z) const {
        double chi = comoving_distance(z);
        
        if (std::abs(params_.omega_k) < 1e-10) {
            // Flat universe
            return chi / (1.0 + z);
        } else if (params_.omega_k > 0) {
            // Open universe
            double sqrt_ok = std::sqrt(params_.omega_k);
            double DH = c_light / params_.H0();
            return DH / sqrt_ok * std::sinh(sqrt_ok * chi / DH) / (1.0 + z);
        } else {
            // Closed universe
            double sqrt_ok = std::sqrt(-params_.omega_k);
            double DH = c_light / params_.H0();
            return DH / sqrt_ok * std::sin(sqrt_ok * chi / DH) / (1.0 + z);
        }
    }
    
    // Luminosity distance (in Mpc)
    double luminosity_distance(double z) const {
        return angular_diameter_distance(z) * std::pow(1.0 + z, 2);
    }
    
    // Convert redshift to scale factor
    static double z_to_a(double z) {
        return 1.0 / (1.0 + z);
    }
    
    // Convert scale factor to redshift
    static double a_to_z(double a) {
        return 1.0 / a - 1.0;
    }
    
    // Get parameters
    const CosmologyParams& get_params() const { return params_; }
    
    // Additional declarations for implementation file
    std::vector<double> integrate_scale_factor(double a_start, double a_end, int n_steps) const;
    double conformal_time(double a) const;
    double cosmic_time(double a) const;
    double age_at_redshift(double z) const;
    void print_summary() const;
    double power_spectrum(double k, double z) const;
    double variance_at_scale(double R) const;
};

} // namespace physics