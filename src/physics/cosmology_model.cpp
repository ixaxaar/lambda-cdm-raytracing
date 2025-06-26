#include "physics/cosmology_model.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace physics {

// Additional implementation for advanced cosmological calculations

// Integrate scale factor evolution
std::vector<double> CosmologyModel::integrate_scale_factor(
    double a_start, double a_end, int n_steps) const {

    std::vector<double> scale_factors;
    scale_factors.reserve(n_steps);

    double da = (a_end - a_start) / (n_steps - 1);

    for (int i = 0; i < n_steps; i++) {
        scale_factors.push_back(a_start + i * da);
    }

    return scale_factors;
}

// Compute conformal time
double CosmologyModel::conformal_time(double a) const {
    // tau = integral from 0 to a of da'/H(a')/a'^2
    const int n_steps = 1000;
    double da_step = a / n_steps;
    double tau = 0.0;

    for (int i = 1; i <= n_steps; i++) {
        double a_mid = (i - 0.5) * da_step;
        double H = hubble_parameter_a(a_mid);
        tau += da_step / (H * a_mid * a_mid);
    }

    // Convert from 1/H0 units to Mpc
    return tau * c_light / params_.H0();
}

// Compute cosmic time (age of universe at scale factor a)
double CosmologyModel::cosmic_time(double a) const {
    // t = integral from 0 to a of da'/H(a')/a'
    const int n_steps = 1000;
    double da_step = a / n_steps;
    double t = 0.0;

    for (int i = 1; i <= n_steps; i++) {
        double a_mid = (i - 0.5) * da_step;
        double H = hubble_parameter_a(a_mid) / params_.H0();  // H/H0
        t += da_step / (H * a_mid);
    }

    // Convert from 1/H0 units to Gyr
    // H0 = 100h km/s/Mpc, 1/H0 = 9.778 h^-1 Gyr
    double H0_inv_gyr = 9.778 / params_.h;  // 1/H0 in Gyr
    return t * H0_inv_gyr;
}

// Compute age of universe at redshift z
double CosmologyModel::age_at_redshift(double z) const {
    double a = z_to_a(z);
    return cosmic_time(a);
}

// Print cosmology summary
void CosmologyModel::print_summary() const {
    std::cout << "\n=== Lambda-CDM Cosmology Model ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Omega_m:     " << params_.omega_m << std::endl;
    std::cout << "Omega_Lambda: " << params_.omega_lambda << std::endl;
    std::cout << "Omega_k:     " << params_.omega_k << std::endl;
    std::cout << "h:           " << params_.h << std::endl;
    std::cout << "H0:          " << params_.H0() << " km/s/Mpc" << std::endl;
    std::cout << "sigma_8:     " << params_.sigma_8 << std::endl;
    std::cout << "n_s:         " << params_.n_s << std::endl;

    // Derived quantities
    std::cout << "\nDerived quantities:" << std::endl;
    std::cout << "Age of universe: " << cosmic_time(1.0) << " Gyr" << std::endl;
    std::cout << "Age at z=1: " << age_at_redshift(1.0) << " Gyr" << std::endl;
    std::cout << "Age at z=2: " << age_at_redshift(2.0) << " Gyr" << std::endl;
    std::cout << "Age at z=5: " << age_at_redshift(5.0) << " Gyr" << std::endl;

    // Distances
    std::cout << "\nCosmological distances:" << std::endl;
    std::cout << "Comoving distance to z=1: " << comoving_distance(1.0) << " Mpc" << std::endl;
    std::cout << "Angular diameter distance to z=1: " << angular_diameter_distance(1.0) << " Mpc" << std::endl;
    std::cout << "Luminosity distance to z=1: " << luminosity_distance(1.0) << " Mpc" << std::endl;
}

// Power spectrum of density fluctuations (simplified BBKS transfer function)
double CosmologyModel::power_spectrum(double k, double z) const {
    // k in h/Mpc
    double a = z_to_a(z);
    double D = growth_factor(a);

    // BBKS transfer function
    double q = k / (params_.omega_m * params_.h * params_.h);
    double T = std::log(1.0 + 2.34 * q) / (2.34 * q) *
               std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) +
                       std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);

    // Primordial power spectrum
    double k_pivot = 0.05;  // Mpc^-1
    double P_primordial = std::pow(k / k_pivot, params_.n_s - 1.0);

    // Full power spectrum
    double P_k = P_primordial * T * T * D * D;

    // For now, just return unnormalized power spectrum
    // TODO: Implement proper normalization without recursion
    return P_k;
}

// Variance of density field smoothed on scale R (Mpc/h)
double CosmologyModel::variance_at_scale(double R) const {
    // Integrate P(k) * W(kR)^2 * k^2 / (2π^2)
    const int n_k = 1000;
    double k_min = 1e-4;
    double k_max = 1e2;
    double dlogk = std::log(k_max / k_min) / n_k;

    double variance = 0.0;
    for (int i = 0; i < n_k; i++) {
        double logk = std::log(k_min) + (i + 0.5) * dlogk;
        double k = std::exp(logk);

        // Top-hat window function in Fourier space
        double x = k * R;
        double W = 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);

        // Power spectrum at z=0
        double P_k = power_spectrum(k, 0.0);

        variance += P_k * W * W * k * k * k * dlogk / (2.0 * M_PI * M_PI);
    }

    return variance;
}

} // namespace physics