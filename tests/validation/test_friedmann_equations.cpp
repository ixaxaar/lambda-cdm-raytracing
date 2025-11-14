#include <gtest/gtest.h>
#include "physics/lambda_cdm.hpp"
#include <cmath>

using namespace physics;

class FriedmannEquationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard Planck 2018 cosmology
        params.omega_m = 0.315;
        params.omega_lambda = 0.685;
        params.h = 0.674;
        params.sigma_8 = 0.811;
        params.n_s = 0.965;

        model = std::make_unique<CosmologyModel>(params);
    }

    CosmologyParams params;
    std::unique_ptr<CosmologyModel> model;
};

// Test: Friedmann equation - flat universe
TEST_F(FriedmannEquationsTest, FlatUniverseConstraint) {
    // For a flat universe: Ω_m + Ω_Λ + Ω_k = 1
    double sum = params.omega_m + params.omega_lambda + params.omega_k;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// Test: Hubble parameter satisfies first Friedmann equation
TEST_F(FriedmannEquationsTest, FirstFriedmannEquation) {
    // H²(a) = H₀² [Ω_m a^(-3) + Ω_Λ]  (for flat universe with no radiation)

    for (double a = 0.2; a <= 1.0; a += 0.2) {
        double H = model->hubble_parameter_a(a);
        double H_expected = params.H0() * std::sqrt(
            params.omega_m * std::pow(a, -3.0) + params.omega_lambda
        );

        // Should match within 0.1%
        EXPECT_NEAR(H, H_expected, 0.001 * H_expected);
    }
}

// Test: Matter density evolution
TEST_F(FriedmannEquationsTest, MatterDensityEvolution) {
    // ρ_m(a) = ρ_m0 * a^(-3)
    // Ω_m(a) = Ω_m0 * a^(-3) / E²(a)
    // where E(a) = H(a)/H0

    for (double a = 0.3; a <= 1.0; a += 0.2) {
        double Om_a = model->omega_matter_a(a);
        double H = model->hubble_parameter_a(a);
        double E2 = (H * H) / (params.H0() * params.H0());

        double Om_expected = params.omega_m * std::pow(a, -3.0) / E2;

        EXPECT_NEAR(Om_a, Om_expected, 1e-6);
    }
}

// Test: Dark energy density evolution
TEST_F(FriedmannEquationsTest, DarkEnergyEvolution) {
    // For cosmological constant: ρ_Λ is constant
    // Ω_Λ(a) = Ω_Λ0 / E²(a)

    for (double a = 0.3; a <= 1.0; a += 0.2) {
        double OL_a = model->omega_lambda_a(a);
        double H = model->hubble_parameter_a(a);
        double E2 = (H * H) / (params.H0() * params.H0());

        double OL_expected = params.omega_lambda / E2;

        EXPECT_NEAR(OL_a, OL_expected, 1e-6);
    }
}

// Test: Scale factor evolution (matter-dominated)
TEST_F(FriedmannEquationsTest, MatterDominatedEvolution) {
    // In matter-dominated era: a(t) ∝ t^(2/3)
    // H(a) = 2/(3t)

    // At high redshift where matter dominates
    double a_early = 0.01;  // z = 99
    double H_early = model->hubble_parameter_a(a_early);

    // In matter domination: H² ≈ H₀² Ω_m a^(-3)
    double H_expected = params.H0() * std::sqrt(params.omega_m) * std::pow(a_early, -1.5);

    // Should be close at high redshift
    EXPECT_NEAR(H_early, H_expected, 0.05 * H_expected);  // Within 5%
}

// Test: Deceleration parameter
TEST_F(FriedmannEquationsTest, DecelerationParameter) {
    // q = -ä·a/ȧ² = 1/2 [Ω_m - 2Ω_Λ]  (approximately, for flat universe)

    double a = 1.0;  // Today
    double Om = model->omega_matter_a(a);
    double OL = model->omega_lambda_a(a);

    double q_expected = 0.5 * (Om - 2.0 * OL);

    // For Planck cosmology, q < 0 (accelerating)
    EXPECT_LT(q_expected, 0.0);
}

// Test: Age of universe calculation
TEST_F(FriedmannEquationsTest, AgeOfUniverse) {
    // Age at z=0 should be ~13.8 Gyr for Planck cosmology
    double age = model->cosmic_time(1.0);

    EXPECT_GT(age, 13.0);   // At least 13 Gyr
    EXPECT_LT(age, 14.5);   // Less than 14.5 Gyr

    // Most likely around 13.8 Gyr
    EXPECT_NEAR(age, 13.8, 0.5);
}

// Test: Lookback time
TEST_F(FriedmannEquationsTest, LookbackTime) {
    // Lookback time to z=1 should be ~half the age of universe
    double age_now = model->cosmic_time(1.0);
    double age_z1 = model->age_at_redshift(1.0);

    double lookback = age_now - age_z1;

    // Lookback to z=1 is ~7-8 Gyr
    EXPECT_GT(lookback, 6.0);
    EXPECT_LT(lookback, 9.0);
}

// Test: Comoving distance integral
TEST_F(FriedmannEquationsTest, ComovingDistanceIntegral) {
    // d_c(z) = c/H₀ ∫₀^z dz'/E(z')
    // where E(z) = H(z)/H₀

    double z = 1.0;
    double d_c = model->comoving_distance(z);

    // For z=1, comoving distance should be ~3000-3500 Mpc for Planck
    EXPECT_GT(d_c, 2500.0);
    EXPECT_LT(d_c, 4000.0);
}

// Test: Light travel time
TEST_F(FriedmannEquationsTest, LightTravelTime) {
    // Light travel time = lookback time (for photons)

    for (double z = 0.1; z <= 3.0; z += 0.5) {
        double age_now = model->cosmic_time(1.0);
        double age_then = model->age_at_redshift(z);
        double light_travel_time = age_now - age_then;

        // Should be positive
        EXPECT_GT(light_travel_time, 0.0);

        // Should increase with redshift
        if (z > 0.1) {
            double prev_time = age_now - model->age_at_redshift(z - 0.5);
            EXPECT_GT(light_travel_time, prev_time);
        }
    }
}

// Test: Distance modulus
TEST_F(FriedmannEquationsTest, DistanceModulus) {
    // μ = 5 log₁₀(d_L/10 pc) = 25 + 5 log₁₀(d_L/Mpc)

    for (double z = 0.1; z <= 2.0; z += 0.5) {
        double d_L = model->luminosity_distance(z);  // in Mpc

        double mu = 25.0 + 5.0 * std::log10(d_L);

        // Sanity checks
        EXPECT_GT(mu, 35.0);   // Should be > 35 for z > 0.1
        EXPECT_LT(mu, 50.0);   // Should be < 50 for z < 2
    }
}

// Test: Consistency between different distance measures
TEST_F(FriedmannEquationsTest, DistanceConsistency) {
    for (double z = 0.5; z <= 2.0; z += 0.5) {
        double d_C = model->comoving_distance(z);
        double d_A = model->angular_diameter_distance(z);
        double d_L = model->luminosity_distance(z);

        // d_A = d_C / (1+z)
        EXPECT_NEAR(d_A, d_C / (1.0 + z), 1e-6 * d_C);

        // d_L = d_C * (1+z)
        EXPECT_NEAR(d_L, d_C * (1.0 + z), 1e-6 * d_C);

        // d_L = d_A * (1+z)²
        EXPECT_NEAR(d_L, d_A * (1.0 + z) * (1.0 + z), 1e-6 * d_L);
    }
}

// Test: Radiation-matter equality (if implemented)
TEST_F(FriedmannEquationsTest, MatterRadiationEquality) {
    // z_eq ≈ 3400 for Planck cosmology
    // At z >> z_eq, universe is radiation dominated
    // At z << z_eq, universe is matter dominated

    // This is a sanity check for very early times
    // (Our current model may not include radiation properly)
}
