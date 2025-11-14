#include <gtest/gtest.h>
#include "physics/lambda_cdm.hpp"
#include <cmath>

using namespace physics;

class CosmologyModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Planck 2018 cosmology
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

// Test: Cosmological parameters
TEST_F(CosmologyModelTest, ParametersCorrect) {
    EXPECT_NEAR(params.omega_m, 0.315, 1e-6);
    EXPECT_NEAR(params.omega_lambda, 0.685, 1e-6);
    EXPECT_NEAR(params.omega_k, 0.0, 1e-6);  // Flat universe
    EXPECT_NEAR(params.H0(), 67.4, 0.1);
}

// Test: Hubble parameter evolution
TEST_F(CosmologyModelTest, HubbleParameterToday) {
    // H(z=0) should equal H0
    double H_today = model->hubble_parameter(0.0);
    EXPECT_NEAR(H_today, params.H0(), 0.1);
}

TEST_F(CosmologyModelTest, HubbleParameterHighRedshift) {
    // At high redshift, matter dominates: H(z) ≈ H0 * (Ωm)^0.5 * (1+z)^1.5
    double z = 10.0;
    double H_expected = params.H0() * std::sqrt(params.omega_m) * std::pow(1 + z, 1.5);
    double H_actual = model->hubble_parameter(z);

    // Should be within 10% at z=10 (some dark energy contribution)
    EXPECT_NEAR(H_actual, H_expected, 0.1 * H_expected);
}

// Test: Scale factor conversion
TEST_F(CosmologyModelTest, ScaleFactorRedshiftConversion) {
    double z = 1.0;
    double a = model->z_to_a(z);
    EXPECT_NEAR(a, 0.5, 1e-6);

    double z_back = model->a_to_z(a);
    EXPECT_NEAR(z_back, z, 1e-6);
}

// Test: Growth factor
TEST_F(CosmologyModelTest, GrowthFactorToday) {
    // D(a=1) should be normalized to 1
    double D = model->growth_factor(1.0);
    EXPECT_NEAR(D, 1.0, 1e-3);
}

TEST_F(CosmologyModelTest, GrowthFactorMonotonic) {
    // Growth factor should increase monotonically with scale factor
    double D1 = model->growth_factor(0.5);
    double D2 = model->growth_factor(0.7);
    double D3 = model->growth_factor(0.9);

    EXPECT_LT(D1, D2);
    EXPECT_LT(D2, D3);
    EXPECT_LT(D3, 1.0);
}

// Test: Comoving distance
TEST_F(CosmologyModelTest, ComovingDistanceZero) {
    // Distance to z=0 should be zero
    double d = model->comoving_distance(0.0);
    EXPECT_NEAR(d, 0.0, 1e-6);
}

TEST_F(CosmologyModelTest, ComovingDistanceMonotonic) {
    // Distance should increase with redshift
    double d1 = model->comoving_distance(0.5);
    double d2 = model->comoving_distance(1.0);
    double d3 = model->comoving_distance(2.0);

    EXPECT_GT(d1, 0.0);
    EXPECT_GT(d2, d1);
    EXPECT_GT(d3, d2);
}

// Test: Angular diameter distance
TEST_F(CosmologyModelTest, AngularDiameterDistance) {
    double z = 1.0;
    double d_C = model->comoving_distance(z);
    double d_A = model->angular_diameter_distance(z);

    // d_A = d_C / (1+z)
    EXPECT_NEAR(d_A, d_C / (1.0 + z), 1e-3);
}

// Test: Luminosity distance
TEST_F(CosmologyModelTest, LuminosityDistance) {
    double z = 1.0;
    double d_C = model->comoving_distance(z);
    double d_L = model->luminosity_distance(z);

    // d_L = d_C * (1+z)
    EXPECT_NEAR(d_L, d_C * (1.0 + z), 1e-3);
}

// Test: Distance duality relation
TEST_F(CosmologyModelTest, DistanceDuality) {
    double z = 1.5;
    double d_A = model->angular_diameter_distance(z);
    double d_L = model->luminosity_distance(z);

    // d_L = d_A * (1+z)^2
    EXPECT_NEAR(d_L, d_A * std::pow(1.0 + z, 2), 1e-3);
}

// Test: Cosmic time
TEST_F(CosmologyModelTest, CosmicTimePositive) {
    // Age should be positive for all scale factors
    EXPECT_GT(model->cosmic_time(0.5), 0.0);
    EXPECT_GT(model->cosmic_time(1.0), 0.0);
}

TEST_F(CosmologyModelTest, CosmicTimeMonotonic) {
    // Time should increase with scale factor
    double t1 = model->cosmic_time(0.3);
    double t2 = model->cosmic_time(0.6);
    double t3 = model->cosmic_time(0.9);

    EXPECT_LT(t1, t2);
    EXPECT_LT(t2, t3);
}

TEST_F(CosmologyModelTest, AgeOfUniverse) {
    // Age at z=0 should be ~13.8 Gyr for Planck cosmology
    double age = model->cosmic_time(1.0);
    EXPECT_GT(age, 13.0);  // At least 13 Gyr
    EXPECT_LT(age, 14.5);  // Less than 14.5 Gyr
}

// Test: Power spectrum
TEST_F(CosmologyModelTest, PowerSpectrumPositive) {
    // Power spectrum should be positive for all k
    EXPECT_GT(model->power_spectrum(0.01, 0.0), 0.0);
    EXPECT_GT(model->power_spectrum(0.1, 0.0), 0.0);
    EXPECT_GT(model->power_spectrum(1.0, 0.0), 0.0);
}

TEST_F(CosmologyModelTest, PowerSpectrumScaleDependent) {
    // P(k) should have a peak around k ~ 0.02-0.05 h/Mpc
    double P_small_k = model->power_spectrum(0.01, 0.0);
    double P_peak = model->power_spectrum(0.05, 0.0);
    double P_large_k = model->power_spectrum(1.0, 0.0);

    EXPECT_GT(P_peak, P_small_k);
    EXPECT_GT(P_peak, P_large_k);
}

TEST_F(CosmologyModelTest, PowerSpectrumEvolution) {
    // P(k, z) should decrease with redshift (growth suppression)
    double k = 0.1;
    double P_z0 = model->power_spectrum(k, 0.0);
    double P_z1 = model->power_spectrum(k, 1.0);
    double P_z2 = model->power_spectrum(k, 2.0);

    EXPECT_GT(P_z0, P_z1);
    EXPECT_GT(P_z1, P_z2);
}

// Test: Matter density evolution
TEST_F(CosmologyModelTest, OmegaMatterEvolution) {
    // Omega_m should increase with redshift
    double Om_z0 = model->omega_matter_a(1.0);
    double Om_z1 = model->omega_matter_a(0.5);  // z=1

    EXPECT_NEAR(Om_z0, params.omega_m, 1e-6);  // Today
    EXPECT_GT(Om_z1, Om_z0);  // Higher at earlier times
}

// Test: Dark energy density evolution
TEST_F(CosmologyModelTest, OmegaLambdaEvolution) {
    // Omega_Lambda should decrease with redshift
    double OL_z0 = model->omega_lambda_a(1.0);
    double OL_z1 = model->omega_lambda_a(0.5);  // z=1

    EXPECT_NEAR(OL_z0, params.omega_lambda, 1e-6);  // Today
    EXPECT_LT(OL_z1, OL_z0);  // Lower at earlier times
}

// Test: Density sum
TEST_F(CosmologyModelTest, DensitySumUnity) {
    // Ωm + ΩΛ + Ωk should equal 1 for all redshifts
    for (double a = 0.2; a <= 1.0; a += 0.2) {
        double sum = model->omega_matter_a(a) + model->omega_lambda_a(a) + params.omega_k;
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}
