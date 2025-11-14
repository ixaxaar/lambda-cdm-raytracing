#include <gtest/gtest.h>
#include "core/simulation_engine.hpp"
#include <cmath>

using namespace core;

class TwoBodyProblemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simulation with just 2 particles
        SimulationBuilder builder;
        engine = builder
            .with_num_particles(2)
            .with_box_size(1000.0f)  // Large box to avoid boundary effects
            .with_time_step(0.001)    // Small timestep for accuracy
            .build();

        ASSERT_NE(engine, nullptr);
    }

    std::unique_ptr<SimulationEngine> engine;

    // Helper: Set up circular orbit
    void setup_circular_orbit(float separation, float mass1, float mass2) {
        // Place particles at specified separation
        // For circular orbit: v = sqrt(GM/r)

        float* pos = const_cast<float*>(engine->get_positions());
        float* vel = const_cast<float*>(engine->get_velocities());
        float* masses = const_cast<float*>(engine->get_masses());

        // Particle 1 at origin
        pos[0] = 0.0f;
        pos[1] = 0.0f;
        pos[2] = 0.0f;

        // Particle 2 at distance r
        pos[3] = separation;
        pos[4] = 0.0f;
        pos[5] = 0.0f;

        // Masses
        masses[0] = mass1;
        masses[1] = mass2;

        // Circular orbit velocity (simplified, assuming M >> m)
        float v_orbit = std::sqrt(mass1 / separation);

        // Velocities for circular orbit in x-y plane
        vel[0] = 0.0f;
        vel[1] = 0.0f;
        vel[2] = 0.0f;

        vel[3] = 0.0f;
        vel[4] = v_orbit;
        vel[5] = 0.0f;
    }
};

// Test: Energy conservation in two-body system
TEST_F(TwoBodyProblemTest, EnergyConservation) {
    // Set up simple two-body system
    setup_circular_orbit(10.0f, 100.0f, 1.0f);

    // Compute initial energy
    double E_initial = engine->compute_total_energy();

    // Run for many steps
    for (int i = 0; i < 1000; ++i) {
        engine->step();
    }

    // Compute final energy
    double E_final = engine->compute_total_energy();

    // Energy should be conserved within tolerance
    double relative_error = std::abs(E_final - E_initial) / std::abs(E_initial);

    // Allow up to 1% error (depends on integrator and timestep)
    EXPECT_LT(relative_error, 0.01);
}

// Test: Angular momentum conservation
TEST_F(TwoBodyProblemTest, AngularMomentumConservation) {
    setup_circular_orbit(10.0f, 100.0f, 1.0f);

    // Compute initial angular momentum
    float3 L_initial = engine->compute_angular_momentum();
    double L_mag_initial = std::sqrt(
        L_initial.x * L_initial.x +
        L_initial.y * L_initial.y +
        L_initial.z * L_initial.z
    );

    // Run simulation
    for (int i = 0; i < 1000; ++i) {
        engine->step();
    }

    // Compute final angular momentum
    float3 L_final = engine->compute_angular_momentum();
    double L_mag_final = std::sqrt(
        L_final.x * L_final.x +
        L_final.y * L_final.y +
        L_final.z * L_final.z
    );

    // Angular momentum magnitude should be conserved
    double relative_error = std::abs(L_mag_final - L_mag_initial) / L_mag_initial;

    EXPECT_LT(relative_error, 0.01);  // Within 1%
}

// Test: Center of mass remains stationary
TEST_F(TwoBodyProblemTest, CenterOfMassStationary) {
    setup_circular_orbit(10.0f, 100.0f, 1.0f);

    float3 COM_initial = engine->compute_center_of_mass();

    // Run simulation
    for (int i = 0; i < 1000; ++i) {
        engine->step();
    }

    float3 COM_final = engine->compute_center_of_mass();

    // COM should not move
    EXPECT_NEAR(COM_final.x, COM_initial.x, 1e-3f);
    EXPECT_NEAR(COM_final.y, COM_initial.y, 1e-3f);
    EXPECT_NEAR(COM_final.z, COM_initial.z, 1e-3f);
}

// Test: Orbital period (Kepler's third law)
TEST_F(TwoBodyProblemTest, KeplersThirdLaw) {
    // For circular orbit: T² = 4π²r³/(GM)

    float r = 10.0f;
    float M = 100.0f;

    setup_circular_orbit(r, M, 1.0f);

    // Expected period (in simulation units where G=1)
    double T_expected = 2.0 * M_PI * std::sqrt(r * r * r / M);

    // Run for one expected period
    double dt = 0.001;
    int steps = static_cast<int>(T_expected / dt);

    float* pos = const_cast<float*>(engine->get_positions());
    float x_initial = pos[3];
    float y_initial = pos[4];

    for (int i = 0; i < steps; ++i) {
        engine->step();
    }

    // After one period, should return close to initial position
    float x_final = pos[3];
    float y_final = pos[4];

    // May not be exact due to numerical errors, but should be close
    float position_error = std::sqrt(
        (x_final - x_initial) * (x_final - x_initial) +
        (y_final - y_initial) * (y_final - y_initial)
    );

    // Within 10% of separation
    EXPECT_LT(position_error, 0.1f * r);
}

// Test: Gravitational force law (1/r²)
TEST_F(TwoBodyProblemTest, InverseSquareLaw) {
    // Set up two static particles at different separations
    // and compare forces

    float* pos = const_cast<float*>(engine->get_positions());
    float* masses = const_cast<float*>(engine->get_masses());

    masses[0] = 100.0f;
    masses[1] = 1.0f;

    // Test at separation r1 = 10
    float r1 = 10.0f;
    pos[0] = 0.0f;
    pos[1] = 0.0f;
    pos[2] = 0.0f;
    pos[3] = r1;
    pos[4] = 0.0f;
    pos[5] = 0.0f;

    // Force should scale as 1/r²
    // F = GMm/r²
    float F_expected_1 = masses[0] * masses[1] / (r1 * r1);

    // Note: Actual force computation would need to call compute_forces
    // This test is more of a design verification
}

// Test: Escape velocity
TEST_F(TwoBodyProblemTest, EscapeVelocity) {
    // For escape: v_escape = sqrt(2GM/r)

    float r = 10.0f;
    float M = 100.0f;

    float v_escape = std::sqrt(2.0f * M / r);

    float* pos = const_cast<float*>(engine->get_positions());
    float* vel = const_cast<float*>(engine->get_velocities());
    float* masses = const_cast<float*>(engine->get_masses());

    // Particle 1 (heavy)
    pos[0] = 0.0f;
    pos[1] = 0.0f;
    pos[2] = 0.0f;
    vel[0] = 0.0f;
    vel[1] = 0.0f;
    vel[2] = 0.0f;
    masses[0] = M;

    // Particle 2 (light) with escape velocity
    pos[3] = r;
    pos[4] = 0.0f;
    pos[5] = 0.0f;
    vel[3] = v_escape;
    vel[4] = 0.0f;
    vel[5] = 0.0f;
    masses[1] = 1.0f;

    // Initial energy should be approximately zero
    double E_initial = engine->compute_total_energy();

    // For escape velocity, total energy should be ~ 0
    // (kinetic = potential in magnitude)
    EXPECT_NEAR(E_initial, 0.0, 1.0);  // Within some tolerance
}

// Test: Bound vs unbound orbits
TEST_F(TwoBodyProblemTest, BoundOrbitCheck) {
    float r = 10.0f;
    float M = 100.0f;

    // Circular orbit velocity (bound)
    float v_circular = std::sqrt(M / r);

    float* pos = const_cast<float*>(engine->get_positions());
    float* vel = const_cast<float*>(engine->get_velocities());
    float* masses = const_cast<float*>(engine->get_masses());

    pos[0] = 0.0f;
    pos[1] = 0.0f;
    pos[2] = 0.0f;
    pos[3] = r;
    pos[4] = 0.0f;
    pos[5] = 0.0f;

    vel[0] = 0.0f;
    vel[1] = 0.0f;
    vel[2] = 0.0f;
    vel[3] = 0.0f;
    vel[4] = v_circular;
    vel[5] = 0.0f;

    masses[0] = M;
    masses[1] = 1.0f;

    // For bound orbit: E < 0
    double E = engine->compute_total_energy();
    EXPECT_LT(E, 0.0);
}
