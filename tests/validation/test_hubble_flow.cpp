#include <gtest/gtest.h>
#include "core/simulation_engine.hpp"
#include <cmath>

using namespace core;

class HubbleFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simulation with many particles in a box
        SimulationBuilder builder;
        engine = builder
            .with_num_particles(1000)
            .with_box_size(100.0f)
            .with_time_step(0.01)
            .build();

        ASSERT_NE(engine, nullptr);
    }

    std::unique_ptr<SimulationEngine> engine;

    // Helper: Set up uniform distribution with Hubble flow velocities
    void setup_hubble_flow(double hubble_constant) {
        float* pos = const_cast<float*>(engine->get_positions());
        float* vel = const_cast<float*>(engine->get_velocities());
        float* masses = const_cast<float*>(engine->get_masses());

        size_t num_particles = 1000;
        float box_size = 100.0f;

        // Distribute particles uniformly
        for (size_t i = 0; i < num_particles; ++i) {
            // Position: uniform distribution
            pos[3*i + 0] = (i % 10) * (box_size / 10.0f);
            pos[3*i + 1] = ((i / 10) % 10) * (box_size / 10.0f);
            pos[3*i + 2] = ((i / 100) % 10) * (box_size / 10.0f);

            // Velocity: Hubble flow v = H * r (relative to box center)
            float cx = box_size / 2.0f;
            float cy = box_size / 2.0f;
            float cz = box_size / 2.0f;

            float rx = pos[3*i + 0] - cx;
            float ry = pos[3*i + 1] - cy;
            float rz = pos[3*i + 2] - cz;

            vel[3*i + 0] = hubble_constant * rx;
            vel[3*i + 1] = hubble_constant * ry;
            vel[3*i + 2] = hubble_constant * rz;

            // Uniform masses
            masses[i] = 1.0f;
        }
    }

    // Helper: Compute average distance from center
    double compute_average_distance() {
        const float* pos = engine->get_positions();
        size_t num_particles = 1000;
        float box_size = 100.0f;

        float cx = box_size / 2.0f;
        float cy = box_size / 2.0f;
        float cz = box_size / 2.0f;

        double total_distance = 0.0;
        for (size_t i = 0; i < num_particles; ++i) {
            float dx = pos[3*i + 0] - cx;
            float dy = pos[3*i + 1] - cy;
            float dz = pos[3*i + 2] - cz;

            total_distance += std::sqrt(dx*dx + dy*dy + dz*dz);
        }

        return total_distance / num_particles;
    }

    // Helper: Compute average velocity magnitude
    double compute_average_velocity() {
        const float* vel = engine->get_velocities();
        size_t num_particles = 1000;

        double total_velocity = 0.0;
        for (size_t i = 0; i < num_particles; ++i) {
            float vx = vel[3*i + 0];
            float vy = vel[3*i + 1];
            float vz = vel[3*i + 2];

            total_velocity += std::sqrt(vx*vx + vy*vy + vz*vz);
        }

        return total_velocity / num_particles;
    }
};

// Test: Free expansion follows Hubble's law (v = H * d)
TEST_F(HubbleFlowTest, HubbleLaw) {
    double H0 = 0.1;  // Hubble constant in simulation units
    setup_hubble_flow(H0);

    // Check that velocity follows v = H * d
    const float* pos = engine->get_positions();
    const float* vel = engine->get_velocities();
    float box_size = 100.0f;

    float cx = box_size / 2.0f;
    float cy = box_size / 2.0f;
    float cz = box_size / 2.0f;

    for (size_t i = 0; i < 100; ++i) {  // Check first 100 particles
        float dx = pos[3*i + 0] - cx;
        float dy = pos[3*i + 1] - cy;
        float dz = pos[3*i + 2] - cz;
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

        float vx = vel[3*i + 0];
        float vy = vel[3*i + 1];
        float vz = vel[3*i + 2];
        float velocity = std::sqrt(vx*vx + vy*vy + vz*vz);

        float expected_velocity = H0 * distance;

        if (distance > 1e-3f) {  // Skip particles very close to center
            EXPECT_NEAR(velocity, expected_velocity, 0.1f * expected_velocity);
        }
    }
}

// Test: Linear expansion (a(t) ∝ t for matter-free universe)
TEST_F(HubbleFlowTest, LinearExpansion) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    double distance_initial = compute_average_distance();
    double dt = 0.01;
    int steps = 100;

    // Run simulation
    for (int i = 0; i < steps; ++i) {
        engine->step();
    }

    double distance_final = compute_average_distance();

    // For free expansion: d(t) = d0 * (1 + H0*t)
    double time_elapsed = dt * steps;
    double expected_expansion_factor = 1.0 + H0 * time_elapsed;
    double actual_expansion_factor = distance_final / distance_initial;

    // Allow for some error due to self-gravity
    EXPECT_GT(actual_expansion_factor, expected_expansion_factor * 0.9);
    EXPECT_LT(actual_expansion_factor, expected_expansion_factor * 1.1);
}

// Test: Velocity remains proportional to distance during expansion
TEST_F(HubbleFlowTest, VelocityDistanceProportionality) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    // Run simulation for a while
    for (int i = 0; i < 50; ++i) {
        engine->step();
    }

    // Check that v/d ratio is approximately constant
    const float* pos = engine->get_positions();
    const float* vel = engine->get_velocities();
    float box_size = 100.0f;

    float cx = box_size / 2.0f;
    float cy = box_size / 2.0f;
    float cz = box_size / 2.0f;

    std::vector<double> ratios;
    for (size_t i = 0; i < 100; ++i) {
        float dx = pos[3*i + 0] - cx;
        float dy = pos[3*i + 1] - cy;
        float dz = pos[3*i + 2] - cz;
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

        float vx = vel[3*i + 0];
        float vy = vel[3*i + 1];
        float vz = vel[3*i + 2];
        float velocity = std::sqrt(vx*vx + vy*vy + vz*vz);

        if (distance > 1e-3f) {
            ratios.push_back(velocity / distance);
        }
    }

    // Compute mean and standard deviation
    double mean = 0.0;
    for (double r : ratios) {
        mean += r;
    }
    mean /= ratios.size();

    double std_dev = 0.0;
    for (double r : ratios) {
        std_dev += (r - mean) * (r - mean);
    }
    std_dev = std::sqrt(std_dev / ratios.size());

    // Standard deviation should be small (ratio approximately constant)
    EXPECT_LT(std_dev / mean, 0.2);  // Within 20% variation
}

// Test: Center of expansion remains stationary
TEST_F(HubbleFlowTest, CenterOfExpansionStationary) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    float3 COM_initial = engine->compute_center_of_mass();

    // Run simulation
    for (int i = 0; i < 100; ++i) {
        engine->step();
    }

    float3 COM_final = engine->compute_center_of_mass();

    // Center of mass should not move significantly
    EXPECT_NEAR(COM_final.x, COM_initial.x, 1.0f);
    EXPECT_NEAR(COM_final.y, COM_initial.y, 1.0f);
    EXPECT_NEAR(COM_final.z, COM_initial.z, 1.0f);
}

// Test: Momentum conservation (total momentum should be zero)
TEST_F(HubbleFlowTest, MomentumConservation) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    // Compute initial total momentum
    const float* vel = engine->get_velocities();
    const float* masses = engine->get_masses();
    size_t num_particles = 1000;

    float px_initial = 0.0f, py_initial = 0.0f, pz_initial = 0.0f;
    for (size_t i = 0; i < num_particles; ++i) {
        px_initial += masses[i] * vel[3*i + 0];
        py_initial += masses[i] * vel[3*i + 1];
        pz_initial += masses[i] * vel[3*i + 2];
    }

    // Run simulation
    for (int i = 0; i < 100; ++i) {
        engine->step();
    }

    // Compute final total momentum
    float px_final = 0.0f, py_final = 0.0f, pz_final = 0.0f;
    for (size_t i = 0; i < num_particles; ++i) {
        px_final += masses[i] * vel[3*i + 0];
        py_final += masses[i] * vel[3*i + 1];
        pz_final += masses[i] * vel[3*i + 2];
    }

    // Total momentum should remain small (approximately zero)
    EXPECT_NEAR(px_final, px_initial, 1.0f);
    EXPECT_NEAR(py_final, py_initial, 1.0f);
    EXPECT_NEAR(pz_final, pz_initial, 1.0f);
}

// Test: Expansion rate (Hubble parameter extraction)
TEST_F(HubbleFlowTest, ExpansionRateMeasurement) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    double distance_initial = compute_average_distance();
    double velocity_initial = compute_average_velocity();

    // Measured Hubble parameter: H = v / d
    double H_measured = velocity_initial / distance_initial;

    // Should match the initial Hubble constant
    EXPECT_NEAR(H_measured, H0, 0.01 * H0);
}

// Test: No gravitational collapse in low-density regime
TEST_F(HubbleFlowTest, NoCollapse) {
    // Set up very low density expansion
    double H0 = 0.2;  // High expansion rate
    setup_hubble_flow(H0);

    // Set very small masses to minimize self-gravity
    float* masses = const_cast<float*>(engine->get_masses());
    for (size_t i = 0; i < 1000; ++i) {
        masses[i] = 0.001f;
    }

    double distance_initial = compute_average_distance();

    // Run simulation
    for (int i = 0; i < 100; ++i) {
        engine->step();
    }

    double distance_final = compute_average_distance();

    // Distance should increase (no collapse)
    EXPECT_GT(distance_final, distance_initial);
}

// Test: Isotropy (expansion should be isotropic)
TEST_F(HubbleFlowTest, IsotropicExpansion) {
    double H0 = 0.1;
    setup_hubble_flow(H0);

    const float* pos = engine->get_positions();
    float box_size = 100.0f;
    float cx = box_size / 2.0f;
    float cy = box_size / 2.0f;
    float cz = box_size / 2.0f;

    // Compute average distance in each direction
    double sum_dx = 0.0, sum_dy = 0.0, sum_dz = 0.0;
    size_t num_particles = 1000;

    for (size_t i = 0; i < num_particles; ++i) {
        sum_dx += std::abs(pos[3*i + 0] - cx);
        sum_dy += std::abs(pos[3*i + 1] - cy);
        sum_dz += std::abs(pos[3*i + 2] - cz);
    }

    double avg_dx = sum_dx / num_particles;
    double avg_dy = sum_dy / num_particles;
    double avg_dz = sum_dz / num_particles;

    // All directions should have similar average distances
    EXPECT_NEAR(avg_dx, avg_dy, 0.2 * avg_dx);
    EXPECT_NEAR(avg_dy, avg_dz, 0.2 * avg_dy);
    EXPECT_NEAR(avg_dz, avg_dx, 0.2 * avg_dz);
}
