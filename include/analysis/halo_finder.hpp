#pragma once

#include "physics/lambda_cdm.hpp"
#include <vector>
#include <memory>
#include <unordered_set>

namespace analysis {

struct Halo {
    std::vector<int> particle_indices;
    float3 center_of_mass;
    float3 velocity_cm;
    float total_mass;
    float virial_radius;
    float virial_mass;
    int id;
    
    // Additional properties
    float max_circular_velocity;
    float3 angular_momentum;
    float concentration;
    float spin_parameter;
};

struct HaloStatistics {
    std::vector<Halo> halos;
    std::vector<float> mass_function;
    std::vector<float> mass_bins;
    float total_mass_in_halos;
    int total_particles_in_halos;
    float box_size;
};

class FriendsOfFriends {
private:
    float linking_length_;
    int min_particles_;
    float box_size_;
    bool periodic_boundaries_;
    
    // Grid for efficient neighbor finding
    int grid_size_;
    float cell_size_;
    std::vector<std::vector<int>> grid_cells_;

public:
    FriendsOfFriends(float linking_length, int min_particles = 20, bool periodic = true);
    ~FriendsOfFriends() = default;

    // Main halo finding function
    HaloStatistics find_halos(const std::vector<physics::Particle>& particles, float box_size);
    
    // Configuration
    void set_linking_length(float b) { linking_length_ = b; }
    void set_min_particles(int min_p) { min_particles_ = min_p; }
    void set_periodic_boundaries(bool periodic) { periodic_boundaries_ = periodic; }
    
    // Analysis functions
    std::vector<float> compute_mass_function(const std::vector<Halo>& halos,
                                           const std::vector<float>& mass_bins) const;
    void compute_halo_properties(Halo& halo, const std::vector<physics::Particle>& particles) const;
    
    // Utility functions
    float compute_virial_radius(float mass, float redshift = 0.0f) const;
    float compute_concentration(const Halo& halo, const std::vector<physics::Particle>& particles) const;

private:
    void initialize_grid(float box_size);
    void assign_particles_to_grid(const std::vector<physics::Particle>& particles);
    std::vector<int> find_neighbors(int particle_idx, const std::vector<physics::Particle>& particles) const;
    void union_find_clustering(const std::vector<physics::Particle>& particles,
                             std::vector<int>& cluster_ids);
    void extract_halos_from_clusters(const std::vector<physics::Particle>& particles,
                                   const std::vector<int>& cluster_ids,
                                   std::vector<Halo>& halos);
    
    float distance_with_periodic(const float3& pos1, const float3& pos2) const;
    int get_grid_index(const float3& position) const;
    std::vector<int> get_neighboring_cells(int cell_index) const;
};

class SphericalOverdensity {
private:
    float overdensity_threshold_;
    float background_density_;
    bool use_critical_density_;

public:
    SphericalOverdensity(float overdensity = 200.0f, bool critical = true);
    
    // Find SO halos using existing FoF groups as seeds
    HaloStatistics find_so_halos(const std::vector<physics::Particle>& particles,
                               const std::vector<Halo>& fof_halos,
                               float box_size);
    
    void set_overdensity_threshold(float delta) { overdensity_threshold_ = delta; }
    void set_background_density(float rho_bg) { background_density_ = rho_bg; }

private:
    float compute_enclosed_mass(const float3& center, float radius,
                              const std::vector<physics::Particle>& particles) const;
    float find_virial_radius(const float3& center, const std::vector<physics::Particle>& particles,
                           float initial_guess = 1.0f) const;
};

// Utility functions for halo analysis
namespace halo_utils {
    // NFW profile fitting
    struct NFWProfile {
        float rs;           // Scale radius
        float rho_s;        // Scale density
        float concentration;
        float virial_radius;
        float virial_mass;
    };
    
    NFWProfile fit_nfw_profile(const Halo& halo, const std::vector<physics::Particle>& particles);
    
    // Halo merger tree construction
    struct HaloMergerNode {
        int halo_id;
        float mass;
        float redshift;
        std::vector<int> progenitor_ids;
        int descendant_id;
    };
    
    std::vector<HaloMergerNode> build_merger_tree(const std::vector<HaloStatistics>& halo_catalogs,
                                                  const std::vector<float>& redshifts);
    
    // Mass function comparisons
    float sheth_tormen_mass_function(float mass, float sigma, float delta_c = 1.686f);
    float press_schechter_mass_function(float mass, float sigma, float delta_c = 1.686f);
    
    // Bias calculations
    float linear_bias(float mass, float sigma, float delta_c = 1.686f);
    
    // Halo occupation distributions
    float mean_occupation_centrals(float mass, float M_min, float sigma_log_M);
    float mean_occupation_satellites(float mass, float M_min, float M_1, float alpha);
}

}