#include "analysis/halo_finder.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <queue>
#include <numeric>

namespace analysis {

FriendsOfFriends::FriendsOfFriends(float linking_length, int min_particles, bool periodic)
    : linking_length_(linking_length), min_particles_(min_particles), 
      periodic_boundaries_(periodic), box_size_(0.0f) {
}

HaloStatistics FriendsOfFriends::find_halos(const std::vector<physics::Particle>& particles, float box_size) {
    box_size_ = box_size;
    
    // Initialize spatial grid for efficient neighbor finding
    initialize_grid(box_size);
    assign_particles_to_grid(particles);
    
    // Perform union-find clustering
    std::vector<int> cluster_ids(particles.size(), -1);
    union_find_clustering(particles, cluster_ids);
    
    // Extract halos from clusters
    HaloStatistics stats;
    stats.box_size = box_size;
    extract_halos_from_clusters(particles, cluster_ids, stats.halos);
    
    // Compute halo properties
    for (auto& halo : stats.halos) {
        compute_halo_properties(halo, particles);
    }
    
    // Compute statistics
    stats.total_particles_in_halos = 0;
    stats.total_mass_in_halos = 0.0f;
    for (const auto& halo : stats.halos) {
        stats.total_particles_in_halos += halo.particle_indices.size();
        stats.total_mass_in_halos += halo.total_mass;
    }
    
    return stats;
}

void FriendsOfFriends::initialize_grid(float box_size) {
    // Choose grid size to have roughly linking_length_ sized cells
    grid_size_ = static_cast<int>(box_size / linking_length_);
    grid_size_ = std::max(grid_size_, 1);
    
    cell_size_ = box_size / grid_size_;
    grid_cells_.clear();
    grid_cells_.resize(grid_size_ * grid_size_ * grid_size_);
}

void FriendsOfFriends::assign_particles_to_grid(const std::vector<physics::Particle>& particles) {
    // Clear existing assignments
    for (auto& cell : grid_cells_) {
        cell.clear();
    }
    
    // Assign particles to grid cells
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        int cell_idx = get_grid_index(particles[i].position);
        if (cell_idx >= 0 && cell_idx < static_cast<int>(grid_cells_.size())) {
            grid_cells_[cell_idx].push_back(i);
        }
    }
}

int FriendsOfFriends::get_grid_index(const float3& position) const {
    int ix = static_cast<int>(position.x / cell_size_);
    int iy = static_cast<int>(position.y / cell_size_);
    int iz = static_cast<int>(position.z / cell_size_);
    
    // Handle periodic boundaries
    if (periodic_boundaries_) {
        ix = ((ix % grid_size_) + grid_size_) % grid_size_;
        iy = ((iy % grid_size_) + grid_size_) % grid_size_;
        iz = ((iz % grid_size_) + grid_size_) % grid_size_;
    } else {
        if (ix < 0 || ix >= grid_size_ || 
            iy < 0 || iy >= grid_size_ || 
            iz < 0 || iz >= grid_size_) {
            return -1; // Outside grid
        }
    }
    
    return ix * grid_size_ * grid_size_ + iy * grid_size_ + iz;
}

std::vector<int> FriendsOfFriends::get_neighboring_cells(int cell_index) const {
    std::vector<int> neighbors;
    
    // Convert linear index to 3D coordinates
    int ix = cell_index / (grid_size_ * grid_size_);
    int iy = (cell_index / grid_size_) % grid_size_;
    int iz = cell_index % grid_size_;
    
    // Check all 27 neighboring cells (including self)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;
                
                if (periodic_boundaries_) {
                    nx = ((nx % grid_size_) + grid_size_) % grid_size_;
                    ny = ((ny % grid_size_) + grid_size_) % grid_size_;
                    nz = ((nz % grid_size_) + grid_size_) % grid_size_;
                } else {
                    if (nx < 0 || nx >= grid_size_ || 
                        ny < 0 || ny >= grid_size_ || 
                        nz < 0 || nz >= grid_size_) {
                        continue;
                    }
                }
                
                int neighbor_idx = nx * grid_size_ * grid_size_ + ny * grid_size_ + nz;
                neighbors.push_back(neighbor_idx);
            }
        }
    }
    
    return neighbors;
}

float FriendsOfFriends::distance_with_periodic(const float3& pos1, const float3& pos2) const {
    float dx = pos2.x - pos1.x;
    float dy = pos2.y - pos1.y;
    float dz = pos2.z - pos1.z;
    
    if (periodic_boundaries_) {
        // Apply minimum image convention
        if (dx > box_size_ * 0.5f) dx -= box_size_;
        if (dx < -box_size_ * 0.5f) dx += box_size_;
        if (dy > box_size_ * 0.5f) dy -= box_size_;
        if (dy < -box_size_ * 0.5f) dy += box_size_;
        if (dz > box_size_ * 0.5f) dz -= box_size_;
        if (dz < -box_size_ * 0.5f) dz += box_size_;
    }
    
    return sqrt(dx*dx + dy*dy + dz*dz);
}

std::vector<int> FriendsOfFriends::find_neighbors(int particle_idx, 
                                                const std::vector<physics::Particle>& particles) const {
    std::vector<int> neighbors;
    
    int cell_idx = get_grid_index(particles[particle_idx].position);
    if (cell_idx < 0) return neighbors;
    
    auto neighboring_cells = get_neighboring_cells(cell_idx);
    
    for (int cell : neighboring_cells) {
        if (cell < 0 || cell >= static_cast<int>(grid_cells_.size())) continue;
        
        for (int other_idx : grid_cells_[cell]) {
            if (other_idx == particle_idx) continue;
            
            float dist = distance_with_periodic(particles[particle_idx].position,
                                               particles[other_idx].position);
            
            if (dist <= linking_length_) {
                neighbors.push_back(other_idx);
            }
        }
    }
    
    return neighbors;
}

void FriendsOfFriends::union_find_clustering(const std::vector<physics::Particle>& particles,
                                           std::vector<int>& cluster_ids) {
    int num_particles = static_cast<int>(particles.size());
    
    // Initialize each particle as its own cluster
    std::vector<int> parent(num_particles);
    std::iota(parent.begin(), parent.end(), 0);
    
    // Union-find with path compression
    auto find_root = [&](int x) -> int {
        if (parent[x] != x) {
            parent[x] = parent[parent[x]]; // Path compression
        }
        return parent[x];
    };
    
    auto union_clusters = [&](int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_y] = root_x;
        }
    };
    
    // Find all neighboring pairs and union them
    for (int i = 0; i < num_particles; ++i) {
        auto neighbors = find_neighbors(i, particles);
        for (int neighbor : neighbors) {
            union_clusters(i, neighbor);
        }
    }
    
    // Assign final cluster IDs
    for (int i = 0; i < num_particles; ++i) {
        cluster_ids[i] = find_root(i);
    }
}

void FriendsOfFriends::extract_halos_from_clusters(const std::vector<physics::Particle>& particles,
                                                  const std::vector<int>& cluster_ids,
                                                  std::vector<Halo>& halos) {
    // Group particles by cluster ID
    std::unordered_map<int, std::vector<int>> clusters;
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        clusters[cluster_ids[i]].push_back(i);
    }
    
    // Create halos from clusters that meet minimum particle requirement
    int halo_id = 0;
    for (const auto& [cluster_id, particle_list] : clusters) {
        if (static_cast<int>(particle_list.size()) >= min_particles_) {
            Halo halo;
            halo.id = halo_id++;
            halo.particle_indices = particle_list;
            halos.push_back(halo);
        }
    }
}

void FriendsOfFriends::compute_halo_properties(Halo& halo, 
                                              const std::vector<physics::Particle>& particles) const {
    if (halo.particle_indices.empty()) return;
    
    // Compute center of mass and total mass
    float3 com = make_float3(0.0f, 0.0f, 0.0f);
    float3 vel_cm = make_float3(0.0f, 0.0f, 0.0f);
    float total_mass = 0.0f;
    
    for (int idx : halo.particle_indices) {
        const auto& particle = particles[idx];
        float mass = particle.mass;
        
        com.x += mass * particle.position.x;
        com.y += mass * particle.position.y;
        com.z += mass * particle.position.z;
        
        vel_cm.x += mass * particle.velocity.x;
        vel_cm.y += mass * particle.velocity.y;
        vel_cm.z += mass * particle.velocity.z;
        
        total_mass += mass;
    }
    
    if (total_mass > 0.0f) {
        com.x /= total_mass;
        com.y /= total_mass;
        com.z /= total_mass;
        
        vel_cm.x /= total_mass;
        vel_cm.y /= total_mass;
        vel_cm.z /= total_mass;
    }
    
    halo.center_of_mass = com;
    halo.velocity_cm = vel_cm;
    halo.total_mass = total_mass;
    
    // Compute virial radius (rough estimate)
    halo.virial_radius = compute_virial_radius(total_mass);
    halo.virial_mass = total_mass; // For FoF, virial mass = total mass
    
    // Compute maximum circular velocity (simplified)
    float max_v_circ = 0.0f;
    for (int idx : halo.particle_indices) {
        const auto& particle = particles[idx];
        float3 rel_pos = make_float3(
            particle.position.x - com.x,
            particle.position.y - com.y,
            particle.position.z - com.z
        );
        float r = sqrt(rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z);
        
        if (r > 0.0f) {
            // Simplified circular velocity assuming enclosed mass
            float v_circ = sqrt(total_mass / r); // In simulation units
            max_v_circ = std::max(max_v_circ, v_circ);
        }
    }
    halo.max_circular_velocity = max_v_circ;
    
    // Compute angular momentum
    float3 L = make_float3(0.0f, 0.0f, 0.0f);
    for (int idx : halo.particle_indices) {
        const auto& particle = particles[idx];
        float3 rel_pos = make_float3(
            particle.position.x - com.x,
            particle.position.y - com.y,
            particle.position.z - com.z
        );
        float3 rel_vel = make_float3(
            particle.velocity.x - vel_cm.x,
            particle.velocity.y - vel_cm.y,
            particle.velocity.z - vel_cm.z
        );
        
        // L = r × mv
        L.x += particle.mass * (rel_pos.y * rel_vel.z - rel_pos.z * rel_vel.y);
        L.y += particle.mass * (rel_pos.z * rel_vel.x - rel_pos.x * rel_vel.z);
        L.z += particle.mass * (rel_pos.x * rel_vel.y - rel_pos.y * rel_vel.x);
    }
    halo.angular_momentum = L;
    
    // Compute spin parameter (simplified)
    float L_mag = sqrt(L.x*L.x + L.y*L.y + L.z*L.z);
    if (total_mass > 0.0f && halo.virial_radius > 0.0f) {
        halo.spin_parameter = L_mag / (sqrt(2.0f) * total_mass * max_v_circ * halo.virial_radius);
    } else {
        halo.spin_parameter = 0.0f;
    }
}

float FriendsOfFriends::compute_virial_radius(float mass, float redshift) const {
    // Simple estimate based on virial theorem
    // R_vir = (3 * M / (4 * pi * Delta * rho_crit))^(1/3)
    
    float Delta = 200.0f; // Overdensity parameter
    float rho_crit = 1.0f; // Critical density in simulation units
    
    float volume = mass / (Delta * rho_crit);
    float radius = pow(3.0f * volume / (4.0f * M_PI), 1.0f/3.0f);
    
    return radius;
}

std::vector<float> FriendsOfFriends::compute_mass_function(const std::vector<Halo>& halos,
                                                         const std::vector<float>& mass_bins) const {
    std::vector<float> mass_function(mass_bins.size() - 1, 0.0f);
    
    for (const auto& halo : halos) {
        // Find which mass bin this halo belongs to
        for (size_t i = 0; i < mass_bins.size() - 1; ++i) {
            if (halo.total_mass >= mass_bins[i] && halo.total_mass < mass_bins[i+1]) {
                mass_function[i] += 1.0f;
                break;
            }
        }
    }
    
    // Normalize by bin width and volume
    float volume = box_size_ * box_size_ * box_size_;
    for (size_t i = 0; i < mass_function.size(); ++i) {
        float bin_width = mass_bins[i+1] - mass_bins[i];
        mass_function[i] /= (bin_width * volume);
    }
    
    return mass_function;
}

// Spherical Overdensity implementation
SphericalOverdensity::SphericalOverdensity(float overdensity, bool critical)
    : overdensity_threshold_(overdensity), use_critical_density_(critical) {
}

HaloStatistics SphericalOverdensity::find_so_halos(const std::vector<physics::Particle>& particles,
                                                  const std::vector<Halo>& fof_halos,
                                                  float box_size) {
    HaloStatistics stats;
    stats.box_size = box_size;
    
    // Use FoF halos as seeds for SO analysis
    for (const auto& fof_halo : fof_halos) {
        // Find virial radius for this halo
        float virial_radius = find_virial_radius(fof_halo.center_of_mass, particles, 
                                                fof_halo.virial_radius);
        
        if (virial_radius > 0.0f) {
            Halo so_halo = fof_halo; // Copy basic properties
            so_halo.virial_radius = virial_radius;
            
            // Recompute mass within virial radius
            so_halo.virial_mass = compute_enclosed_mass(fof_halo.center_of_mass, 
                                                       virial_radius, particles);
            
            stats.halos.push_back(so_halo);
        }
    }
    
    return stats;
}

float SphericalOverdensity::compute_enclosed_mass(const float3& center, float radius,
                                                const std::vector<physics::Particle>& particles) const {
    float enclosed_mass = 0.0f;
    
    for (const auto& particle : particles) {
        float3 rel_pos = make_float3(
            particle.position.x - center.x,
            particle.position.y - center.y,
            particle.position.z - center.z
        );
        float r = sqrt(rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z);
        
        if (r <= radius) {
            enclosed_mass += particle.mass;
        }
    }
    
    return enclosed_mass;
}

float SphericalOverdensity::find_virial_radius(const float3& center, 
                                             const std::vector<physics::Particle>& particles,
                                             float initial_guess) const {
    float radius = initial_guess;
    const float tolerance = 0.01f;
    const int max_iterations = 50;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        float enclosed_mass = compute_enclosed_mass(center, radius, particles);
        float volume = (4.0f/3.0f) * M_PI * radius * radius * radius;
        float density = enclosed_mass / volume;
        
        float target_density = overdensity_threshold_ * background_density_;
        
        if (abs(density - target_density) / target_density < tolerance) {
            return radius;
        }
        
        // Simple bisection-like adjustment
        if (density > target_density) {
            radius *= 0.9f;
        } else {
            radius *= 1.1f;
        }
        
        if (radius <= 0.0f) return 0.0f;
    }
    
    return radius; // Return best estimate
}

}