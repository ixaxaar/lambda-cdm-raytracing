#include "mpi/cluster_comm.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>

namespace mpi {

ClusterCommunicator::ClusterCommunicator(MPI_Comm comm, float ghost_width)
    : comm_(comm), ghost_zone_width_(ghost_width), box_size_(0.0f) {
    MPI_Comm_rank(comm_, &domain_.rank);
    MPI_Comm_size(comm_, &domain_.size);
    dims_[0] = dims_[1] = dims_[2] = 0;
}

ClusterCommunicator::~ClusterCommunicator() {
    // MPI_Finalize should be called by the application
}

bool ClusterCommunicator::initialize(float box_size) {
    box_size_ = box_size;
    decompose_domain(box_size);
    return true;
}

void ClusterCommunicator::decompose_domain(float box_size) {
    box_size_ = box_size;
    // Simple 3D domain decomposition
    dims_[0] = dims_[1] = dims_[2] = 0;
    MPI_Dims_create(domain_.size, 3, dims_);
    
    int dims[3] = {dims_[0], dims_[1], dims_[2]};
    
    int coords[3];
    int rank_temp = domain_.rank;
    coords[2] = rank_temp % dims[2];
    rank_temp /= dims[2];
    coords[1] = rank_temp % dims[1];
    coords[0] = rank_temp / dims[1];
    
    // Calculate local domain boundaries
    float dx = box_size / dims[0];
    float dy = box_size / dims[1];
    float dz = box_size / dims[2];
    
    domain_.local_box_min = make_float3(
        coords[0] * dx,
        coords[1] * dy,
        coords[2] * dz
    );
    
    domain_.local_box_max = make_float3(
        (coords[0] + 1) * dx,
        (coords[1] + 1) * dy,
        (coords[2] + 1) * dz
    );
    
    domain_.local_box_size = make_float3(dx, dy, dz);
    
    // Find neighbor ranks
    domain_.neighbor_ranks.clear();
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                
                int neighbor_coords[3] = {
                    coords[0] + i,
                    coords[1] + j,
                    coords[2] + k
                };
                
                // Periodic boundary conditions
                if (neighbor_coords[0] < 0) neighbor_coords[0] += dims[0];
                if (neighbor_coords[0] >= dims[0]) neighbor_coords[0] -= dims[0];
                if (neighbor_coords[1] < 0) neighbor_coords[1] += dims[1];
                if (neighbor_coords[1] >= dims[1]) neighbor_coords[1] -= dims[1];
                if (neighbor_coords[2] < 0) neighbor_coords[2] += dims[2];
                if (neighbor_coords[2] >= dims[2]) neighbor_coords[2] -= dims[2];
                
                int neighbor_rank = neighbor_coords[0] * dims[1] * dims[2] +
                                  neighbor_coords[1] * dims[2] +
                                  neighbor_coords[2];
                
                domain_.neighbor_ranks.push_back(neighbor_rank);
            }
        }
    }
}

void ClusterCommunicator::exchange_particles(std::vector<physics::Particle>& particles) {
    // Separate particles into local and export lists
    std::vector<std::vector<physics::Particle>> export_lists(domain_.size);
    std::vector<physics::Particle> local_particles;
    
    for (const auto& particle : particles) {
        if (is_particle_local(particle)) {
            local_particles.push_back(particle);
        } else {
            // Determine which rank should own this particle based on position
            int target_rank = find_owner_rank(particle.position);
            if (target_rank >= 0 && target_rank < domain_.size) {
                export_lists[target_rank].push_back(particle);
            }
        }
    }
    
    // Exchange particle counts
    std::vector<int> send_counts(domain_.size, 0);
    std::vector<int> recv_counts(domain_.size, 0);
    
    for (int rank = 0; rank < domain_.size; ++rank) {
        send_counts[rank] = export_lists[rank].size();
    }
    
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm_);
    
    // Exchange particles
    std::vector<int> send_displs(domain_.size, 0);
    std::vector<int> recv_displs(domain_.size, 0);
    
    for (int rank = 1; rank < domain_.size; ++rank) {
        send_displs[rank] = send_displs[rank-1] + send_counts[rank-1];
        recv_displs[rank] = recv_displs[rank-1] + recv_counts[rank-1];
    }
    
    int total_send = send_displs[domain_.size-1] + send_counts[domain_.size-1];
    int total_recv = recv_displs[domain_.size-1] + recv_counts[domain_.size-1];
    
    send_buffer_.resize(total_send);
    recv_buffer_.resize(total_recv);
    
    // Pack send buffer
    int offset = 0;
    for (int rank = 0; rank < domain_.size; ++rank) {
        std::copy(export_lists[rank].begin(), export_lists[rank].end(),
                 send_buffer_.begin() + offset);
        offset += export_lists[rank].size();
    }
    
    // Use MPI_BYTE for simplicity - in production, custom MPI datatype would be better
    size_t particle_size = sizeof(physics::Particle);
    
    // Perform the exchange using MPI_BYTE
    std::vector<int> send_byte_counts(domain_.size);
    std::vector<int> recv_byte_counts(domain_.size);
    std::vector<int> send_byte_displs(domain_.size);
    std::vector<int> recv_byte_displs(domain_.size);
    
    for (int i = 0; i < domain_.size; ++i) {
        send_byte_counts[i] = send_counts[i] * particle_size;
        recv_byte_counts[i] = recv_counts[i] * particle_size;
        send_byte_displs[i] = send_displs[i] * particle_size;
        recv_byte_displs[i] = recv_displs[i] * particle_size;
    }
    
    MPI_Alltoallv(send_buffer_.data(), send_byte_counts.data(), send_byte_displs.data(), MPI_BYTE,
                  recv_buffer_.data(), recv_byte_counts.data(), recv_byte_displs.data(), MPI_BYTE,
                  comm_);
    
    // Update particle list
    particles = local_particles;
    particles.insert(particles.end(), recv_buffer_.begin(), recv_buffer_.end());
}

void ClusterCommunicator::exchange_ghost_particles(const std::vector<physics::Particle>& particles) {
    ghost_particles_.clear();
    
    // Find particles in ghost zones
    std::vector<std::vector<physics::Particle>> ghost_exports(domain_.neighbor_ranks.size());
    
    for (const auto& particle : particles) {
        if (is_particle_ghost(particle)) {
            // Send to all relevant neighbors
            for (size_t i = 0; i < domain_.neighbor_ranks.size(); ++i) {
                ghost_exports[i].push_back(particle);
            }
        }
    }
    
    // Exchange ghost particles with neighbors
    for (size_t i = 0; i < domain_.neighbor_ranks.size(); ++i) {
        int neighbor_rank = domain_.neighbor_ranks[i];
        int send_count = ghost_exports[i].size();
        int recv_count;
        
        // Exchange counts
        MPI_Sendrecv(&send_count, 1, MPI_INT, neighbor_rank, 0,
                     &recv_count, 1, MPI_INT, neighbor_rank, 0,
                     comm_, MPI_STATUS_IGNORE);
        
        if (recv_count > 0) {
            std::vector<physics::Particle> recv_ghosts(recv_count);
            
            // Exchange particles
            MPI_Sendrecv(ghost_exports[i].data(), send_count * sizeof(physics::Particle), MPI_BYTE,
                        neighbor_rank, 1,
                        recv_ghosts.data(), recv_count * sizeof(physics::Particle), MPI_BYTE,
                        neighbor_rank, 1,
                        comm_, MPI_STATUS_IGNORE);
            
            ghost_particles_.insert(ghost_particles_.end(), 
                                  recv_ghosts.begin(), recv_ghosts.end());
        }
    }
}

void ClusterCommunicator::all_reduce_forces(std::vector<float3>& forces) {
    // Reduce forces across all processes
    std::vector<float3> reduced_forces(forces.size());
    
    MPI_Allreduce(forces.data(), reduced_forces.data(), 
                  forces.size() * 3, MPI_FLOAT, MPI_SUM, comm_);
    
    forces = std::move(reduced_forces);
}

void ClusterCommunicator::gather_all_particles(const std::vector<physics::Particle>& local_particles,
                                             std::vector<physics::Particle>& all_particles) {
    // Gather particle counts from all processes
    std::vector<int> counts(domain_.size);
    int local_count = local_particles.size();
    
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);
    
    // Calculate displacements
    std::vector<int> displs(domain_.size, 0);
    for (int i = 1; i < domain_.size; ++i) {
        displs[i] = displs[i-1] + counts[i-1];
    }
    
    int total_particles = displs[domain_.size-1] + counts[domain_.size-1];
    all_particles.resize(total_particles);
    
    // Convert counts and displacements to bytes
    std::vector<int> byte_counts(domain_.size);
    std::vector<int> byte_displs(domain_.size);
    
    for (int i = 0; i < domain_.size; ++i) {
        byte_counts[i] = counts[i] * sizeof(physics::Particle);
        byte_displs[i] = displs[i] * sizeof(physics::Particle);
    }
    
    // Gather all particles
    MPI_Allgatherv(local_particles.data(), local_count * sizeof(physics::Particle), MPI_BYTE,
                   all_particles.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE, comm_);
}

bool ClusterCommunicator::is_particle_local(const physics::Particle& particle) const {
    return (particle.position.x >= domain_.local_box_min.x &&
            particle.position.x < domain_.local_box_max.x &&
            particle.position.y >= domain_.local_box_min.y &&
            particle.position.y < domain_.local_box_max.y &&
            particle.position.z >= domain_.local_box_min.z &&
            particle.position.z < domain_.local_box_max.z);
}

bool ClusterCommunicator::is_particle_ghost(const physics::Particle& particle) const {
    float3 expanded_min = make_float3(
        domain_.local_box_min.x - ghost_zone_width_,
        domain_.local_box_min.y - ghost_zone_width_,
        domain_.local_box_min.z - ghost_zone_width_
    );
    
    float3 expanded_max = make_float3(
        domain_.local_box_max.x + ghost_zone_width_,
        domain_.local_box_max.y + ghost_zone_width_,
        domain_.local_box_max.z + ghost_zone_width_
    );
    
    return (particle.position.x >= expanded_min.x &&
            particle.position.x < expanded_max.x &&
            particle.position.y >= expanded_min.y &&
            particle.position.y < expanded_max.y &&
            particle.position.z >= expanded_min.z &&
            particle.position.z < expanded_max.z) &&
           !is_particle_local(particle);
}

int ClusterCommunicator::find_owner_rank(const float3& position) const {
    if (dims_[0] == 0 || dims_[1] == 0 || dims_[2] == 0 || box_size_ <= 0.0f) {
        return -1; // Invalid state
    }
    
    // Handle periodic boundary conditions
    float3 wrapped_pos = position;
    while (wrapped_pos.x < 0.0f) wrapped_pos.x += box_size_;
    while (wrapped_pos.x >= box_size_) wrapped_pos.x -= box_size_;
    while (wrapped_pos.y < 0.0f) wrapped_pos.y += box_size_;
    while (wrapped_pos.y >= box_size_) wrapped_pos.y -= box_size_;
    while (wrapped_pos.z < 0.0f) wrapped_pos.z += box_size_;
    while (wrapped_pos.z >= box_size_) wrapped_pos.z -= box_size_;
    
    // Calculate domain indices
    float dx = box_size_ / dims_[0];
    float dy = box_size_ / dims_[1];
    float dz = box_size_ / dims_[2];
    
    int coord_x = static_cast<int>(wrapped_pos.x / dx);
    int coord_y = static_cast<int>(wrapped_pos.y / dy);
    int coord_z = static_cast<int>(wrapped_pos.z / dz);
    
    // Clamp to valid range
    coord_x = std::min(coord_x, dims_[0] - 1);
    coord_y = std::min(coord_y, dims_[1] - 1);
    coord_z = std::min(coord_z, dims_[2] - 1);
    
    // Calculate rank from coordinates
    int target_rank = coord_x * dims_[1] * dims_[2] + coord_y * dims_[2] + coord_z;
    
    return target_rank;
}

LoadBalancer::LoadBalancer(ClusterCommunicator* comm) : comm_(comm) {
    particle_counts_.resize(comm_->get_size());
    computation_times_.resize(comm_->get_size());
}

void LoadBalancer::update_load_info(size_t local_particles, double computation_time) {
    // Gather load information from all processes
    MPI_Allgather(&local_particles, 1, MPI_UNSIGNED_LONG,
                  particle_counts_.data(), 1, MPI_UNSIGNED_LONG,
                  MPI_COMM_WORLD);
    
    MPI_Allgather(&computation_time, 1, MPI_DOUBLE,
                  computation_times_.data(), 1, MPI_DOUBLE,
                  MPI_COMM_WORLD);
}

bool LoadBalancer::needs_rebalancing(double threshold) const {
    if (computation_times_.empty()) return false;
    
    double max_time = *std::max_element(computation_times_.begin(), computation_times_.end());
    double min_time = *std::min_element(computation_times_.begin(), computation_times_.end());
    
    return (max_time - min_time) / max_time > threshold;
}

void LoadBalancer::rebalance_domain(float box_size) {
    // Implement dynamic load balancing
    // This would adjust domain boundaries based on particle distribution
    // and computation times
    
    if (!needs_rebalancing()) return;
    
    // Simple rebalancing strategy: adjust domain sizes based on particle counts
    // In practice, this would be more sophisticated
    comm_->decompose_domain(box_size);
}

}