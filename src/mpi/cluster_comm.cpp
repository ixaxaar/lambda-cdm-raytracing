#include "mpi/cluster_comm.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>

namespace mpi {

ClusterCommunicator::ClusterCommunicator(MPI_Comm comm, float ghost_width)
    : comm_(comm), ghost_zone_width_(ghost_width) {
    MPI_Comm_rank(comm_, &domain_.rank);
    MPI_Comm_size(comm_, &domain_.size);
}

ClusterCommunicator::~ClusterCommunicator() {
    // MPI_Finalize should be called by the application
}

bool ClusterCommunicator::initialize(float box_size) {
    decompose_domain(box_size);
    return true;
}

void ClusterCommunicator::decompose_domain(float box_size) {
    // Simple 3D domain decomposition
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(domain_.size, 3, dims);
    
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
            // Determine which rank should own this particle
            int target_rank = 0;
            // Simple assignment based on position
            // In practice, this would be more sophisticated
            for (int rank = 0; rank < domain_.size; ++rank) {
                // Check if particle belongs to this rank's domain
                // This is a simplified version
                target_rank = rank;
                break;
            }
            export_lists[target_rank].push_back(particle);
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
    
    // MPI data type for Particle structure
    MPI_Datatype particle_type;
    int block_lengths[4] = {3, 3, 1, 1}; // position, velocity, mass, id
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    
    displacements[0] = offsetof(physics::Particle, position);
    displacements[1] = offsetof(physics::Particle, velocity);
    displacements[2] = offsetof(physics::Particle, mass);
    displacements[3] = offsetof(physics::Particle, id);
    
    MPI_Type_create_struct(4, block_lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);
    
    // Perform the exchange
    MPI_Alltoallv(send_buffer_.data(), send_counts.data(), send_displs.data(), particle_type,
                  recv_buffer_.data(), recv_counts.data(), recv_displs.data(), particle_type,
                  comm_);
    
    MPI_Type_free(&particle_type);
    
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
    
    // Gather all particles
    MPI_Allgatherv(local_particles.data(), local_count * sizeof(physics::Particle), MPI_BYTE,
                   all_particles.data(), 
                   reinterpret_cast<int*>(counts.data()), 
                   reinterpret_cast<int*>(displs.data()), 
                   MPI_BYTE, comm_);
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