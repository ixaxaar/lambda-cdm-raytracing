#pragma once

#include <mpi.h>
#include <vector>
#include <memory>
#include <cstdint>
#include "physics/lambda_cdm.hpp"

namespace mpi {

struct DomainDecomposition {
    int rank;
    int size;
    float3 local_box_min;
    float3 local_box_max;
    float3 local_box_size;
    std::vector<int> neighbor_ranks;
};

class ClusterCommunicator {
private:
    MPI_Comm comm_;
    DomainDecomposition domain_;
    std::vector<physics::Particle> ghost_particles_;
    std::vector<physics::Particle> send_buffer_;
    std::vector<physics::Particle> recv_buffer_;

    float ghost_zone_width_;
    float box_size_;
    int dims_[3];

public:
    ClusterCommunicator(MPI_Comm comm = MPI_COMM_WORLD, float ghost_width = 0.1f);
    ~ClusterCommunicator();

    bool initialize(float box_size);
    void decompose_domain(float box_size);

    void exchange_particles(std::vector<physics::Particle>& particles);
    void exchange_ghost_particles(const std::vector<physics::Particle>& particles);
    void all_reduce_forces(std::vector<float3>& forces);

    void gather_all_particles(const std::vector<physics::Particle>& local_particles,
                             std::vector<physics::Particle>& all_particles);

    int get_rank() const { return domain_.rank; }
    int get_size() const { return domain_.size; }
    const DomainDecomposition& get_domain() const { return domain_; }

    bool is_particle_local(const physics::Particle& particle) const;
    bool is_particle_ghost(const physics::Particle& particle) const;
    int find_owner_rank(const float3& position) const;
};

class LoadBalancer {
private:
    ClusterCommunicator* comm_;
    std::vector<size_t> particle_counts_;
    std::vector<double> computation_times_;

public:
    LoadBalancer(ClusterCommunicator* comm);

    void update_load_info(size_t local_particles, double computation_time);
    bool needs_rebalancing(double threshold = 0.2) const;
    void rebalance_domain(float box_size);
};

// Domain decomposition utilities
void adaptive_domain_decomposition(const std::vector<physics::Particle>& particles,
                                 float box_size, int num_processes,
                                 std::vector<float3>& domain_bounds);

void uniform_domain_decomposition(float box_size, int num_processes,
                                std::vector<float3>& domain_bounds);

void morton_order_traversal(const std::vector<std::vector<std::vector<int>>>& density_grid,
                          std::vector<int>& cell_counts);

uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z);

void distribute_cells_by_load(const std::vector<int>& cell_counts, int num_processes,
                            std::vector<float3>& domain_bounds, float box_size, int grid_res);

}