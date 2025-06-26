#include "mpi/cluster_comm.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <cstdint>

namespace mpi {

class DomainDecomposer {
private:
    int dims_[3];
    float box_size_;
    MPI_Comm comm_;

public:
    DomainDecomposer(MPI_Comm comm, float box_size)
        : box_size_(box_size), comm_(comm) {
        dims_[0] = dims_[1] = dims_[2] = 0;
    }

    void create_3d_decomposition(int num_processes) {
        // Find optimal 3D factorization
        MPI_Dims_create(num_processes, 3, dims_);

        // Optimize for cube-like domains
        optimize_dimensions();
    }

    void create_cartesian_topology(MPI_Comm* cart_comm) {
        int periods[3] = {1, 1, 1}; // Periodic in all dimensions
        int reorder = 1; // Allow MPI to reorder ranks

        MPI_Cart_create(comm_, 3, dims_, periods, reorder, cart_comm);
    }

    void get_local_domain(int rank, float3& min_bounds, float3& max_bounds) {
        int coords[3];
        MPI_Cart_coords(comm_, rank, 3, coords);

        float dx = box_size_ / dims_[0];
        float dy = box_size_ / dims_[1];
        float dz = box_size_ / dims_[2];

        min_bounds = make_float3(
            coords[0] * dx,
            coords[1] * dy,
            coords[2] * dz
        );

        max_bounds = make_float3(
            (coords[0] + 1) * dx,
            (coords[1] + 1) * dy,
            (coords[2] + 1) * dz
        );
    }

    std::vector<int> get_neighbor_ranks(int rank) {
        std::vector<int> neighbors;
        int coords[3];
        MPI_Cart_coords(comm_, rank, 3, coords);

        // Get all 26 neighbors (3^3 - 1)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;

                    int neighbor_coords[3] = {
                        coords[0] + dx,
                        coords[1] + dy,
                        coords[2] + dz
                    };

                    int neighbor_rank;
                    MPI_Cart_rank(comm_, neighbor_coords, &neighbor_rank);
                    neighbors.push_back(neighbor_rank);
                }
            }
        }

        return neighbors;
    }

    void get_dimensions(int* dims) const {
        dims[0] = dims_[0];
        dims[1] = dims_[1];
        dims[2] = dims_[2];
    }

private:
    void optimize_dimensions() {
        // Try to make dimensions as close to each other as possible
        // for better load balancing and communication

        std::sort(dims_, dims_ + 3);

        // If one dimension is much larger, try to redistribute
        while (dims_[2] > 2 * dims_[0] && dims_[2] % 2 == 0) {
            if (dims_[0] < dims_[1]) {
                dims_[0] *= 2;
                dims_[2] /= 2;
            } else if (dims_[1] < dims_[2] / 2) {
                dims_[1] *= 2;
                dims_[2] /= 2;
            } else {
                break;
            }
            std::sort(dims_, dims_ + 3);
        }
    }
};

void adaptive_domain_decomposition(const std::vector<physics::Particle>& particles,
                                 float box_size, int num_processes,
                                 std::vector<float3>& domain_bounds) {
    // Implement space-filling curve based decomposition for better load balancing

    if (particles.empty()) {
        // Fall back to uniform decomposition
        uniform_domain_decomposition(box_size, num_processes, domain_bounds);
        return;
    }

    // Calculate particle density in grid cells
    const int grid_res = 32; // Resolution for density calculation
    std::vector<std::vector<std::vector<int>>> density_grid(
        grid_res, std::vector<std::vector<int>>(grid_res, std::vector<int>(grid_res, 0)));

    float cell_size = box_size / grid_res;

    for (const auto& particle : particles) {
        int ix = std::min(static_cast<int>(particle.position.x / cell_size), grid_res - 1);
        int iy = std::min(static_cast<int>(particle.position.y / cell_size), grid_res - 1);
        int iz = std::min(static_cast<int>(particle.position.z / cell_size), grid_res - 1);

        density_grid[ix][iy][iz]++;
    }

    // Use Z-order curve to traverse cells and assign to processes
    std::vector<int> cell_counts;
    morton_order_traversal(density_grid, cell_counts);

    // Distribute cells among processes to balance particle counts
    distribute_cells_by_load(cell_counts, num_processes, domain_bounds, box_size, grid_res);
}

void uniform_domain_decomposition(float box_size, int num_processes,
                                std::vector<float3>& domain_bounds) {
    domain_bounds.resize(num_processes * 2); // min and max for each domain

    DomainDecomposer decomposer(MPI_COMM_WORLD, box_size);
    decomposer.create_3d_decomposition(num_processes);

    for (int rank = 0; rank < num_processes; ++rank) {
        float3 min_bounds, max_bounds;
        decomposer.get_local_domain(rank, min_bounds, max_bounds);

        domain_bounds[rank * 2] = min_bounds;
        domain_bounds[rank * 2 + 1] = max_bounds;
    }
}

void morton_order_traversal(const std::vector<std::vector<std::vector<int>>>& density_grid,
                          std::vector<int>& cell_counts) {
    int grid_res = density_grid.size();
    cell_counts.clear();
    cell_counts.reserve(grid_res * grid_res * grid_res);

    // Generate Morton order indices
    for (int z = 0; z < grid_res; ++z) {
        for (int y = 0; y < grid_res; ++y) {
            for (int x = 0; x < grid_res; ++x) {
                uint32_t morton_index = morton_encode_3d(x, y, z);
                cell_counts.push_back(density_grid[x][y][z]);
            }
        }
    }

    // Sort by Morton order for better spatial locality
    std::vector<std::pair<uint32_t, int>> morton_pairs;
    for (size_t i = 0; i < cell_counts.size(); ++i) {
        int x = i % grid_res;
        int y = (i / grid_res) % grid_res;
        int z = i / (grid_res * grid_res);
        morton_pairs.emplace_back(morton_encode_3d(x, y, z), cell_counts[i]);
    }

    std::sort(morton_pairs.begin(), morton_pairs.end());

    for (size_t i = 0; i < morton_pairs.size(); ++i) {
        cell_counts[i] = morton_pairs[i].second;
    }
}

uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    // Interleave bits of x, y, z to create 3D Morton code
    auto part1by2 = [](uint32_t n) -> uint32_t {
        n &= 0x000003ff;
        n = (n ^ (n << 16)) & 0xff0000ff;
        n = (n ^ (n << 8))  & 0x0300f00f;
        n = (n ^ (n << 4))  & 0x030c30c3;
        n = (n ^ (n << 2))  & 0x09249249;
        return n;
    };

    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
}

void distribute_cells_by_load(const std::vector<int>& cell_counts, int num_processes,
                            std::vector<float3>& domain_bounds, float box_size, int grid_res) {
    // Simple greedy algorithm to distribute cells among processes
    std::vector<int> process_loads(num_processes, 0);
    std::vector<std::vector<int>> process_cells(num_processes);

    int total_particles = std::accumulate(cell_counts.begin(), cell_counts.end(), 0);
    int target_load = total_particles / num_processes;

    int current_process = 0;
    for (size_t i = 0; i < cell_counts.size(); ++i) {
        if (current_process < num_processes - 1 &&
            process_loads[current_process] + cell_counts[i] > target_load) {
            current_process++;
        }

        process_loads[current_process] += cell_counts[i];
        process_cells[current_process].push_back(i);
    }

    // Convert cell assignments to domain bounds
    domain_bounds.resize(num_processes * 2);
    float cell_size = box_size / grid_res;

    for (int proc = 0; proc < num_processes; ++proc) {
        if (process_cells[proc].empty()) {
            domain_bounds[proc * 2] = make_float3(0, 0, 0);
            domain_bounds[proc * 2 + 1] = make_float3(0, 0, 0);
            continue;
        }

        float3 min_bounds = make_float3(box_size, box_size, box_size);
        float3 max_bounds = make_float3(0, 0, 0);

        for (int cell_idx : process_cells[proc]) {
            int x = cell_idx % grid_res;
            int y = (cell_idx / grid_res) % grid_res;
            int z = cell_idx / (grid_res * grid_res);

            float3 cell_min = make_float3(x * cell_size, y * cell_size, z * cell_size);
            float3 cell_max = make_float3((x + 1) * cell_size, (y + 1) * cell_size, (z + 1) * cell_size);

            min_bounds.x = std::min(min_bounds.x, cell_min.x);
            min_bounds.y = std::min(min_bounds.y, cell_min.y);
            min_bounds.z = std::min(min_bounds.z, cell_min.z);

            max_bounds.x = std::max(max_bounds.x, cell_max.x);
            max_bounds.y = std::max(max_bounds.y, cell_max.y);
            max_bounds.z = std::max(max_bounds.z, cell_max.z);
        }

        domain_bounds[proc * 2] = min_bounds;
        domain_bounds[proc * 2 + 1] = max_bounds;
    }
}

}