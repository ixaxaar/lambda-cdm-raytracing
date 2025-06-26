#include "forces/tree_force_computer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

// Forward declare CUDA functions to avoid header conflicts
#ifdef HAVE_CUDA
    extern "C" {
        typedef int cudaError_t;
        cudaError_t cudaGetDeviceCount(int* count);
        cudaError_t cudaMalloc(void** devPtr, size_t size);
        cudaError_t cudaFree(void* devPtr);
        const int cudaSuccess = 0;
    }
#endif

namespace forces {

TreeForceComputer::TreeForceComputer(const std::string& name)
    : TreeForceComputer(name, 0.5f, 8, 20)
{
}

TreeForceComputer::TreeForceComputer(const std::string& name, float theta,
                                   size_t leaf_capacity, int max_depth)
    : name_(name)
    , theta_(theta)
    , leaf_capacity_(leaf_capacity)
    , max_depth_(max_depth)
    , box_size_(100.0f)  // Default box size
    , use_gpu_(true)
    , d_positions_(nullptr)
    , d_masses_(nullptr)
    , d_forces_(nullptr)
    , d_tree_nodes_(nullptr)
    , max_particles_(1000000)
    , force_evaluations_(0)
    , tree_traversals_(0)
{
}

TreeForceComputer::~TreeForceComputer() {
    cleanup_gpu_resources();
}

bool TreeForceComputer::initialize(const core::SimulationContext& context) {
    try {
        // TODO: Get configuration from context
        // For now, use default values
        (void)context;  // Suppress unused parameter warning

        // Initialize GPU resources if needed
        if (use_gpu_) {
#ifdef HAVE_CUDA
            if (!initialize_gpu_resources(max_particles_)) {
                std::cerr << "Failed to initialize GPU resources, falling back to CPU" << std::endl;
                use_gpu_ = false;
            }
#else
            std::cerr << "CUDA not available, falling back to CPU" << std::endl;
            use_gpu_ = false;
#endif
        }

        std::cout << "TreeForceComputer initialized: " << name_
                  << " (theta=" << theta_ << ", GPU=" << use_gpu_ << ")" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "TreeForceComputer initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void TreeForceComputer::finalize() {
    cleanup_gpu_resources();
    std::cout << "TreeForceComputer finalized: " << name_ << std::endl;
}

void TreeForceComputer::compute_forces(const float* positions, const float* masses,
                                     float* forces, size_t num_particles,
                                     const std::any& params) {
    if (num_particles == 0) return;
    (void)params;  // Suppress unused parameter warning

    try {
        if (use_gpu_) {
            compute_forces_gpu(positions, masses, forces, num_particles);
        } else {
            compute_forces_cpu(positions, masses, forces, num_particles);
        }
    } catch (const std::exception& e) {
        std::cerr << "TreeForceComputer::compute_forces failed: " << e.what() << std::endl;
        throw;
    }
}

void TreeForceComputer::compute_forces_gpu(const float* positions, const float* masses,
                                         float* forces, size_t num_particles) {
#ifdef HAVE_CUDA
    // TODO: Implement GPU Barnes-Hut tree force computation
    // For now, fall back to CPU implementation
    (void)positions; (void)masses; (void)forces; (void)num_particles;

    std::cerr << "GPU tree force computation not yet fully implemented, falling back to CPU" << std::endl;

    // Fall back to CPU
    use_gpu_ = false;
    compute_forces_cpu(positions, masses, forces, num_particles);

    force_evaluations_ += num_particles;
    tree_traversals_ += num_particles;
#else
    throw std::runtime_error("CUDA not available for GPU force computation");
#endif
}

void TreeForceComputer::compute_forces_cpu(const float* positions, const float* masses,
                                         float* forces, size_t num_particles) {
    // Build CPU tree
    build_tree_cpu(positions, masses, num_particles);

    // Compute forces using tree traversal
    compute_tree_forces(positions, forces, num_particles);

    force_evaluations_ += num_particles;
    tree_traversals_ += num_particles;
}

void TreeForceComputer::build_tree_cpu(const float* positions, const float* masses, size_t num_particles) {
    // Create root node
    float3 center = make_float3(0.0f, 0.0f, 0.0f);
    root_ = std::make_unique<OctreeNode>(center, box_size_);

    // Insert all particles
    for (size_t i = 0; i < num_particles; ++i) {
        insert_particle(root_.get(), i, positions, masses);
    }

    // Compute centers of mass bottom-up
    compute_center_of_mass(root_.get(), positions, masses);
}

void TreeForceComputer::insert_particle(OctreeNode* node, size_t particle_index,
                                       const float* positions, const float* masses) {
    // If node is a leaf and not full, add particle
    if (node->is_leaf) {
        if (node->particle_indices.size() < leaf_capacity_ && node->level < max_depth_) {
            node->particle_indices.push_back(particle_index);
            return;
        } else if (node->level < max_depth_) {
            // Subdivide node
            subdivide_node(node);
        } else {
            // Max depth reached, force add to leaf
            node->particle_indices.push_back(particle_index);
            return;
        }
    }

    // Find appropriate child
    float3 particle_pos;
    particle_pos.x = positions[particle_index * 3 + 0];
    particle_pos.y = positions[particle_index * 3 + 1];
    particle_pos.z = positions[particle_index * 3 + 2];

    int octant = get_octant(particle_pos, node->center);

    // Recursively insert into child
    insert_particle(node->children[octant].get(), particle_index, positions, masses);
}

void TreeForceComputer::subdivide_node(OctreeNode* node) {
    node->is_leaf = false;
    float half_size = node->size * 0.5f;

    // Create 8 children
    for (int i = 0; i < 8; ++i) {
        float3 child_center;
        child_center.x = node->center.x + ((i & 1) ? half_size : -half_size) * 0.5f;
        child_center.y = node->center.y + ((i & 2) ? half_size : -half_size) * 0.5f;
        child_center.z = node->center.z + ((i & 4) ? half_size : -half_size) * 0.5f;

        node->children[i] = std::make_unique<OctreeNode>(child_center, half_size, node->level + 1);
    }
}

int TreeForceComputer::get_octant(const float3& particle_pos, const float3& node_center) const {
    int octant = 0;
    if (particle_pos.x > node_center.x) octant |= 1;
    if (particle_pos.y > node_center.y) octant |= 2;
    if (particle_pos.z > node_center.z) octant |= 4;
    return octant;
}

void TreeForceComputer::compute_center_of_mass(OctreeNode* node, const float* positions, const float* masses) {
    if (node->is_leaf) {
        // Leaf node: compute center of mass from particles
        float total_mass = 0.0f;
        float3 weighted_pos = make_float3(0.0f, 0.0f, 0.0f);

        for (size_t particle_idx : node->particle_indices) {
            float mass = masses[particle_idx];
            total_mass += mass;

            weighted_pos.x += positions[particle_idx * 3 + 0] * mass;
            weighted_pos.y += positions[particle_idx * 3 + 1] * mass;
            weighted_pos.z += positions[particle_idx * 3 + 2] * mass;
        }

        if (total_mass > 0.0f) {
            node->center_of_mass.x = weighted_pos.x / total_mass;
            node->center_of_mass.y = weighted_pos.y / total_mass;
            node->center_of_mass.z = weighted_pos.z / total_mass;
        }
        node->total_mass = total_mass;
    } else {
        // Internal node: compute from children
        float total_mass = 0.0f;
        float3 weighted_pos = make_float3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                compute_center_of_mass(node->children[i].get(), positions, masses);

                float child_mass = node->children[i]->total_mass;
                if (child_mass > 0.0f) {
                    total_mass += child_mass;
                    weighted_pos.x += node->children[i]->center_of_mass.x * child_mass;
                    weighted_pos.y += node->children[i]->center_of_mass.y * child_mass;
                    weighted_pos.z += node->children[i]->center_of_mass.z * child_mass;
                }
            }
        }

        if (total_mass > 0.0f) {
            node->center_of_mass.x = weighted_pos.x / total_mass;
            node->center_of_mass.y = weighted_pos.y / total_mass;
            node->center_of_mass.z = weighted_pos.z / total_mass;
        }
        node->total_mass = total_mass;
    }
}

void TreeForceComputer::compute_tree_forces(const float* positions, float* forces, size_t num_particles) const {
    // Initialize forces to zero
    for (size_t i = 0; i < num_particles * 3; ++i) {
        forces[i] = 0.0f;
    }

    // Compute force on each particle
    for (size_t i = 0; i < num_particles; ++i) {
        compute_force_on_particle(i, positions, nullptr, forces, root_.get());
    }
}

void TreeForceComputer::compute_force_on_particle(size_t particle_index, const float* positions,
                                                const float* masses, float* forces,
                                                const OctreeNode* node) const {
    if (!node || node->total_mass == 0.0f) return;

    float3 particle_pos;
    particle_pos.x = positions[particle_index * 3 + 0];
    particle_pos.y = positions[particle_index * 3 + 1];
    particle_pos.z = positions[particle_index * 3 + 2];

    if (node->is_leaf) {
        // Leaf node: compute direct interactions
        compute_node_particle_interaction(particle_index, positions, masses, forces, node);
    } else {
        // Internal node: check opening criterion
        if (satisfies_opening_criterion(particle_pos, node)) {
            // Use node's center of mass
            float3 dr;
            dr.x = node->center_of_mass.x - particle_pos.x;
            dr.y = node->center_of_mass.y - particle_pos.y;
            dr.z = node->center_of_mass.z - particle_pos.z;

            float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
            float softening = 0.01f;
            r2 += softening * softening;

            float r = sqrtf(r2);
            float r3 = r2 * r;

            float force_magnitude = node->total_mass / r3;

            forces[particle_index * 3 + 0] += force_magnitude * dr.x;
            forces[particle_index * 3 + 1] += force_magnitude * dr.y;
            forces[particle_index * 3 + 2] += force_magnitude * dr.z;
        } else {
            // Recurse to children
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    compute_force_on_particle(particle_index, positions, masses, forces, node->children[i].get());
                }
            }
        }
    }
}

bool TreeForceComputer::satisfies_opening_criterion(const float3& particle_pos, const OctreeNode* node) const {
    float3 dr;
    dr.x = node->center_of_mass.x - particle_pos.x;
    dr.y = node->center_of_mass.y - particle_pos.y;
    dr.z = node->center_of_mass.z - particle_pos.z;

    float r = sqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
    return (node->size / r) < theta_;
}

void TreeForceComputer::compute_node_particle_interaction(size_t particle_index, const float* positions,
                                                        const float* masses, float* forces,
                                                        const OctreeNode* node) const {
    float3 particle_pos;
    particle_pos.x = positions[particle_index * 3 + 0];
    particle_pos.y = positions[particle_index * 3 + 1];
    particle_pos.z = positions[particle_index * 3 + 2];

    for (size_t other_idx : node->particle_indices) {
        if (other_idx == particle_index) continue;  // Skip self-interaction

        float3 other_pos;
        other_pos.x = positions[other_idx * 3 + 0];
        other_pos.y = positions[other_idx * 3 + 1];
        other_pos.z = positions[other_idx * 3 + 2];

        float3 dr;
        dr.x = other_pos.x - particle_pos.x;
        dr.y = other_pos.y - particle_pos.y;
        dr.z = other_pos.z - particle_pos.z;

        float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        float softening = 0.01f;
        r2 += softening * softening;

        float r = sqrtf(r2);
        float r3 = r2 * r;

        float other_mass = masses ? masses[other_idx] : 1.0f;
        float force_magnitude = other_mass / r3;

        forces[particle_index * 3 + 0] += force_magnitude * dr.x;
        forces[particle_index * 3 + 1] += force_magnitude * dr.y;
        forces[particle_index * 3 + 2] += force_magnitude * dr.z;
    }
}

bool TreeForceComputer::initialize_gpu_resources(size_t max_particles) {
#ifdef HAVE_CUDA
    try {
        cudaError_t err;

        // Check for CUDA device
        int device_count;
        err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return false;
        }

        // Allocate GPU memory
        err = cudaMalloc((void**)&d_positions_, max_particles * 3 * sizeof(float));
        if (err != cudaSuccess) return false;

        err = cudaMalloc((void**)&d_masses_, max_particles * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_positions_);
            return false;
        }

        err = cudaMalloc((void**)&d_forces_, max_particles * 3 * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_positions_);
            cudaFree(d_masses_);
            return false;
        }

        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

void TreeForceComputer::cleanup_gpu_resources() {
#ifdef HAVE_CUDA
    if (d_positions_) {
        cudaFree(d_positions_);
        d_positions_ = nullptr;
    }
    if (d_masses_) {
        cudaFree(d_masses_);
        d_masses_ = nullptr;
    }
    if (d_forces_) {
        cudaFree(d_forces_);
        d_forces_ = nullptr;
    }
    if (d_tree_nodes_) {
        cudaFree(d_tree_nodes_);
        d_tree_nodes_ = nullptr;
    }
#endif
}

size_t TreeForceComputer::get_tree_depth() const {
    return root_ ? compute_tree_depth(root_.get()) : 0;
}

size_t TreeForceComputer::get_node_count() const {
    return root_ ? count_nodes(root_.get()) : 0;
}

size_t TreeForceComputer::get_leaf_count() const {
    return root_ ? count_leaves(root_.get()) : 0;
}

float TreeForceComputer::get_tree_efficiency() const {
    size_t n2 = get_max_particles() * get_max_particles();
    return n2 > 0 ? static_cast<float>(force_evaluations_) / n2 : 0.0f;
}

size_t TreeForceComputer::count_nodes(const OctreeNode* node) const {
    if (!node) return 0;

    size_t count = 1;
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            count += count_nodes(node->children[i].get());
        }
    }
    return count;
}

size_t TreeForceComputer::count_leaves(const OctreeNode* node) const {
    if (!node) return 0;

    if (node->is_leaf) return 1;

    size_t count = 0;
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            count += count_leaves(node->children[i].get());
        }
    }
    return count;
}

int TreeForceComputer::compute_tree_depth(const OctreeNode* node) const {
    if (!node) return 0;

    if (node->is_leaf) return 1;

    int max_depth = 0;
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            max_depth = std::max(max_depth, compute_tree_depth(node->children[i].get()));
        }
    }
    return max_depth + 1;
}

} // namespace forces