#pragma once

#include "core/interfaces.hpp"
#include "core/math_types.hpp"
#include "force_computer_factory.hpp"
#include <memory>
#include <vector>
#include <array>

namespace forces {

struct OctreeNode {
    float3 center;
    float size;
    float total_mass;
    float3 center_of_mass;
    
    // Children nodes (8 for octree)
    std::array<std::unique_ptr<OctreeNode>, 8> children;
    
    // Particles in this node (for leaf nodes)
    std::vector<size_t> particle_indices;
    
    bool is_leaf;
    int level;
    
    OctreeNode(const float3& center, float size, int level = 0)
        : center(center), size(size), total_mass(0.0f), 
          center_of_mass(make_float3(0, 0, 0)), is_leaf(true), level(level) {}
};

class TreeForceComputer : public core::IForceComputer {
private:
    std::string name_;
    std::unique_ptr<OctreeNode> root_;
    std::unique_ptr<IForceKernel> force_kernel_;
    
    // Configuration parameters
    float theta_;  // Opening angle criterion
    size_t leaf_capacity_;
    int max_depth_;
    float box_size_;
    bool use_gpu_;
    
    // GPU resources
    float* d_positions_;
    float* d_masses_;
    float* d_forces_;
    OctreeNode* d_tree_nodes_;
    size_t max_particles_;
    
    // Performance counters
    mutable size_t force_evaluations_;
    mutable size_t tree_traversals_;
    
public:
    explicit TreeForceComputer(const std::string& name);
    TreeForceComputer(const std::string& name, float theta, 
                     size_t leaf_capacity = 8, int max_depth = 20);
    ~TreeForceComputer();
    
    // IComponent interface
    bool initialize(const core::SimulationContext& context) override;
    void finalize() override;
    std::string get_type() const override { return "TreeForceComputer"; }
    std::string get_name() const override { return name_; }
    std::string get_version() const override { return "1.0.0"; }
    
    // IForceComputer interface
    void compute_forces(const float* positions, const float* masses,
                       float* forces, size_t num_particles,
                       const std::any& params = {}) override;
    
    bool supports_gpu() const override { return use_gpu_; }
    bool supports_mpi() const override { return true; }
    size_t get_max_particles() const override { return max_particles_; }
    
    // Tree-specific methods
    void build_tree(const float* positions, const float* masses, size_t num_particles);
    void compute_tree_forces(const float* positions, float* forces, size_t num_particles) const;
    
    // Configuration
    void set_opening_angle(float theta) { theta_ = theta; }
    void set_leaf_capacity(size_t capacity) { leaf_capacity_ = capacity; }
    void set_max_depth(int depth) { max_depth_ = depth; }
    void set_box_size(float size) { box_size_ = size; }
    void set_force_kernel(std::unique_ptr<IForceKernel> kernel) { force_kernel_ = std::move(kernel); }
    
    // Getters
    float get_opening_angle() const { return theta_; }
    size_t get_leaf_capacity() const { return leaf_capacity_; }
    int get_max_depth() const { return max_depth_; }
    float get_box_size() const { return box_size_; }
    
    // Statistics
    size_t get_force_evaluations() const { return force_evaluations_; }
    size_t get_tree_traversals() const { return tree_traversals_; }
    void reset_statistics() const { force_evaluations_ = 0; tree_traversals_ = 0; }
    
    // Tree introspection
    size_t get_tree_depth() const;
    size_t get_node_count() const;
    size_t get_leaf_count() const;
    float get_tree_efficiency() const; // Ratio of force evaluations to O(N^2)
    
private:
    // Tree construction helpers
    void insert_particle(OctreeNode* node, size_t particle_index,
                         const float* positions, const float* masses);
    void subdivide_node(OctreeNode* node);
    int get_octant(const float3& particle_pos, const float3& node_center) const;
    void compute_center_of_mass(OctreeNode* node, const float* positions, const float* masses);
    
    // Force computation helpers
    void compute_force_on_particle(size_t particle_index, const float* positions,
                                  const float* masses, float* forces,
                                  const OctreeNode* node) const;
    
    bool satisfies_opening_criterion(const float3& particle_pos, const OctreeNode* node) const;
    
    void compute_node_particle_interaction(size_t particle_index, const float* positions,
                                          const float* masses, float* forces,
                                          const OctreeNode* node) const;
    
    // GPU methods
    bool initialize_gpu_resources(size_t max_particles);
    void cleanup_gpu_resources();
    void upload_tree_to_gpu();
    void compute_forces_gpu(const float* positions, const float* masses,
                           float* forces, size_t num_particles);
    
    // CPU methods
    void compute_forces_cpu(const float* positions, const float* masses,
                           float* forces, size_t num_particles);
    void build_tree_cpu(const float* positions, const float* masses, size_t num_particles);
    
    // Tree traversal on GPU
    void launch_tree_traversal_kernel(const float* positions, const float* masses,
                                     float* forces, size_t num_particles) const;
    
    // Utility methods
    void print_tree_statistics() const;
    void validate_tree() const;
    size_t count_nodes(const OctreeNode* node) const;
    size_t count_leaves(const OctreeNode* node) const;
    int compute_tree_depth(const OctreeNode* node) const;
};

// CUDA kernels declarations (only when needed)
#ifdef HAVE_CUDA
extern "C" {
    void launch_tree_force_kernel(const float* positions, const float* masses,
                                 float* forces, const OctreeNode* tree_nodes,
                                 size_t num_particles, float theta);
    
    void launch_tree_build_kernel(const float* positions, const float* masses,
                                 OctreeNode* tree_nodes, size_t num_particles,
                                 float box_size, size_t leaf_capacity);
}
#endif

}