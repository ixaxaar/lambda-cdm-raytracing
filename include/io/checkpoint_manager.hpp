#pragma once

#include "core/interfaces.hpp"
#include "data_export.hpp"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace io {

/**
 * @brief Manages simulation checkpoints for save/restore functionality
 *
 * Features:
 * - Automatic checkpoint creation based on time/step intervals
 * - Checkpoint rotation (keep N most recent checkpoints)
 * - Full simulation state save/restore
 * - Metadata tracking (timestamp, version, etc.)
 * - Restart capability from any checkpoint
 */
class CheckpointManager {
private:
    std::string checkpoint_directory_;
    size_t checkpoint_frequency_steps_;
    double checkpoint_frequency_time_;
    size_t max_checkpoints_;  // Keep only N most recent
    bool enabled_;

    std::vector<std::string> checkpoint_list_;
    std::chrono::system_clock::time_point last_checkpoint_time_;

    std::shared_ptr<DataExporter> exporter_;

public:
    CheckpointManager();
    explicit CheckpointManager(const std::string& checkpoint_dir);
    ~CheckpointManager() = default;

    // Configuration
    void set_checkpoint_directory(const std::string& dir) { checkpoint_directory_ = dir; }
    void set_frequency_by_steps(size_t steps) { checkpoint_frequency_steps_ = steps; }
    void set_frequency_by_time(double time_interval) { checkpoint_frequency_time_ = time_interval; }
    void set_max_checkpoints(size_t max) { max_checkpoints_ = max; }
    void enable(bool enabled = true) { enabled_ = enabled; }
    void set_exporter(std::shared_ptr<DataExporter> exporter) { exporter_ = exporter; }

    // Checkpoint operations
    bool should_create_checkpoint(size_t current_step, double current_time);

    bool create_checkpoint(const std::string& checkpoint_name,
                          const float* positions, const float* velocities,
                          const float* masses, size_t num_particles,
                          double time, const SnapshotMetadata& metadata);

    bool restore_from_checkpoint(const std::string& checkpoint_name,
                                 float* positions, float* velocities,
                                 float* masses, size_t& num_particles,
                                 double& time, SnapshotMetadata& metadata);

    // Checkpoint management
    std::vector<std::string> list_checkpoints() const;
    std::string get_latest_checkpoint() const;
    bool delete_checkpoint(const std::string& checkpoint_name);
    bool delete_old_checkpoints();  // Keep only max_checkpoints_ most recent

    // Utility
    std::string generate_checkpoint_name(size_t step, double time) const;
    bool checkpoint_exists(const std::string& checkpoint_name) const;
    bool validate_checkpoint(const std::string& checkpoint_name) const;

    // Status
    bool is_enabled() const { return enabled_; }
    size_t get_checkpoint_count() const { return checkpoint_list_.size(); }

private:
    void update_checkpoint_list();
    void ensure_directory_exists(const std::string& dir);
};

}
