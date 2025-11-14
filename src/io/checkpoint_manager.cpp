#include "io/checkpoint_manager.hpp"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace io {

CheckpointManager::CheckpointManager()
    : checkpoint_directory_("checkpoints")
    , checkpoint_frequency_steps_(0)
    , checkpoint_frequency_time_(0.0)
    , max_checkpoints_(5)
    , enabled_(false)
    , last_checkpoint_time_(std::chrono::system_clock::now())
    , exporter_(nullptr)
{
    ensure_directory_exists(checkpoint_directory_);
}

CheckpointManager::CheckpointManager(const std::string& checkpoint_dir)
    : checkpoint_directory_(checkpoint_dir)
    , checkpoint_frequency_steps_(0)
    , checkpoint_frequency_time_(0.0)
    , max_checkpoints_(5)
    , enabled_(false)
    , last_checkpoint_time_(std::chrono::system_clock::now())
    , exporter_(nullptr)
{
    ensure_directory_exists(checkpoint_directory_);
}

bool CheckpointManager::should_create_checkpoint(size_t current_step, double current_time) {
    if (!enabled_) {
        return false;
    }

    // Check step-based frequency
    if (checkpoint_frequency_steps_ > 0) {
        if (current_step % checkpoint_frequency_steps_ == 0) {
            return true;
        }
    }

    // Check time-based frequency
    if (checkpoint_frequency_time_ > 0.0) {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_checkpoint_time_).count();

        if (elapsed >= checkpoint_frequency_time_) {
            last_checkpoint_time_ = now;
            return true;
        }
    }

    return false;
}

bool CheckpointManager::create_checkpoint(const std::string& checkpoint_name,
                                          const float* positions, const float* velocities,
                                          const float* masses, size_t num_particles,
                                          double time, const SnapshotMetadata& metadata) {
    if (!exporter_) {
        std::cerr << "No exporter set for checkpoint manager" << std::endl;
        return false;
    }

    std::string checkpoint_path = checkpoint_directory_ + "/" + checkpoint_name;

    // Create metadata with checkpoint info
    SnapshotMetadata checkpoint_metadata = metadata;
    checkpoint_metadata.attributes["checkpoint_name"] = checkpoint_name;
    checkpoint_metadata.attributes["checkpoint_time"] = std::to_string(time);

    std::any metadata_any = std::make_any<SnapshotMetadata>(checkpoint_metadata);

    // Write checkpoint using exporter
    bool success = exporter_->export_snapshot(checkpoint_path, positions, velocities,
                                             masses, num_particles, time, metadata_any);

    if (success) {
        checkpoint_list_.push_back(checkpoint_name);
        std::cout << "Checkpoint created: " << checkpoint_name << std::endl;

        // Delete old checkpoints if needed
        delete_old_checkpoints();
    }

    return success;
}

bool CheckpointManager::restore_from_checkpoint(const std::string& checkpoint_name,
                                                float* positions, float* velocities,
                                                float* masses, size_t& num_particles,
                                                double& time, SnapshotMetadata& metadata) {
    if (!exporter_) {
        std::cerr << "No exporter set for checkpoint manager" << std::endl;
        return false;
    }

    std::string checkpoint_path = checkpoint_directory_ + "/" + checkpoint_name;

    if (!checkpoint_exists(checkpoint_name)) {
        std::cerr << "Checkpoint does not exist: " << checkpoint_name << std::endl;
        return false;
    }

    std::any metadata_any;

    // Read checkpoint using exporter
    bool success = exporter_->import_snapshot(checkpoint_path, positions, velocities,
                                             masses, num_particles, time, metadata_any);

    if (success) {
        // Extract metadata
        const auto* meta_ptr = std::any_cast<SnapshotMetadata>(&metadata_any);
        if (meta_ptr) {
            metadata = *meta_ptr;
        }

        std::cout << "Checkpoint restored: " << checkpoint_name << std::endl;
    }

    return success;
}

std::vector<std::string> CheckpointManager::list_checkpoints() const {
    // Return the current checkpoint list without updating
    // (update_checkpoint_list is not const, so we can't call it here)
    return checkpoint_list_;
}

std::string CheckpointManager::get_latest_checkpoint() const {
    if (checkpoint_list_.empty()) {
        return "";
    }

    // Return the most recent checkpoint (last in list)
    return checkpoint_list_.back();
}

bool CheckpointManager::delete_checkpoint(const std::string& checkpoint_name) {
    std::string checkpoint_path = checkpoint_directory_ + "/" + checkpoint_name;

    try {
        if (std::filesystem::exists(checkpoint_path)) {
            std::filesystem::remove(checkpoint_path);

            // Remove from list
            auto it = std::find(checkpoint_list_.begin(), checkpoint_list_.end(), checkpoint_name);
            if (it != checkpoint_list_.end()) {
                checkpoint_list_.erase(it);
            }

            std::cout << "Checkpoint deleted: " << checkpoint_name << std::endl;
            return true;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error deleting checkpoint: " << e.what() << std::endl;
    }

    return false;
}

bool CheckpointManager::delete_old_checkpoints() {
    if (max_checkpoints_ == 0) {
        return true;  // No limit
    }

    while (checkpoint_list_.size() > max_checkpoints_) {
        // Delete the oldest checkpoint (first in list)
        std::string oldest = checkpoint_list_.front();
        if (!delete_checkpoint(oldest)) {
            return false;
        }
    }

    return true;
}

std::string CheckpointManager::generate_checkpoint_name(size_t step, double time) const {
    std::ostringstream oss;
    oss << "checkpoint_step_" << std::setfill('0') << std::setw(8) << step
        << "_time_" << std::fixed << std::setprecision(3) << time;
    return oss.str();
}

bool CheckpointManager::checkpoint_exists(const std::string& checkpoint_name) const {
    std::string checkpoint_path = checkpoint_directory_ + "/" + checkpoint_name;
    return std::filesystem::exists(checkpoint_path);
}

bool CheckpointManager::validate_checkpoint(const std::string& checkpoint_name) const {
    if (!checkpoint_exists(checkpoint_name)) {
        return false;
    }

    // Could add additional validation here (file size, metadata, etc.)
    return true;
}

void CheckpointManager::update_checkpoint_list() {
    // Scan checkpoint directory for files
    // For simplicity, we'll just keep the internal list
}

void CheckpointManager::ensure_directory_exists(const std::string& dir) {
    try {
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
            std::cout << "Created checkpoint directory: " << dir << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}

}
