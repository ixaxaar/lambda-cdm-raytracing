#pragma once

#include "interfaces.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>

namespace core {

// Memory pool for efficient allocation/deallocation
template<typename T>
class MemoryPool {
private:
    struct Block {
        T* ptr;
        size_t size;
        bool in_use;
        
        Block(T* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    std::vector<Block> blocks_;
    std::mutex pool_mutex_;
    size_t total_allocated_;
    size_t total_in_use_;
    
public:
    MemoryPool() : total_allocated_(0), total_in_use_(0) {}
    
    ~MemoryPool() {
        cleanup();
    }
    
    T* allocate(size_t count);
    void deallocate(T* ptr);
    void cleanup();
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_total_in_use() const { return total_in_use_; }
    size_t get_fragmentation() const;
    void defragment();
};

class GPUResourceManager : public IResourceManager {
private:
    // Memory pools for different data types
    std::unique_ptr<MemoryPool<float>> float_pool_;
    std::unique_ptr<MemoryPool<double>> double_pool_;
    std::unique_ptr<MemoryPool<int>> int_pool_;
    std::unique_ptr<MemoryPool<char>> byte_pool_;
    
    // CUDA context management
    int device_id_;
    cudaDeviceProp device_properties_;
    std::vector<cudaStream_t> streams_;
    size_t num_streams_;
    std::atomic<size_t> next_stream_;
    
    // Memory tracking
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::atomic<size_t> total_gpu_memory_allocated_;
    std::atomic<size_t> total_host_memory_allocated_;
    std::mutex allocation_mutex_;
    
    // Memory limits
    size_t max_gpu_memory_;
    size_t max_host_memory_;
    float memory_usage_threshold_;
    
    // Performance monitoring
    std::atomic<size_t> allocation_count_;
    std::atomic<size_t> deallocation_count_;
    std::atomic<double> total_allocation_time_;
    
public:
    GPUResourceManager(int device_id = 0, size_t num_streams = 4);
    ~GPUResourceManager();
    
    // IResourceManager interface
    void* allocate_gpu_memory(size_t size) override;
    void* allocate_host_memory(size_t size) override;
    void free_gpu_memory(void* ptr) override;
    void free_host_memory(void* ptr) override;
    size_t get_gpu_memory_usage() const override { return total_gpu_memory_allocated_.load(); }
    size_t get_host_memory_usage() const override { return total_host_memory_allocated_.load(); }
    bool has_sufficient_gpu_memory(size_t required) const override;
    
    // Enhanced GPU memory management
    template<typename T>
    T* allocate_gpu_array(size_t count);
    
    template<typename T>
    T* allocate_host_array(size_t count);
    
    template<typename T>
    void free_gpu_array(T* ptr);
    
    template<typename T>
    void free_host_array(T* ptr);
    
    // Pinned memory for faster transfers
    void* allocate_pinned_memory(size_t size);
    void free_pinned_memory(void* ptr);
    
    // CUDA stream management
    cudaStream_t get_next_stream();
    cudaStream_t get_stream(size_t index);
    void synchronize_all_streams();
    
    // Memory transfer utilities
    void copy_to_gpu(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void copy_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void copy_gpu_to_gpu(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    
    // Asynchronous transfers
    void copy_to_gpu_async(void* dst, const void* src, size_t size, cudaStream_t stream);
    void copy_to_host_async(void* dst, const void* src, size_t size, cudaStream_t stream);
    
    // Memory pool management
    void optimize_memory_pools();
    void reset_memory_pools();
    size_t get_memory_pool_fragmentation() const;
    
    // Device management
    bool initialize_device();
    void finalize_device();
    int get_device_id() const { return device_id_; }
    const cudaDeviceProp& get_device_properties() const { return device_properties_; }
    
    // Memory limits and monitoring
    void set_memory_limits(size_t max_gpu, size_t max_host);
    void set_memory_usage_threshold(float threshold) { memory_usage_threshold_ = threshold; }
    bool is_memory_usage_critical() const;
    
    // Statistics and diagnostics
    size_t get_allocation_count() const { return allocation_count_.load(); }
    size_t get_deallocation_count() const { return deallocation_count_.load(); }
    double get_average_allocation_time() const;
    void print_memory_statistics() const;
    void print_device_info() const;
    
    // Memory defragmentation
    void defragment_gpu_memory();
    void garbage_collect();
    
    // Error handling
    bool check_cuda_error(cudaError_t error, const std::string& operation) const;
    std::string get_last_cuda_error() const;
    
private:
    void initialize_memory_pools();
    void cleanup_memory_pools();
    void track_allocation(void* ptr, size_t size);
    void untrack_allocation(void* ptr);
    size_t get_available_gpu_memory() const;
};

// Template implementations
template<typename T>
T* GPUResourceManager::allocate_gpu_array(size_t count) {
    return static_cast<T*>(allocate_gpu_memory(count * sizeof(T)));
}

template<typename T>
T* GPUResourceManager::allocate_host_array(size_t count) {
    return static_cast<T*>(allocate_host_memory(count * sizeof(T)));
}

template<typename T>
void GPUResourceManager::free_gpu_array(T* ptr) {
    free_gpu_memory(ptr);
}

template<typename T>
void GPUResourceManager::free_host_array(T* ptr) {
    free_host_memory(ptr);
}

// Utility class for RAII GPU memory management
template<typename T>
class GPUArray {
private:
    T* gpu_ptr_;
    size_t size_;
    GPUResourceManager* manager_;
    
public:
    GPUArray(GPUResourceManager* manager, size_t count)
        : gpu_ptr_(nullptr), size_(count), manager_(manager) {
        if (manager_ && count > 0) {
            gpu_ptr_ = manager_->allocate_gpu_array<T>(count);
        }
    }
    
    ~GPUArray() {
        if (gpu_ptr_ && manager_) {
            manager_->free_gpu_array(gpu_ptr_);
        }
    }
    
    // Move semantics
    GPUArray(GPUArray&& other) noexcept
        : gpu_ptr_(other.gpu_ptr_), size_(other.size_), manager_(other.manager_) {
        other.gpu_ptr_ = nullptr;
        other.size_ = 0;
        other.manager_ = nullptr;
    }
    
    GPUArray& operator=(GPUArray&& other) noexcept {
        if (this != &other) {
            if (gpu_ptr_ && manager_) {
                manager_->free_gpu_array(gpu_ptr_);
            }
            gpu_ptr_ = other.gpu_ptr_;
            size_ = other.size_;
            manager_ = other.manager_;
            other.gpu_ptr_ = nullptr;
            other.size_ = 0;
            other.manager_ = nullptr;
        }
        return *this;
    }
    
    // Disable copy
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;
    
    T* get() const { return gpu_ptr_; }
    size_t size() const { return size_; }
    bool is_valid() const { return gpu_ptr_ != nullptr; }
    
    void copy_from_host(const T* host_data, cudaStream_t stream = 0) {
        if (gpu_ptr_ && host_data && manager_) {
            manager_->copy_to_gpu(gpu_ptr_, host_data, size_ * sizeof(T), stream);
        }
    }
    
    void copy_to_host(T* host_data, cudaStream_t stream = 0) const {
        if (gpu_ptr_ && host_data && manager_) {
            manager_->copy_to_host(host_data, gpu_ptr_, size_ * sizeof(T), stream);
        }
    }
};

}