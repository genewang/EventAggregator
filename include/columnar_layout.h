#pragma once

// Columnar (Structure of Arrays) Layout for GPU
// Enables warp-coalesced memory access patterns for better performance

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace event_aggregator {

// Columnar event batch - Structure of Arrays (SoA) layout
// Better for GPU processing than Array of Structures (AoS)
struct ColumnarEventBatch {
    // Timestamps (column 0)
    uint64_t* timestamps;
    
    // Region/Device IDs (columns 1-2)
    uint16_t* regions;
    uint16_t* devices;
    uint32_t* node_name_hashes;
    
    // Metrics (columns 3-5)
    float* gpu_utils;
    float* power_caps;
    float* power_actuals;
    
    size_t count;
    size_t capacity;
    
    // Device-side pointers (for CUDA kernels)
    uint64_t* d_timestamps;
    uint16_t* d_regions;
    uint16_t* d_devices;
    uint32_t* d_node_name_hashes;
    float* d_gpu_utils;
    float* d_power_caps;
    float* d_power_actuals;
    
    // Memory management
    bool is_pinned;
    bool is_on_device;
};

// Columnar batch manager
class ColumnarBatchManager {
public:
    ColumnarBatchManager(size_t capacity, bool use_pinned = true);
    ~ColumnarBatchManager();
    
    // Convert AoS events to SoA columnar format
    void pack_events(const struct Event* events, size_t count);
    
    // Get columnar batch (for GPU processing)
    const ColumnarEventBatch& get_batch() const { return batch_; }
    ColumnarEventBatch& get_batch() { return batch_; }
    
    // Transfer to device (async)
    void transfer_to_device(cudaStream_t stream);
    
    // Transfer from device (async)
    void transfer_from_device(cudaStream_t stream);
    
    // Clear batch
    void clear();
    
    size_t count() const { return batch_.count; }
    size_t capacity() const { return batch_.capacity; }
    
private:
    ColumnarEventBatch batch_;
    void allocate_host();
    void allocate_device();
    void free_all();
};

} // namespace event_aggregator

