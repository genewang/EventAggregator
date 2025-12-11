// columnar_layout.cpp
// Columnar (SoA) layout implementation

#include "columnar_layout.h"
#include "event_aggregator.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace event_aggregator {

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        throw std::runtime_error("CUDA error: " +                 \
            std::string(cudaGetErrorString(err)));                \
    }                                                               \
} while(0)

ColumnarBatchManager::ColumnarBatchManager(size_t capacity, bool use_pinned)
    : is_pinned_(use_pinned) {
    batch_.capacity = capacity;
    batch_.count = 0;
    batch_.is_pinned = use_pinned;
    batch_.is_on_device = false;
    
    allocate_host();
    allocate_device();
}

ColumnarBatchManager::~ColumnarBatchManager() {
    free_all();
}

void ColumnarBatchManager::allocate_host() {
    size_t size = batch_.capacity * sizeof(uint64_t);
    
    if (is_pinned_) {
        CUDA_CHECK(cudaHostAlloc(&batch_.timestamps, size, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.regions, batch_.capacity * sizeof(uint16_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.devices, batch_.capacity * sizeof(uint16_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.node_name_hashes, batch_.capacity * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.gpu_utils, batch_.capacity * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.power_caps, batch_.capacity * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&batch_.power_actuals, batch_.capacity * sizeof(float), cudaHostAllocDefault));
    } else {
        batch_.timestamps = new uint64_t[batch_.capacity];
        batch_.regions = new uint16_t[batch_.capacity];
        batch_.devices = new uint16_t[batch_.capacity];
        batch_.node_name_hashes = new uint32_t[batch_.capacity];
        batch_.gpu_utils = new float[batch_.capacity];
        batch_.power_caps = new float[batch_.capacity];
        batch_.power_actuals = new float[batch_.capacity];
    }
}

void ColumnarBatchManager::allocate_device() {
    size_t size = batch_.capacity * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&batch_.d_timestamps, size));
    CUDA_CHECK(cudaMalloc(&batch_.d_regions, batch_.capacity * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&batch_.d_devices, batch_.capacity * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&batch_.d_node_name_hashes, batch_.capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&batch_.d_gpu_utils, batch_.capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_.d_power_caps, batch_.capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_.d_power_actuals, batch_.capacity * sizeof(float)));
}

void ColumnarBatchManager::free_all() {
    if (batch_.timestamps) {
        if (is_pinned_) {
            cudaFreeHost(batch_.timestamps);
            cudaFreeHost(batch_.regions);
            cudaFreeHost(batch_.devices);
            cudaFreeHost(batch_.node_name_hashes);
            cudaFreeHost(batch_.gpu_utils);
            cudaFreeHost(batch_.power_caps);
            cudaFreeHost(batch_.power_actuals);
        } else {
            delete[] batch_.timestamps;
            delete[] batch_.regions;
            delete[] batch_.devices;
            delete[] batch_.node_name_hashes;
            delete[] batch_.gpu_utils;
            delete[] batch_.power_caps;
            delete[] batch_.power_actuals;
        }
    }
    
    if (batch_.d_timestamps) {
        cudaFree(batch_.d_timestamps);
        cudaFree(batch_.d_regions);
        cudaFree(batch_.d_devices);
        cudaFree(batch_.d_node_name_hashes);
        cudaFree(batch_.d_gpu_utils);
        cudaFree(batch_.d_power_caps);
        cudaFree(batch_.d_power_actuals);
    }
}

void ColumnarBatchManager::pack_events(const Event* events, size_t count) {
    if (count > batch_.capacity) {
        throw std::runtime_error("Event count exceeds batch capacity");
    }
    
    batch_.count = count;
    
    // Pack into columnar format (SoA)
    for (size_t i = 0; i < count; ++i) {
        batch_.timestamps[i] = events[i].ts;
        batch_.regions[i] = events[i].region;
        batch_.devices[i] = events[i].device;
        batch_.node_name_hashes[i] = events[i].node_name_hash;
        batch_.gpu_utils[i] = events[i].gpu_util;
        batch_.power_caps[i] = events[i].power_cap;
        batch_.power_actuals[i] = events[i].power_actual;
    }
}

void ColumnarBatchManager::transfer_to_device(cudaStream_t stream) {
    size_t count_bytes = batch_.count * sizeof(uint64_t);
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_timestamps, batch_.timestamps, count_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_regions, batch_.regions,
        batch_.count * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_devices, batch_.devices,
        batch_.count * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_node_name_hashes, batch_.node_name_hashes,
        batch_.count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_gpu_utils, batch_.gpu_utils,
        batch_.count * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_power_caps, batch_.power_caps,
        batch_.count * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch_.d_power_actuals, batch_.power_actuals,
        batch_.count * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    batch_.is_on_device = true;
}

void ColumnarBatchManager::transfer_from_device(cudaStream_t stream) {
    size_t count_bytes = batch_.count * sizeof(uint64_t);
    CUDA_CHECK(cudaMemcpyAsync(batch_.timestamps, batch_.d_timestamps, count_bytes,
        cudaMemcpyDeviceToHost, stream));
    // ... similar for other columns if needed
    
    batch_.is_on_device = false;
}

void ColumnarBatchManager::clear() {
    batch_.count = 0;
}

} // namespace event_aggregator

