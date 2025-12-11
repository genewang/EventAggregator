// multi_gpu_demo.cpp
// Demonstration of multi-GPU aggregation (requires NCCL)

#ifdef ENABLE_NCCL

#include "event_aggregator.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace event_aggregator;

int main() {
    std::cout << "Event Aggregator - Multi-GPU Demo\n";
    std::cout << "==================================\n\n";
    
    // Check available GPUs
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "Available GPUs: " << num_gpus << "\n";
    
    if (num_gpus < 2) {
        std::cout << "Warning: Need at least 2 GPUs for multi-GPU demo\n";
        return 1;
    }
    
    // Configure multi-GPU aggregator
    MultiGpuAggregator::Config config;
    config.gpu_ids = {0, 1};  // Use first two GPUs
    config.num_regions = 16;
    config.num_devices_per_region = 64;
    config.bins_per_day = 24 * 60;
    config.days_window = 7;
    
    MultiGpuAggregator aggregator(config);
    
    std::cout << "Multi-GPU aggregator initialized with " 
              << config.gpu_ids.size() << " GPUs\n";
    
    // Generate and ingest events
    auto now = std::chrono::system_clock::now();
    uint64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    std::vector<Event> events(10000);
    for (int i = 0; i < 10000; ++i) {
        events[i].ts = now_ts - (10000 - i) * 60;
        events[i].region = i % 16;
        events[i].device = i % 64;
        events[i].node_name = "gpu-node-" + std::to_string(i % 8);
        events[i].gpu_util = 50.0f + (i % 50);
        events[i].power_cap = 300.0f;
        events[i].power_actual = 250.0f + (i % 50);
    }
    
    std::cout << "Ingesting " << events.size() << " events across GPUs...\n";
    aggregator.ingest_batch(events.data(), events.size());
    
    std::cout << "Synchronizing aggregates across GPUs...\n";
    aggregator.synchronize();
    
    // Query
    QueryParams params;
    params.region = 0;
    params.device = 0;
    
    auto result = aggregator.query(params);
    std::cout << "Query result: " << result.total_count << " events aggregated\n";
    
    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "Multi-GPU support not enabled. Rebuild with -DENABLE_NCCL=ON\n";
    return 1;
}

#endif

