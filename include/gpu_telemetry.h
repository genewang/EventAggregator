#pragma once

// GPU Telemetry Integration
// Provides real-time GPU metrics via NVML/DCGM for self-monitoring
// and performance optimization

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace event_aggregator {

// GPU telemetry metrics
struct GpuTelemetry {
    uint32_t gpu_id;
    float utilization_gpu;      // 0-100%
    float utilization_memory;    // 0-100%
    uint64_t memory_used_bytes;
    uint64_t memory_total_bytes;
    float power_usage_watts;
    float power_limit_watts;
    float temperature_celsius;
    uint32_t clock_graphics_mhz;
    uint32_t clock_memory_mhz;
    uint64_t timestamp_ns;
    
    // SM-level metrics (if available)
    float sm_throughput;        // Instructions per cycle
    float warp_efficiency;      // Active warps / total warps
    uint64_t l2_cache_hits;
    uint64_t l2_cache_misses;
};

// Telemetry collector interface
class GpuTelemetryCollector {
public:
    virtual ~GpuTelemetryCollector() = default;
    
    // Initialize telemetry collection
    virtual bool initialize() = 0;
    
    // Collect metrics for a specific GPU
    virtual bool collect(uint32_t gpu_id, GpuTelemetry& metrics) = 0;
    
    // Collect metrics for all GPUs
    virtual bool collect_all(std::vector<GpuTelemetry>& metrics) = 0;
    
    // Set power cap (if supported)
    virtual bool set_power_cap(uint32_t gpu_id, float watts) = 0;
    
    // Get available GPU count
    virtual uint32_t get_gpu_count() const = 0;
    
    // Shutdown
    virtual void shutdown() = 0;
};

// NVML-based implementation (requires libnvidia-ml)
class NvmlTelemetryCollector : public GpuTelemetryCollector {
public:
    NvmlTelemetryCollector();
    ~NvmlTelemetryCollector() override;
    
    bool initialize() override;
    bool collect(uint32_t gpu_id, GpuTelemetry& metrics) override;
    bool collect_all(std::vector<GpuTelemetry>& metrics) override;
    bool set_power_cap(uint32_t gpu_id, float watts) override;
    uint32_t get_gpu_count() const override;
    void shutdown() override;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Factory function
std::unique_ptr<GpuTelemetryCollector> create_telemetry_collector();

} // namespace event_aggregator

