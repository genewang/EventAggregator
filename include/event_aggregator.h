#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

namespace event_aggregator {

// Forward declarations
class GpuAggregator;
class QueryResult;

// Event structure matching the ingestion format
// Note: node_name_hash is a hash of the node name for GPU processing
// The actual node_name is stored in a host-side mapping
struct Event {
    uint64_t ts;            // epoch seconds
    uint16_t region;        // region id
    uint16_t device;        // device id
    uint32_t node_name_hash; // hash of node name (for GPU processing)
    float gpu_util;         // 0..100
    float power_cap;        // watts (power cap setting)
    float power_actual;     // watts (actual power consumption)
};

// Aggregated cell for a time bin
struct AggCell {
    unsigned long long count;      // number of events
    double sum_util;                // sum of gpu_util
    float max_power_actual;        // maximum actual power
    float max_power_cap;           // maximum power cap
    float min_power_cap;           // minimum power cap
};

// Query parameters
struct QueryParams {
    int region = -1;                // -1 for all regions
    int device = -1;                // -1 for all devices
    std::string node_name;          // empty for all nodes
    uint64_t start_ts = 0;          // 0 for auto (7 days ago)
    uint64_t end_ts = 0;            // 0 for auto (now)
    int bin_size_seconds = 60;     // aggregation bin size
};

// Query result
struct QueryResult {
    std::vector<AggCell> bins;
    std::vector<uint64_t> bin_timestamps;
    int region;
    int device;
    std::string node_name;
    uint64_t start_ts;
    uint64_t end_ts;
    
    // Computed statistics
    unsigned long long total_count;
    double avg_gpu_util;
    float max_power_actual;
    float max_power_cap;
    float min_power_cap;
    
    // Convert to JSON string (for REST API)
    std::string to_json() const;
};

// Main aggregator class
class GpuAggregator {
public:
    // Configuration
    struct Config {
        int num_regions = 16;
        int num_devices_per_region = 64;
        int bins_per_day = 24 * 60;        // per-minute bins
        int days_window = 7;
        int max_events_per_batch = 16384;
        int gpu_threads = 256;
        int shared_mem_reduce_size = 1024;  // threads per block for reduction
        bool use_double_buffering = true;
        int num_streams = 2;                // for double-buffering
        int gpu_id = 0;                     // which GPU to use
    };
    
    explicit GpuAggregator(const Config& config = Config());
    ~GpuAggregator();
    
    // Non-copyable
    GpuAggregator(const GpuAggregator&) = delete;
    GpuAggregator& operator=(const GpuAggregator&) = delete;
    
    // Moveable
    GpuAggregator(GpuAggregator&&) noexcept;
    GpuAggregator& operator=(GpuAggregator&&) noexcept;
    
    // Ingest a batch of events (async, thread-safe)
    void ingest_batch(const Event* events, int n_events);
    
    // Query aggregated results
    QueryResult query(const QueryParams& params);
    
    // Update sliding window (call periodically to evict old data)
    void update_window(uint64_t current_time);
    
    // Get current window bounds
    std::pair<uint64_t, uint64_t> get_window_bounds() const;
    
    // Statistics
    size_t get_total_events_ingested() const { return total_events_ingested_; }
    double get_ingestion_rate() const;  // events/sec
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    size_t total_events_ingested_ = 0;
    std::chrono::steady_clock::time_point start_time_;
};

// Multi-GPU aggregator (requires NCCL)
#ifdef ENABLE_NCCL
class MultiGpuAggregator {
public:
    struct Config {
        std::vector<int> gpu_ids;
        int num_regions = 16;
        int num_devices_per_region = 64;
        int bins_per_day = 24 * 60;
        int days_window = 7;
    };
    
    explicit MultiGpuAggregator(const Config& config);
    ~MultiGpuAggregator();
    
    void ingest_batch(const Event* events, int n_events);
    QueryResult query(const QueryParams& params);
    void synchronize();  // Sync aggregates across GPUs
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};
#endif

} // namespace event_aggregator

