#pragma once

// Tiered Storage System
// Hot (GPU HBM) -> Warm (CPU RAM) -> Cold (Disk/NVMe)

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include "event_aggregator.h"

namespace event_aggregator {

// Storage tier definitions
enum class StorageTier {
    HOT,   // 0-24h: GPU HBM (fastest, limited capacity)
    WARM,  // 24h-7d: CPU RAM (fast, larger capacity)
    COLD   // >7d: Disk/NVMe (slower, unlimited capacity)
};

// Time-based tier boundaries
struct TierConfig {
    uint64_t hot_duration_seconds = 24 * 3600;    // 24 hours
    uint64_t warm_duration_seconds = 7 * 24 * 3600; // 7 days
    size_t hot_max_bins = 1440;  // 24h * 60 min
    size_t warm_max_bins = 10080; // 7d * 24h * 60 min
};

// Tiered storage manager
class TieredStorageManager {
public:
    explicit TieredStorageManager(const TierConfig& config = TierConfig());
    ~TieredStorageManager();
    
    // Determine which tier a timestamp belongs to
    StorageTier get_tier(uint64_t timestamp) const;
    
    // Spill hot data to warm tier
    void spill_hot_to_warm(uint64_t current_time);
    
    // Spill warm data to cold tier
    void spill_warm_to_cold(uint64_t current_time, const std::string& cold_path);
    
    // Query across tiers (transparent)
    QueryResult query_cross_tier(const QueryParams& params);
    
    // Get tier statistics
    struct TierStats {
        size_t hot_bins;
        size_t warm_bins;
        size_t cold_bins;
        size_t hot_memory_bytes;
        size_t warm_memory_bytes;
        size_t cold_disk_bytes;
    };
    TierStats get_stats() const;
    
private:
    TierConfig config_;
    uint64_t window_start_ts_;
    
    // Hot tier (GPU)
    std::unique_ptr<GpuAggregator> hot_aggregator_;
    
    // Warm tier (CPU RAM) - compressed columnar format
    struct WarmTier {
        std::vector<AggCell> bins;
        std::vector<uint64_t> timestamps;
        size_t start_bin;
        size_t end_bin;
    };
    std::vector<WarmTier> warm_tiers_;
    
    // Cold tier (disk) - Parquet-like format
    std::string cold_storage_path_;
};

} // namespace event_aggregator

