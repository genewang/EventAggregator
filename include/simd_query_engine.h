#pragma once

// SIMD-Optimized CPU Query Engine
// Uses AVX-512 for vectorized filtering and aggregation

#include "event_aggregator.h"
#include <vector>
#include <cstdint>

namespace event_aggregator {

// SIMD query engine for CPU-side ad-hoc queries
class SimdQueryEngine {
public:
    SimdQueryEngine();
    ~SimdQueryEngine();
    
    // Vectorized filter by region/device
    // Returns indices of matching bins
    std::vector<size_t> filter_by_region_device(
        const std::vector<AggCell>& bins,
        const std::vector<uint64_t>& timestamps,
        int region,
        int device,
        uint64_t start_ts,
        uint64_t end_ts
    );
    
    // Vectorized aggregation (sum, max, min)
    struct AggregationResult {
        unsigned long long total_count;
        double sum_util;
        float max_power_actual;
        float max_power_cap;
        float min_power_cap;
    };
    
    AggregationResult aggregate(
        const std::vector<AggCell>& bins,
        const std::vector<size_t>& indices
    );
    
    // Check if AVX-512 is available
    static bool is_avx512_available();
    
    // Check if AVX2 is available
    static bool is_avx2_available();
    
private:
    // Fallback scalar implementation
    std::vector<size_t> filter_scalar(
        const std::vector<AggCell>& bins,
        const std::vector<uint64_t>& timestamps,
        int region,
        int device,
        uint64_t start_ts,
        uint64_t end_ts
    );
    
    // AVX-512 optimized filter
    std::vector<size_t> filter_avx512(
        const std::vector<AggCell>& bins,
        const std::vector<uint64_t>& timestamps,
        int region,
        int device,
        uint64_t start_ts,
        uint64_t end_ts
    );
    
    // AVX2 optimized filter
    std::vector<size_t> filter_avx2(
        const std::vector<AggCell>& bins,
        const std::vector<uint64_t>& timestamps,
        int region,
        int device,
        uint64_t start_ts,
        uint64_t end_ts
    );
    
    bool avx512_available_;
    bool avx2_available_;
};

} // namespace event_aggregator

