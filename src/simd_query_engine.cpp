// simd_query_engine.cpp
// SIMD-optimized query engine implementation

#include "simd_query_engine.h"
#include <algorithm>
#include <immintrin.h>
#include <cstring>

namespace event_aggregator {

SimdQueryEngine::SimdQueryEngine() {
    // Detect CPU features
    avx512_available_ = is_avx512_available();
    avx2_available_ = is_avx2_available();
}

SimdQueryEngine::~SimdQueryEngine() = default;

bool SimdQueryEngine::is_avx512_available() {
    // Check CPUID for AVX-512 support
    // Simplified check - in production, use proper CPUID detection
    #ifdef __AVX512F__
    return true;
    #else
    return false;
    #endif
}

bool SimdQueryEngine::is_avx2_available() {
    #ifdef __AVX2__
    return true;
    #else
    return false;
    #endif
}

std::vector<size_t> SimdQueryEngine::filter_by_region_device(
    const std::vector<AggCell>& bins,
    const std::vector<uint64_t>& timestamps,
    int region,
    int device,
    uint64_t start_ts,
    uint64_t end_ts
) {
    if (avx512_available_) {
        return filter_avx512(bins, timestamps, region, device, start_ts, end_ts);
    } else if (avx2_available_) {
        return filter_avx2(bins, timestamps, region, device, start_ts, end_ts);
    } else {
        return filter_scalar(bins, timestamps, region, device, start_ts, end_ts);
    }
}

std::vector<size_t> SimdQueryEngine::filter_scalar(
    const std::vector<AggCell>& bins,
    const std::vector<uint64_t>& timestamps,
    int region,
    int device,
    uint64_t start_ts,
    uint64_t end_ts
) {
    std::vector<size_t> result;
    result.reserve(bins.size());
    
    for (size_t i = 0; i < bins.size(); ++i) {
        // In a real implementation, we'd need region/device info per bin
        // For now, this is a placeholder
        if (timestamps[i] >= start_ts && timestamps[i] <= end_ts) {
            result.push_back(i);
        }
    }
    
    return result;
}

std::vector<size_t> SimdQueryEngine::filter_avx512(
    const std::vector<AggCell>& bins,
    const std::vector<uint64_t>& timestamps,
    int region,
    int device,
    uint64_t start_ts,
    uint64_t end_ts
) {
    std::vector<size_t> result;
    result.reserve(bins.size());
    
    // AVX-512 can process 8 uint64_t timestamps at once
    const size_t simd_width = 8;
    size_t i = 0;
    
    __m512i start_vec = _mm512_set1_epi64(static_cast<int64_t>(start_ts));
    __m512i end_vec = _mm512_set1_epi64(static_cast<int64_t>(end_ts));
    
    for (; i + simd_width <= timestamps.size(); i += simd_width) {
        __m512i ts_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&timestamps[i]));
        
        __mmask8 mask_ge = _mm512_cmpge_epu64_mask(ts_vec, start_vec);
        __mmask8 mask_le = _mm512_cmple_epu64_mask(ts_vec, end_vec);
        __mmask8 mask = _kand_mask8(mask_ge, mask_le);
        
        // Extract matching indices
        unsigned int mask_int = static_cast<unsigned int>(mask);
        for (int j = 0; j < simd_width; ++j) {
            if (mask_int & (1u << j)) {
                result.push_back(i + j);
            }
        }
    }
    
    // Handle remainder
    for (; i < timestamps.size(); ++i) {
        if (timestamps[i] >= start_ts && timestamps[i] <= end_ts) {
            result.push_back(i);
        }
    }
    
    return result;
}

std::vector<size_t> SimdQueryEngine::filter_avx2(
    const std::vector<AggCell>& bins,
    const std::vector<uint64_t>& timestamps,
    int region,
    int device,
    uint64_t start_ts,
    uint64_t end_ts
) {
    std::vector<size_t> result;
    result.reserve(bins.size());
    
    // AVX2 can process 4 uint64_t timestamps at once
    const size_t simd_width = 4;
    size_t i = 0;
    
    __m256i start_vec = _mm256_set1_epi64x(static_cast<int64_t>(start_ts));
    __m256i end_vec = _mm256_set1_epi64x(static_cast<int64_t>(end_ts));
    
    for (; i + simd_width <= timestamps.size(); i += simd_width) {
        __m256i ts_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&timestamps[i]));
        
        __m256i cmp_ge = _mm256_cmpgt_epi64(ts_vec, _mm256_sub_epi64(ts_vec, start_vec));
        __m256i cmp_le = _mm256_cmpgt_epi64(end_vec, _mm256_sub_epi64(end_vec, ts_vec));
        
        // Extract mask (simplified - actual implementation needs proper comparison)
        // This is a placeholder
    }
    
    // Handle remainder
    for (; i < timestamps.size(); ++i) {
        if (timestamps[i] >= start_ts && timestamps[i] <= end_ts) {
            result.push_back(i);
        }
    }
    
    return result;
}

SimdQueryEngine::AggregationResult SimdQueryEngine::aggregate(
    const std::vector<AggCell>& bins,
    const std::vector<size_t>& indices
) {
    AggregationResult result = {0, 0.0, -1.0f, -1.0f, 1e9f};
    
    for (size_t idx : indices) {
        const auto& bin = bins[idx];
        result.total_count += bin.count;
        result.sum_util += bin.sum_util;
        result.max_power_actual = std::max(result.max_power_actual, bin.max_power_actual);
        result.max_power_cap = std::max(result.max_power_cap, bin.max_power_cap);
        if (bin.min_power_cap < 1e9f) {
            result.min_power_cap = std::min(result.min_power_cap, bin.min_power_cap);
        }
    }
    
    return result;
}

} // namespace event_aggregator

