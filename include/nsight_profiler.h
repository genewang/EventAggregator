#pragma once

// Nsight Profiling Integration
// Provides hooks for Nsight Compute/Systems profiling

#include <string>
#include <cstdint>
#include <memory>

namespace event_aggregator {

// Nsight profiler interface
class NsightProfiler {
public:
    NsightProfiler();
    ~NsightProfiler();
    
    // Start profiling session
    void start_session(const std::string& session_name);
    
    // Stop profiling session
    void stop_session();
    
    // Mark kernel launch (for Nsight Compute)
    void mark_kernel_launch(
        const std::string& kernel_name,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z,
        size_t shared_mem_bytes
    );
    
    // Mark memory transfer
    void mark_memory_transfer(
        const std::string& transfer_name,
        size_t bytes,
        bool host_to_device
    );
    
    // Enable/disable profiling
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Get profiling statistics
    struct ProfilingStats {
        uint64_t kernel_launches;
        uint64_t memory_transfers;
        uint64_t total_kernel_time_ns;
        uint64_t total_transfer_time_ns;
        size_t total_transfer_bytes;
    };
    ProfilingStats get_stats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    bool enabled_;
};

// RAII profiler scope
class ProfilerScope {
public:
    ProfilerScope(NsightProfiler& profiler, const std::string& name);
    ~ProfilerScope();
    
private:
    NsightProfiler& profiler_;
    std::string name_;
};

// Convenience macros
#define NSIGHT_PROFILE_SCOPE(profiler, name) \
    ProfilerScope _nsight_scope(profiler, name)

#define NSIGHT_MARK_KERNEL(profiler, name, gx, gy, gz, bx, by, bz, smem) \
    profiler.mark_kernel_launch(name, gx, gy, gz, bx, by, bz, smem)

} // namespace event_aggregator

