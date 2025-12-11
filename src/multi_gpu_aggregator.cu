// multi_gpu_aggregator.cu
// NCCL-based multi-GPU aggregation support

#ifdef ENABLE_NCCL

#include "event_aggregator.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <nccl.h>

namespace event_aggregator {

class MultiGpuAggregator::Impl {
public:
    Config config_;
    std::vector<std::unique_ptr<GpuAggregator>> aggregators_;
    ncclComm_t* comms_;
    int num_gpus_;
    
    Impl(const Config& config) : config_(config) {
        num_gpus_ = static_cast<int>(config.gpu_ids.size());
        comms_ = new ncclComm_t[num_gpus_];
        
        // Initialize NCCL
        ncclUniqueId nccl_id;
        if (ncclGetUniqueId(&nccl_id) != ncclSuccess) {
            throw std::runtime_error("Failed to get NCCL unique ID");
        }
        
        // Create communicators for each GPU
        ncclComm_t* comms = new ncclComm_t[num_gpus_];
        cudaStream_t* streams = new cudaStream_t[num_gpus_];
        
        for (int i = 0; i < num_gpus_; ++i) {
            cudaSetDevice(config.gpu_ids[i]);
            cudaStreamCreate(&streams[i]);
        }
        
        if (ncclCommInitAll(comms, num_gpus_, config.gpu_ids.data()) != ncclSuccess) {
            throw std::runtime_error("Failed to initialize NCCL communicators");
        }
        
        comms_ = comms;
        
        // Create per-GPU aggregators
        for (int gpu_id : config.gpu_ids) {
            GpuAggregator::Config agg_config;
            agg_config.num_regions = config.num_regions;
            agg_config.num_devices_per_region = config.num_devices_per_region;
            agg_config.bins_per_day = config.bins_per_day;
            agg_config.days_window = config.days_window;
            agg_config.gpu_id = gpu_id;
            
            aggregators_.push_back(std::make_unique<GpuAggregator>(agg_config));
        }
    }
    
    ~Impl() {
        if (comms_) {
            for (int i = 0; i < num_gpus_; ++i) {
                ncclCommDestroy(comms_[i]);
            }
            delete[] comms_;
        }
    }
    
    void ingest_batch(const Event* events, int n_events) {
        // Distribute events across GPUs (round-robin or hash-based)
        // For simplicity, round-robin
        int events_per_gpu = (n_events + num_gpus_ - 1) / num_gpus_;
        
        for (int i = 0; i < num_gpus_; ++i) {
            int start = i * events_per_gpu;
            int end = std::min(start + events_per_gpu, n_events);
            if (start < n_events) {
                aggregators_[i]->ingest_batch(events + start, end - start);
            }
        }
    }
    
    QueryResult query(const QueryParams& params) {
        // Query from first GPU (in production, would aggregate results)
        return aggregators_[0]->query(params);
    }
    
    void synchronize() {
        // All-reduce aggregates across all GPUs
        // This is a simplified version - in production, you'd need to:
        // 1. Get device pointers from each aggregator
        // 2. Perform NCCL all-reduce on the aggregation arrays
        // 3. Update each GPU's local state
        
        // Placeholder: synchronize streams
        for (int i = 0; i < num_gpus_; ++i) {
            cudaSetDevice(config_.gpu_ids[i]);
            cudaDeviceSynchronize();
        }
    }
};

MultiGpuAggregator::MultiGpuAggregator(const Config& config)
    : pimpl_(std::make_unique<Impl>(config))
{
}

MultiGpuAggregator::~MultiGpuAggregator() = default;

void MultiGpuAggregator::ingest_batch(const Event* events, int n_events) {
    pimpl_->ingest_batch(events, n_events);
}

QueryResult MultiGpuAggregator::query(const QueryParams& params) {
    return pimpl_->query(params);
}

void MultiGpuAggregator::synchronize() {
    pimpl_->synchronize();
}

} // namespace event_aggregator

#endif // ENABLE_NCCL

