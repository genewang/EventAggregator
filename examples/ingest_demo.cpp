// ingest_demo.cpp
// Demonstration of high-throughput event ingestion

#include "event_aggregator.h"
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace event_aggregator;

int main() {
    std::cout << "Event Aggregator - Ingestion Demo\n";
    std::cout << "==================================\n\n";
    
    // Configure aggregator
    GpuAggregator::Config config;
    config.num_regions = 16;
    config.num_devices_per_region = 64;
    config.bins_per_day = 24 * 60;  // per-minute bins
    config.days_window = 7;
    config.max_events_per_batch = 16384;
    config.use_double_buffering = true;
    config.num_streams = 2;
    config.gpu_id = 0;
    
    GpuAggregator aggregator(config);
    
    // Generate synthetic events
    auto now = std::chrono::system_clock::now();
    uint64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    uint64_t window_start = now_ts - static_cast<uint64_t>(7) * 24 * 3600;
    
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint64_t> ts_dist(window_start, now_ts);
    std::uniform_int_distribution<int> region_dist(0, config.num_regions - 1);
    std::uniform_int_distribution<int> device_dist(0, config.num_devices_per_region - 1);
    std::uniform_real_distribution<float> util_dist(0.0f, 100.0f);
    std::uniform_real_distribution<float> power_dist(10.0f, 350.0f);
    
    std::vector<std::string> node_names = {
        "gpu-node-01", "gpu-node-02", "gpu-node-03", "gpu-node-04",
        "gpu-node-05", "gpu-node-06", "gpu-node-07", "gpu-node-08"
    };
    std::uniform_int_distribution<size_t> node_dist(0, node_names.size() - 1);
    
    // Simulate 100k events/sec for 10 seconds
    const int target_rate = 100000;  // events/sec
    const int duration_seconds = 10;
    const int batch_size = config.max_events_per_batch;
    const int batches_per_second = (target_rate + batch_size - 1) / batch_size;
    const auto batch_interval = std::chrono::milliseconds(1000 / batches_per_second);
    
    std::vector<Event> events(batch_size);
    
    auto start_time = std::chrono::steady_clock::now();
    int total_events = 0;
    int batch_count = 0;
    
    std::cout << "Ingesting events at target rate: " << target_rate << " events/sec\n";
    std::cout << "Batch size: " << batch_size << ", Batches/sec: " << batches_per_second << "\n\n";
    
    while (true) {
        auto batch_start = std::chrono::steady_clock::now();
        
        // Generate batch
        int actual_batch_size = std::min(batch_size, 
            target_rate * duration_seconds - total_events);
        if (actual_batch_size <= 0) break;
        
        for (int i = 0; i < actual_batch_size; ++i) {
            events[i].ts = ts_dist(rng);
            events[i].region = static_cast<uint16_t>(region_dist(rng));
            events[i].device = static_cast<uint16_t>(device_dist(rng));
            events[i].node_name = node_names[node_dist(rng)];
            events[i].gpu_util = util_dist(rng);
            events[i].power_cap = power_dist(rng);
            events[i].power_actual = power_dist(rng) * 0.9f;  // slightly below cap
        }
        
        // Ingest batch
        aggregator.ingest_batch(events.data(), actual_batch_size);
        
        total_events += actual_batch_size;
        batch_count++;
        
        // Rate limiting
        auto elapsed = std::chrono::steady_clock::now() - batch_start;
        if (elapsed < batch_interval) {
            std::this_thread::sleep_for(batch_interval - elapsed);
        }
        
        // Progress update
        if (batch_count % 10 == 0) {
            auto runtime = std::chrono::steady_clock::now() - start_time;
            auto runtime_sec = std::chrono::duration_cast<std::chrono::seconds>(runtime).count();
            double current_rate = (runtime_sec > 0) ? total_events / static_cast<double>(runtime_sec) : 0.0;
            std::cout << "Batches: " << batch_count 
                      << ", Events: " << total_events
                      << ", Rate: " << std::fixed << std::setprecision(0) << current_rate << " events/sec\n";
        }
        
        if (total_events >= target_rate * duration_seconds) break;
    }
    
    // Wait for GPU work to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    double actual_rate = (total_elapsed > 0) 
        ? (total_events * 1000.0 / total_elapsed) : 0.0;
    
    std::cout << "\nIngestion complete!\n";
    std::cout << "Total events: " << total_events << "\n";
    std::cout << "Total time: " << total_elapsed << " ms\n";
    std::cout << "Average rate: " << std::fixed << std::setprecision(2) 
              << actual_rate << " events/sec\n";
    std::cout << "Aggregator rate: " << aggregator.get_ingestion_rate() 
              << " events/sec\n";
    
    return 0;
}

