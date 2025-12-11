// query_demo.cpp
// Demonstration of ad-hoc querying

#include "event_aggregator.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <ctime>

using namespace event_aggregator;

void print_query_result(const QueryResult& result) {
    std::cout << "\nQuery Result:\n";
    std::cout << "=============\n";
    std::cout << "Region: " << result.region << "\n";
    std::cout << "Device: " << result.device << "\n";
    std::cout << "Node: " << result.node_name << "\n";
    std::cout << "Time range: " << result.start_ts << " - " << result.end_ts << "\n";
    std::cout << "Total events: " << result.total_count << "\n";
    std::cout << "Average GPU util: " << std::fixed << std::setprecision(2) 
              << result.avg_gpu_util << "%\n";
    std::cout << "Max power (actual): " << result.max_power_actual << " W\n";
    std::cout << "Max power (cap): " << result.max_power_cap << " W\n";
    std::cout << "Min power (cap): " << result.min_power_cap << " W\n";
    std::cout << "Number of bins: " << result.bins.size() << "\n";
    
    if (!result.bins.empty()) {
        std::cout << "\nSample bins (first 5):\n";
        for (size_t i = 0; i < std::min(5UL, result.bins.size()); ++i) {
            const auto& bin = result.bins[i];
            auto time_str = std::chrono::system_clock::from_time_t(result.bin_timestamps[i]);
            auto time_t = std::chrono::system_clock::to_time_t(time_str);
            std::cout << "  " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                      << " - Count: " << bin.count
                      << ", Avg util: " << (bin.count > 0 ? bin.sum_util / bin.count : 0.0)
                      << "%, Max power: " << bin.max_power_actual << " W\n";
        }
    }
}

int main() {
    std::cout << "Event Aggregator - Query Demo\n";
    std::cout << "=============================\n\n";
    
    // Create aggregator (same config as ingest demo)
    GpuAggregator::Config config;
    config.num_regions = 16;
    config.num_devices_per_region = 64;
    config.bins_per_day = 24 * 60;
    config.days_window = 7;
    
    GpuAggregator aggregator(config);
    
    // First, ingest some sample data
    std::cout << "Ingesting sample events...\n";
    
    auto now = std::chrono::system_clock::now();
    uint64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    std::vector<Event> sample_events(1000);
    for (int i = 0; i < 1000; ++i) {
        sample_events[i].ts = now_ts - (1000 - i) * 60;  // spread over ~16 hours
        sample_events[i].region = 3;
        sample_events[i].device = 5;
        sample_events[i].node_name = "gpu-node-03";
        sample_events[i].gpu_util = 50.0f + (i % 50);
        sample_events[i].power_cap = 300.0f;
        sample_events[i].power_actual = 250.0f + (i % 50);
    }
    
    aggregator.ingest_batch(sample_events.data(), sample_events.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Sample events ingested.\n\n";
    
    // Query 1: Specific region and device
    std::cout << "Query 1: Region 3, Device 5 (last 7 days)\n";
    QueryParams params1;
    params1.region = 3;
    params1.device = 5;
    
    auto result1 = aggregator.query(params1);
    print_query_result(result1);
    
    // Query 2: Specific time range
    std::cout << "\n\nQuery 2: Region 3, Device 5 (last 24 hours)\n";
    QueryParams params2;
    params2.region = 3;
    params2.device = 5;
    params2.end_ts = now_ts;
    params2.start_ts = now_ts - 24 * 3600;
    
    auto result2 = aggregator.query(params2);
    print_query_result(result2);
    
    // Query 3: JSON output
    std::cout << "\n\nQuery 3: JSON output\n";
    std::cout << result1.to_json() << "\n";
    
    return 0;
}

