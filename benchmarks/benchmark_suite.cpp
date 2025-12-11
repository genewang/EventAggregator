// benchmark_suite.cpp
// Comprehensive benchmark suite with power profiling

#include "event_aggregator.h"
#include "gpu_telemetry.h"
#include "nsight_profiler.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <thread>
#include <algorithm>
#include <ctime>

using namespace event_aggregator;
using namespace std::chrono;

struct BenchmarkResults {
    double ingestion_rate_eps;      // events per second
    double query_latency_p50_ms;     // milliseconds
    double query_latency_p95_ms;
    double query_latency_p99_ms;
    double gpu_utilization_avg;      // percentage
    double power_consumption_avg;    // watts
    double memory_bandwidth_gbps;     // gigabytes per second
    size_t total_events;
    double total_time_seconds;
};

class BenchmarkSuite {
public:
    BenchmarkSuite(size_t num_events, int num_regions, int num_devices)
        : num_events_(num_events), num_regions_(num_regions), num_devices_(num_devices) {
        
        // Initialize aggregator
        GpuAggregator::Config config;
        config.num_regions = num_regions;
        config.num_devices_per_region = num_devices;
        config.max_events_per_batch = 16384;
        config.use_double_buffering = true;
        
        aggregator_ = std::make_unique<GpuAggregator>(config);
        
        // Initialize telemetry (if available)
        telemetry_ = create_telemetry_collector();
        if (telemetry_) {
            telemetry_->initialize();
        }
        
        // Initialize profiler
        profiler_.set_enabled(true);
    }
    
    BenchmarkResults run_ingestion_benchmark() {
        std::cout << "Running ingestion benchmark...\n";
        std::cout << "Target: " << num_events_ << " events\n";
        
        profiler_.start_session("ingestion_benchmark");
        
        // Generate test events
        auto events = generate_test_events();
        
        // Warm-up
        if (events.size() > 1000) {
            aggregator_->ingest_batch(events.data(), 1000);
            std::this_thread::sleep_for(milliseconds(100));
        }
        
        // Benchmark
        auto start = steady_clock::now();
        
        size_t events_ingested = 0;
        const size_t batch_size = 16384;
        
        for (size_t i = 0; i < events.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, events.size());
            size_t batch_count = batch_end - i;
            
            aggregator_->ingest_batch(events.data() + i, batch_count);
            events_ingested += batch_count;
        }
        
        // Wait for GPU to finish
        std::this_thread::sleep_for(milliseconds(500));
        
        auto end = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(end - start);
        double elapsed_seconds = elapsed.count() / 1000.0;
        
        // Collect telemetry
        double avg_gpu_util = 0.0;
        double avg_power = 0.0;
        if (telemetry_ && telemetry_->get_gpu_count() > 0) {
            std::vector<GpuTelemetry> metrics;
            telemetry_->collect_all(metrics);
            if (!metrics.empty()) {
                avg_gpu_util = metrics[0].utilization_gpu;
                avg_power = metrics[0].power_usage_watts;
            }
        }
        
        profiler_.stop_session();
        
        BenchmarkResults results;
        results.ingestion_rate_eps = events_ingested / elapsed_seconds;
        results.total_events = events_ingested;
        results.total_time_seconds = elapsed_seconds;
        results.gpu_utilization_avg = avg_gpu_util;
        results.power_consumption_avg = avg_power;
        
        std::cout << "Ingestion complete:\n";
        std::cout << "  Events: " << events_ingested << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) 
                  << elapsed_seconds << " seconds\n";
        std::cout << "  Rate: " << results.ingestion_rate_eps << " events/sec\n";
        std::cout << "  GPU Util: " << avg_gpu_util << "%\n";
        std::cout << "  Power: " << avg_power << " W\n";
        
        return results;
    }
    
    BenchmarkResults run_query_benchmark(int num_queries = 100) {
        std::cout << "\nRunning query benchmark...\n";
        std::cout << "Queries: " << num_queries << "\n";
        
        profiler_.start_session("query_benchmark");
        
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> region_dist(0, num_regions_ - 1);
        std::uniform_int_distribution<int> device_dist(0, num_devices_ - 1);
        
        for (int i = 0; i < num_queries; ++i) {
            QueryParams params;
            params.region = region_dist(rng);
            params.device = device_dist(rng);
            
            auto start = steady_clock::now();
            auto result = aggregator_->query(params);
            auto end = steady_clock::now();
            
            auto latency_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
            latencies.push_back(latency_ms);
        }
        
        std::sort(latencies.begin(), latencies.end());
        
        BenchmarkResults results;
        results.query_latency_p50_ms = latencies[latencies.size() / 2];
        results.query_latency_p95_ms = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        results.query_latency_p99_ms = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        
        profiler_.stop_session();
        
        std::cout << "Query latency:\n";
        std::cout << "  P50: " << std::fixed << std::setprecision(2) 
                  << results.query_latency_p50_ms << " ms\n";
        std::cout << "  P95: " << results.query_latency_p95_ms << " ms\n";
        std::cout << "  P99: " << results.query_latency_p99_ms << " ms\n";
        
        return results;
    }
    
    void run_power_profiling(size_t duration_seconds = 60) {
        std::cout << "\nRunning power profiling...\n";
        std::cout << "Duration: " << duration_seconds << " seconds\n";
        
        if (!telemetry_ || telemetry_->get_gpu_count() == 0) {
            std::cout << "Warning: Telemetry not available\n";
            return;
        }
        
        std::ofstream power_log("power_profile.csv");
        power_log << "timestamp,power_watts,util_gpu,util_mem,temp_celsius\n";
        
        auto start = steady_clock::now();
        auto events = generate_test_events();
        size_t event_idx = 0;
        
        while (true) {
            auto now = steady_clock::now();
            auto elapsed = duration_cast<seconds>(now - start).count();
            if (elapsed >= duration_seconds) break;
            
            // Ingest batch
            if (event_idx < events.size()) {
                size_t batch_size = std::min(16384UL, events.size() - event_idx);
                aggregator_->ingest_batch(events.data() + event_idx, batch_size);
                event_idx += batch_size;
            }
            
            // Collect telemetry
            std::vector<GpuTelemetry> metrics;
            telemetry_->collect_all(metrics);
            
            if (!metrics.empty()) {
                const auto& m = metrics[0];
                auto timestamp = duration_cast<milliseconds>(now - start).count();
                power_log << timestamp << ","
                         << m.power_usage_watts << ","
                         << m.utilization_gpu << ","
                         << m.utilization_memory << ","
                         << m.temperature_celsius << "\n";
            }
            
            std::this_thread::sleep_for(milliseconds(100));
        }
        
        power_log.close();
        std::cout << "Power profile saved to power_profile.csv\n";
    }
    
    void generate_report(const BenchmarkResults& ingest, const BenchmarkResults& query) {
        std::ofstream report("benchmark_report.md");
        
        report << "# Benchmark Report\n\n";
        report << "Generated: " << std::put_time(std::localtime(&std::time(nullptr)), "%Y-%m-%d %H:%M:%S") << "\n\n";
        
        report << "## Ingestion Performance\n\n";
        report << "- **Events**: " << ingest.total_events << "\n";
        report << "- **Time**: " << std::fixed << std::setprecision(2) 
               << ingest.total_time_seconds << " seconds\n";
        report << "- **Rate**: " << ingest.ingestion_rate_eps << " events/sec\n";
        report << "- **GPU Utilization**: " << ingest.gpu_utilization_avg << "%\n";
        report << "- **Power Consumption**: " << ingest.power_consumption_avg << " W\n\n";
        
        report << "## Query Performance\n\n";
        report << "- **P50 Latency**: " << query.query_latency_p50_ms << " ms\n";
        report << "- **P95 Latency**: " << query.query_latency_p95_ms << " ms\n";
        report << "- **P99 Latency**: " << query.query_latency_p99_ms << " ms\n\n";
        
        report << "## Profiling\n\n";
        report << "- Run with Nsight: `nsys profile --trace=cuda,nvtx ./benchmark_suite`\n";
        report << "- Power profile: `power_profile.csv`\n";
        
        report.close();
        std::cout << "\nBenchmark report saved to benchmark_report.md\n";
    }
    
private:
    std::vector<Event> generate_test_events() {
        std::vector<Event> events;
        events.reserve(num_events_);
        
        auto now = std::chrono::system_clock::now();
        uint64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
        uint64_t window_start = now_ts - 7 * 24 * 3600;
        
        std::mt19937_64 rng(12345);
        std::uniform_int_distribution<uint64_t> ts_dist(window_start, now_ts);
        std::uniform_int_distribution<int> region_dist(0, num_regions_ - 1);
        std::uniform_int_distribution<int> device_dist(0, num_devices_ - 1);
        std::uniform_real_distribution<float> util_dist(0.0f, 100.0f);
        std::uniform_real_distribution<float> power_dist(10.0f, 350.0f);
        
        std::vector<std::string> node_names = {
            "gpu-node-01", "gpu-node-02", "gpu-node-03", "gpu-node-04"
        };
        std::uniform_int_distribution<size_t> node_dist(0, node_names.size() - 1);
        
        // Simple hash function
        auto hash_node = [](const std::string& s) -> uint32_t {
            uint32_t h = 2166136261u;
            for (char c : s) {
                h ^= static_cast<uint32_t>(c);
                h *= 16777619u;
            }
            return h;
        };
        
        for (size_t i = 0; i < num_events_; ++i) {
            Event ev;
            ev.ts = ts_dist(rng);
            ev.region = static_cast<uint16_t>(region_dist(rng));
            ev.device = static_cast<uint16_t>(device_dist(rng));
            std::string node = node_names[node_dist(rng)];
            ev.node_name_hash = hash_node(node);
            ev.gpu_util = util_dist(rng);
            ev.power_cap = power_dist(rng);
            ev.power_actual = power_dist(rng) * 0.9f;
            events.push_back(ev);
        }
        
        return events;
    }
    
    size_t num_events_;
    int num_regions_;
    int num_devices_;
    std::unique_ptr<GpuAggregator> aggregator_;
    std::unique_ptr<GpuTelemetryCollector> telemetry_;
    NsightProfiler profiler_;
};

int main(int argc, char* argv[]) {
    size_t num_events = 1000000;  // 1M events default
    int num_regions = 16;
    int num_devices = 64;
    
    if (argc > 1) {
        num_events = std::stoull(argv[1]);
    }
    
    std::cout << "Event Aggregator - Benchmark Suite\n";
    std::cout << "==================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Events: " << num_events << "\n";
    std::cout << "  Regions: " << num_regions << "\n";
    std::cout << "  Devices: " << num_devices << "\n\n";
    
    BenchmarkSuite suite(num_events, num_regions, num_devices);
    
    // Run benchmarks
    auto ingest_results = suite.run_ingestion_benchmark();
    auto query_results = suite.run_query_benchmark();
    
    // Power profiling (optional)
    if (argc > 2 && std::string(argv[2]) == "--power") {
        suite.run_power_profiling(60);
    }
    
    // Generate report
    suite.generate_report(ingest_results, query_results);
    
    return 0;
}

