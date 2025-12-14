# Event Aggregator - Modern C++ CUDA Event Processing System

A high-performance event aggregation system built with Modern C++ and CUDA, designed to handle **100k+ events/sec** with ad-hoc queries over sliding time windows.

## Features

- **High-Throughput Ingestion**: Sustained 100k+ events/sec using CUDA-accelerated aggregation
- **Optimized Memory Layout**: Power-of-two circular buffer with efficient sliding window rotation
- **Reduced Contention**: Per-block shared-memory reduction to minimize atomic operations
- **Double-Buffering**: Pinned pre-allocated buffers with async transfers for maximum throughput
- **Multi-GPU Support**: NCCL-based inter-GPU aggregation (optional)
- **Ad-Hoc Queries**: Fast CPU-side queries for last 7 days by region/device
- **Dashboard Ready**: REST API interface for modern dashboard integration

## Architecture

### Event Structure
- **Timestamp**: Epoch seconds
- **Region/Device**: Geographic and device identifiers
- **Node Name**: Hostname/IP identifier
- **GPU Utilization**: 0-100%
- **Power Cap**: Power limit setting (watts)
- **Power Actual**: Actual power consumption (watts)

### Aggregation Strategy
- **Time Bins**: Per-minute bins over 7-day sliding window (10,080 bins)
- **Aggregates**: Count, sum of GPU util, max/min power metrics
- **Storage**: ~2.5 GB GPU memory for 16 regions × 64 devices × 10,080 bins

### Optimizations

1. **Per-Block Shared Memory Reduction**: Reduces atomic contention by accumulating within blocks before global updates
2. **Power-of-Two Layout**: Enables efficient bit-masking for circular buffer rotation
3. **Double-Buffering**: Overlaps CPU event generation with GPU processing
4. **Pinned Memory**: Fast DMA transfers via `cudaHostAlloc`

## Building

### Prerequisites
- CUDA Toolkit 11.0+ (with compute capability 7.5+)
- CMake 3.18+
- C++17 compatible compiler (GCC 7+, Clang 8+, MSVC 2017+)
- (Optional) NCCL for multi-GPU support

### Build Steps

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"  # Adjust for your GPUs
cmake --build . -j$(nproc)
```

### Build Options

- `BUILD_EXAMPLES=ON`: Build example programs (default: ON)
- `ENABLE_NCCL=ON`: Enable NCCL for multi-GPU support (default: OFF)

Example with NCCL:
```bash
cmake .. -DENABLE_NCCL=ON
```

## Usage

### Basic Ingestion

```cpp
#include "event_aggregator.h"

using namespace event_aggregator;

// Configure aggregator
GpuAggregator::Config config;
config.num_regions = 16;
config.num_devices_per_region = 64;
config.gpu_id = 0;

GpuAggregator aggregator(config);

// Create events
std::vector<Event> events(1000);
for (auto& ev : events) {
    ev.ts = get_current_timestamp();
    ev.region = 3;
    ev.device = 5;
    ev.node_name = "gpu-node-01";
    ev.gpu_util = 75.5f;
    ev.power_cap = 300.0f;
    ev.power_actual = 280.0f;
}

// Ingest batch
aggregator.ingest_batch(events.data(), events.size());
```

### Querying

```cpp
// Query specific region/device
QueryParams params;
params.region = 3;
params.device = 5;
params.start_ts = 0;  // Auto: 7 days ago
params.end_ts = 0;    // Auto: now

QueryResult result = aggregator.query(params);

std::cout << "Total events: " << result.total_count << "\n";
std::cout << "Avg GPU util: " << result.avg_gpu_util << "%\n";
std::cout << "Max power: " << result.max_power_actual << " W\n";
```

### Window Management

```cpp
// Update sliding window (call periodically)
uint64_t current_time = get_current_timestamp();
aggregator.update_window(current_time);

// Get window bounds
auto [start, end] = aggregator.get_window_bounds();
```

### Multi-GPU (with NCCL)

```cpp
#ifdef ENABLE_NCCL
MultiGpuAggregator::Config config;
config.gpu_ids = {0, 1, 2, 3};
config.num_regions = 16;
config.num_devices_per_region = 64;

MultiGpuAggregator aggregator(config);
aggregator.ingest_batch(events.data(), events.size());
aggregator.synchronize();  // Sync across GPUs
#endif
```

## Examples

### Ingestion Demo
Demonstrates high-throughput event ingestion:
```bash
./build/examples/ingest_demo
```

### Query Demo
Shows ad-hoc querying capabilities:
```bash
./build/examples/query_demo
```

### Multi-GPU Demo
Demonstrates multi-GPU aggregation (requires NCCL):
```bash
./build/examples/multi_gpu_demo
```

## Performance Tuning

### Batch Size
- Larger batches amortize PCIe latency but increase memory footprint
- Recommended: 10k-50k events per batch
- Default: 16,384 events

### Memory Footprint
Memory scales as: `regions × devices × bins × sizeof(AggCell)`

Example: 16 regions × 64 devices × 10,080 bins × 24 bytes ≈ **2.5 GB**

To reduce memory:
- Reduce `bins_per_day` (e.g., 5-minute bins: `24 * 12 = 288`)
- Reduce `days_window` (e.g., 3 days instead of 7)
- Use sparse storage for low-cardinality regions/devices

### Thread Block Size
- Default: 256 threads per block
- Tune based on GPU architecture and event distribution
- Larger blocks reduce atomic contention but may reduce occupancy

### Double-Buffering
- Enable with `config.use_double_buffering = true`
- Requires `config.num_streams >= 2`
- Overlaps CPU work with GPU processing

## Dashboard Integration

The system provides a REST API interface (see `include/rest_api.h`) for dashboard integration. Example endpoints:

- `GET /query?region=3&device=5` - Query specific region/device
- `GET /query?region=3&device=5&start_ts=1234567890&end_ts=1234567890` - Query time range
- `GET /stats` - Get ingestion statistics

See `examples/dashboard.html` for a complete dashboard example.

## Production Considerations

### Persistence
- Periodically flush snapshots to disk or distributed store
- Consider using HDF5, Parquet, or time-series databases

### Fault Tolerance
- Add checkpointing for recovery
- Replicate aggregates across nodes for redundancy

### Monitoring
- Track ingestion rate, GPU utilization, memory usage
- Alert on backpressure or dropped events

### Security
- Validate timestamps (handle clock skew)
- Sanitize query parameters
- Rate limit queries

### Scaling
- For cluster-level aggregation:
  - Run per-node aggregators
  - Periodically reduce via centralized aggregator or distributed reduce
  - Use message queues (Kafka, Pulsar) for event distribution

## License

[Specify your license here]

## Contributing

[Contributing guidelines]

## Documentation

- **[BUILD_GUIDE.md](BUILD_GUIDE.md)**: Step-by-step guide showing how the system was built, including architecture decisions, optimizations, and implementation details
- **[ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)**: Enhanced architecture with tiered storage, telemetry, and production features
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Core architecture overview
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Project summary and key features
- **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)**: Benchmarking and profiling guide
- **[QUICKSTART.md](QUICKSTART.md)**: Quick start guide

## References

- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- Modern C++ Guidelines: https://isocpp.github.io/CppCoreGuidelines/

