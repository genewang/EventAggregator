# Quick Start Guide

## Prerequisites

- CUDA Toolkit 11.0+ installed
- NVIDIA GPU with compute capability 7.5+
- CMake 3.18+
- C++17 compiler

## Build

```bash
# Basic build
./build.sh

# With specific CUDA architecture
./build.sh --cuda-arch "75;80;86"

# With NCCL support (multi-GPU)
./build.sh --nccl
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
cmake --build . -j$(nproc)
```

## Run Examples

### 1. Ingestion Demo

Simulates 100k events/sec ingestion:

```bash
./build/examples/ingest_demo
```

Expected output:
```
Event Aggregator - Ingestion Demo
==================================

Ingesting events at target rate: 100000 events/sec
Batch size: 16384, Batches/sec: 7

Batches: 10, Events: 163840, Rate: 98523 events/sec
...
Ingestion complete!
Total events: 1000000
Average rate: 98523.45 events/sec
```

### 2. Query Demo

Demonstrates ad-hoc querying:

```bash
./build/examples/query_demo
```

Expected output:
```
Event Aggregator - Query Demo
=============================

Ingesting sample events...
Sample events ingested.

Query 1: Region 3, Device 5 (last 7 days)

Query Result:
=============
Region: 3
Device: 5
Total events: 1000
Average GPU util: 75.25%
Max power (actual): 299.5 W
...
```

### 3. Multi-GPU Demo (requires NCCL)

```bash
./build/examples/multi_gpu_demo
```

## Integration Example

```cpp
#include "event_aggregator.h"
using namespace event_aggregator;

// 1. Create aggregator
GpuAggregator::Config config;
config.num_regions = 16;
config.num_devices_per_region = 64;
config.gpu_id = 0;

GpuAggregator agg(config);

// 2. Ingest events
std::vector<Event> events(10000);
// ... populate events ...
agg.ingest_batch(events.data(), events.size());

// 3. Query
QueryParams params;
params.region = 3;
params.device = 5;

QueryResult result = agg.query(params);
std::cout << "Total: " << result.total_count << "\n";
std::cout << "Avg util: " << result.avg_gpu_util << "%\n";
```

## Dashboard

1. Start your REST API server (integrate `RestApi` class)
2. Open `examples/dashboard.html` in a browser
3. Adjust API endpoint in dashboard JavaScript if needed

## Performance Tuning

### For Higher Throughput

1. **Increase batch size**:
   ```cpp
   config.max_events_per_batch = 32768;  // or higher
   ```

2. **Enable double-buffering**:
   ```cpp
   config.use_double_buffering = true;
   config.num_streams = 4;  // more streams = more overlap
   ```

3. **Tune thread block size**:
   ```cpp
   config.gpu_threads = 512;  // depends on GPU architecture
   ```

### For Lower Memory

1. **Reduce time resolution**:
   ```cpp
   config.bins_per_day = 24 * 12;  // 5-minute bins instead of 1-minute
   ```

2. **Reduce window size**:
   ```cpp
   config.days_window = 3;  // 3 days instead of 7
   ```

3. **Reduce dimensions**:
   ```cpp
   config.num_regions = 8;
   config.num_devices_per_region = 32;
   ```

## Troubleshooting

### CUDA errors

- Check GPU is available: `nvidia-smi`
- Verify CUDA version: `nvcc --version`
- Check compute capability matches your GPU

### Build errors

- Ensure CMake finds CUDA: `cmake ..` should show "Found CUDA"
- Check CUDA architecture matches your GPU
- Verify C++17 support: `g++ --version` or `clang++ --version`

### Performance issues

- Monitor GPU utilization: `nvidia-smi -l 1`
- Check for PCIe bottlenecks (use `nvidia-smi dmon`)
- Verify pinned memory is used (check `cudaHostAlloc` calls)

## Next Steps

- Read [README.md](README.md) for detailed documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check examples/ for more usage patterns
- Integrate REST API for dashboard connectivity

