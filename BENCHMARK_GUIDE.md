# Benchmark Guide

## Running Benchmarks

### Basic Benchmark

```bash
./build/benchmark_suite 1000000
```

This runs:
- Ingestion benchmark with 1M events
- Query benchmark (100 random queries)
- Generates `benchmark_report.md`

### With Power Profiling

```bash
./build/benchmark_suite 1000000 --power
```

Additionally:
- Collects power/telemetry data for 60 seconds
- Generates `power_profile.csv`

### Custom Configuration

Edit `benchmark_suite.cpp` to adjust:
- Number of regions/devices
- Batch sizes
- Query patterns

## Nsight Profiling

### Nsight Systems (System-wide)

```bash
nsys profile --trace=cuda,nvtx ./build/benchmark_suite 1000000
```

Generates `report.nsys-rep` - open with:
```bash
nsys-ui report.nsys-rep
```

**What to look for**:
- Kernel execution timeline
- Memory transfer patterns
- PCIe bandwidth utilization
- GPU utilization gaps

### Nsight Compute (Kernel-level)

```bash
ncu --set full ./build/benchmark_suite 1000000
```

**What to look for**:
- SM throughput (IPC)
- Warp efficiency
- Memory throughput
- Occupancy

### NVTX Markers

The code includes NVTX markers for visualization:
- Kernel launches
- Memory transfers
- Query operations

View in Nsight Systems timeline.

## Interpreting Results

### Ingestion Rate

**Target**: 100K+ events/sec

**If lower**:
- Check GPU utilization (should be >80%)
- Check PCIe bandwidth (use `nvidia-smi dmon`)
- Increase batch size
- Enable double-buffering

### Query Latency

**Targets**:
- P50: <5ms (single device)
- P95: <20ms
- P99: <50ms

**If higher**:
- Check if using SIMD (AVX-512/AVX2)
- Reduce query time range
- Use GPU pushdown for large scans

### Power Consumption

**Monitor**:
- Average power during ingestion
- Power spikes (throttling)
- Power cap effectiveness

**Optimize**:
- Adjust power caps
- Tune kernel sizes
- Use DVFS hints

## Benchmark Report Format

The generated `benchmark_report.md` includes:

1. **Ingestion Performance**
   - Events processed
   - Time taken
   - Rate (events/sec)
   - GPU utilization
   - Power consumption

2. **Query Performance**
   - P50/P95/P99 latencies
   - Query patterns

3. **Profiling Instructions**
   - Nsight commands
   - Power profile location

## Power Profile CSV

Columns:
- `timestamp`: Milliseconds since start
- `power_watts`: Current power consumption
- `util_gpu`: GPU utilization %
- `util_mem`: Memory utilization %
- `temp_celsius`: Temperature

Use for:
- Identifying power spikes
- Correlating with performance
- Optimizing power caps

## Example Workflow

1. **Baseline**:
   ```bash
   ./build/benchmark_suite 1000000
   ```

2. **Profile with Nsight**:
   ```bash
   nsys profile --trace=cuda,nvtx ./build/benchmark_suite 1000000
   nsys-ui report.nsys-rep
   ```

3. **Power Profile**:
   ```bash
   ./build/benchmark_suite 1000000 --power
   # Analyze power_profile.csv
   ```

4. **Optimize** (based on findings):
   - Adjust batch sizes
   - Tune kernel parameters
   - Modify memory layout

5. **Re-run** to verify improvements

## Troubleshooting

### CUDA errors
- Check GPU availability: `nvidia-smi`
- Verify CUDA version compatibility

### Low performance
- Check GPU utilization: `nvidia-smi -l 1`
- Profile with Nsight to find bottlenecks
- Verify SIMD is enabled (check CPU flags)

### Telemetry not available
- Install NVML: `libnvidia-ml-dev` (Linux)
- Or use DCGM: `datacenter-gpu-manager`
- Falls back gracefully if unavailable

