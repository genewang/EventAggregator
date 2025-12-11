# Architecture Overview

## System Design

The Event Aggregator is designed as a high-throughput, low-latency event processing system that leverages GPU acceleration for real-time aggregation while maintaining fast CPU-side query capabilities.

### Core Components

1. **GpuAggregator**: Main aggregation engine
2. **MultiGpuAggregator**: Multi-GPU coordination (optional, requires NCCL)
3. **Query API**: CPU-side query interface
4. **REST API**: HTTP interface for dashboard integration

## Data Flow

```
Events → Pinned Host Buffer → Async Copy → GPU Kernel → Aggregation Array
                                                              ↓
                                                         Query API
                                                              ↓
                                                         Dashboard
```

### Ingestion Path

1. **Event Reception**: Events arrive in batches (typically 10k-50k events)
2. **Host Buffer**: Events stored in pinned host memory (fast DMA)
3. **Async Transfer**: `cudaMemcpyAsync` transfers to device
4. **GPU Processing**: CUDA kernel processes events in parallel
5. **Atomic Updates**: Aggregates updated atomically (with contention reduction)

### Query Path

1. **Query Request**: CPU-side query with region/device/time filters
2. **Device Copy**: Small aggregated arrays copied back to host
3. **Result Assembly**: Results formatted and returned
4. **Dashboard**: JSON response sent to dashboard client

## Memory Layout

### Power-of-Two Circular Buffer

The aggregation array uses a power-of-two size to enable efficient bit-masking:

```
Actual bins needed: 10,080 (7 days × 24 hours × 60 minutes)
Power-of-two size:  16,384 (2^14)
Mask:                0x3FFF (16,383)
```

**Benefits:**
- Fast modulo via bitwise AND: `bin_idx & mask`
- Efficient rotation: `(bin + offset) & mask`
- Cache-friendly alignment

### Sliding Window Rotation

As time progresses, old bins are evicted:

```
Window: [oldest] ... [newest]
         ↓
Rotate: [newest] ... [oldest] (circular)
```

The rotation kernel clears bins that fall outside the 7-day window.

## Performance Optimizations

### 1. Per-Block Shared Memory Reduction

**Problem**: High atomic contention when many events map to the same bin.

**Solution**: Accumulate within thread blocks before global atomic updates.

```
Thread Block Processing:
  ┌─────────────────┐
  │ Shared Memory   │  ← Per-block aggregates
  │ Accumulation    │
  └─────────────────┘
         ↓
  ┌─────────────────┐
  │ Global Memory   │  ← One atomic update per block
  │ Atomic Update   │     (not per event)
  └─────────────────┘
```

**Impact**: Reduces atomic operations by ~256x (assuming 256 threads/block).

### 2. Double-Buffering

**Problem**: CPU must wait for GPU to finish before sending next batch.

**Solution**: Multiple pinned buffers with separate CUDA streams.

```
Stream 0: [Buffer A] → GPU → Process
Stream 1: [Buffer B] → GPU → Process
         ↑                    ↑
    CPU fills            GPU processes
    (concurrent)
```

**Impact**: Overlaps CPU event generation with GPU processing.

### 3. Pinned Memory

**Problem**: Pageable host memory requires staging through pinned memory.

**Solution**: Pre-allocate pinned host buffers via `cudaHostAlloc`.

**Impact**: Direct DMA transfers, ~2-3x faster than pageable memory.

## Scalability Considerations

### Single GPU Limits

- **Memory**: ~2.5 GB for default config (16 regions × 64 devices × 10k bins)
- **Throughput**: ~100k-200k events/sec (depends on GPU and event distribution)
- **Query Latency**: <10ms for single device query

### Multi-GPU Scaling

With NCCL:
- **Horizontal Scaling**: Linear scaling with number of GPUs
- **Synchronization**: Periodic all-reduce to merge aggregates
- **Query**: Query from any GPU (or aggregate results)

### Cluster Scaling

For cluster-level aggregation:
1. **Per-Node Aggregation**: Each node runs local aggregator
2. **Event Distribution**: Use message queue (Kafka, Pulsar)
3. **Periodic Reduction**: Centralized or distributed reduce
4. **Query Routing**: Query coordinator aggregates from nodes

## Query Performance

### Query Types

1. **Single Device**: Fastest (~1-5ms)
   - Copies single contiguous block: `bins × sizeof(AggCell)`
   - Example: 10,080 bins × 24 bytes ≈ 240 KB

2. **Single Region**: Moderate (~5-20ms)
   - Copies all devices in region
   - Example: 64 devices × 240 KB ≈ 15 MB

3. **All Regions**: Slowest (~50-200ms)
   - Copies entire aggregation array
   - Example: 16 regions × 15 MB ≈ 240 MB

### Optimization Strategies

- **Caching**: Cache frequently queried regions/devices
- **Pre-aggregation**: Pre-compute common query patterns
- **Time-range limits**: Restrict query time windows
- **Sampling**: Return sampled bins for large time ranges

## Fault Tolerance

### Current Limitations

- No persistence (aggregates lost on restart)
- No replication (single point of failure)
- No checkpointing

### Production Enhancements

1. **Persistence**:
   - Periodic snapshots to disk (HDF5, Parquet)
   - Write-ahead log for recent events

2. **Replication**:
   - Multi-node replication via distributed reduce
   - Quorum-based consistency

3. **Recovery**:
   - Restore from latest snapshot
   - Replay events from WAL

## Monitoring & Observability

### Key Metrics

- **Ingestion Rate**: Events/sec (target: 100k+)
- **GPU Utilization**: Kernel occupancy, memory bandwidth
- **Query Latency**: P50, P95, P99
- **Memory Usage**: GPU memory, host memory
- **Error Rate**: Dropped events, query failures

### Instrumentation Points

- Event ingestion entry/exit
- GPU kernel launch/completion
- Query request/response
- Window rotation events

## Security Considerations

### Input Validation

- **Timestamp Validation**: Reject events with future timestamps or extreme skew
- **Bounds Checking**: Validate region/device IDs
- **Rate Limiting**: Prevent query DoS

### Data Privacy

- **Node Name Hashing**: Node names hashed on GPU (not stored in plaintext)
- **Access Control**: Restrict queries by region/device
- **Audit Logging**: Log all queries for compliance

## Future Enhancements

1. **Sparse Storage**: Hash-based storage for low-cardinality regions
2. **Compression**: Compress old bins (e.g., delta encoding)
3. **Streaming Queries**: WebSocket for real-time updates
4. **ML Integration**: Anomaly detection, forecasting
5. **GraphQL API**: Flexible query interface

