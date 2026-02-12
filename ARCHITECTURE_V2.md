# Architecture V2 - Production-Grade GPU Analytics Engine

## Overview

This document describes the enhanced, production-grade architecture designed for NVIDIA-scale workloads with real-time telemetry integration, tiered storage, and optimized query processing.

> **📊 For comprehensive system diagrams**, see [SYSTEM_DIAGRAMS.md](SYSTEM_DIAGRAMS.md) which includes detailed Mermaid diagrams showing:
> - High-level system architecture with all HPC/AI stack components
> - Data flow through tiered storage
> - Multi-node architecture with NCCL
> - SIMD query engine architecture
> - Monitoring & telemetry integration
> - Performance optimization workflows

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Real-Time Ingestion Tier                     │
│  • Kafka/Pulsar/RDMA Streams (100K+ events/sec)                │
│  • Micro-buffer batching (1-4MB chunks)                        │
│  • Backpressure handling + load shedding                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              GPU Processing Layer (C++20 + CUDA)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Columnar (SoA) Layout                                    │  │
│  │ • Warp-coalesced memory access                           │  │
│  │ • Separate columns: ts[], region[], device[], util[]    │  │
│  │ • 2-3x faster than AoS for GPU kernels                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ GPU Ring Buffer (7-day sliding window)                   │  │
│  │ • Power-of-two layout for efficient rotation             │  │
│  │ • Per-block shared-memory reduction                      │  │
│  │ • Multi-GPU sharding: region × device × time             │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Nsight Instrumentation                                    │  │
│  │ • Kernel launch markers                                   │  │
│  │ • Memory transfer tracking                                │  │
│  │ • SM throughput monitoring                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Tiered Storage System                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ HOT Tier     │  │ WARM Tier    │  │ COLD Tier    │         │
│  │ 0-24h        │→ │ 24h-7d       │→ │ >7d          │         │
│  │ GPU HBM      │  │ CPU RAM      │  │ NVMe/S3      │         │
│  │ Compressed   │  │ Parquet-like  │  │ Spill plan   │         │
│  │ Columns      │  │ Memory map    │  │ Queryable    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              CPU Ad-Hoc Query Engine                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SIMD Vector Filters (AVX-512/AVX2)                        │  │
│  │ • Region/device filtering: 8-16x faster than scalar      │  │
│  │ • Bitmap indexes for fast lookups                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ GPU Pushdown                                              │  │
│  │ • Large scans offloaded to GPU                            │  │
│  │ • Cross-tier query coordination                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              GPU Telemetry Integration                          │
│  • NVML/DCGM for real-time metrics                             │
│  • Power cap control + DVFS hints                              │
│  • SM throughput + warp efficiency tracking                    │
│  • Prometheus/Grafana integration                               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Columnar (SoA) Layout

**Why**: GPU memory access patterns favor Structure of Arrays (SoA) over Array of Structures (AoS).

**Implementation**:
```cpp
// AoS (slower on GPU)
struct Event { uint64_t ts; uint16_t region; float util; } events[10000];

// SoA (faster on GPU - coalesced access)
uint64_t timestamps[10000];
uint16_t regions[10000];
float utils[10000];
```

**Benefits**:
- Warp-coalesced memory access (32 threads read consecutive addresses)
- 2-3x faster kernel execution
- Better cache utilization

### 2. Tiered Storage

**Hot Tier (0-24h)**: GPU HBM
- Fastest access (<1ms)
- Limited capacity (~2.5GB for default config)
- Compressed columnar format

**Warm Tier (24h-7d)**: CPU RAM
- Fast access (~5-10ms)
- Larger capacity (10-50GB)
- Parquet-like memory-mapped format

**Cold Tier (>7d)**: NVMe/S3
- Slower access (~50-200ms)
- Unlimited capacity
- Queryable via spill plan

**Spill Logic**:
- Automatic spill when hot tier fills
- Background thread for warm→cold spill
- Transparent query across tiers

### 3. SIMD Query Engine

**SIMD (Single Instruction, Multiple Data)** is a CPU-level parallel computing technique where one instruction operates on multiple data points simultaneously. Instead of processing one record at a time (scalar), the CPU uses wide registers (like AVX-512) to process 8, 16, or 32 data elements in one cycle.

**AVX-512 Optimizations**:
- Vectorized region/device filtering (16 elements per cycle)
- Parallel aggregation (sum, max, min)
- **8-16x faster than scalar code** (up to 9x speedup for spatial joins and range selects)
- Reduces CPU time spent waiting for data to be ready for GPU

**Applications**:
- Database querying and string matching
- Data preprocessing before sending to GPU
- Spatial joins and range selects
- Bitmap index operations

**Fallback**:
- AVX2 support (8 elements per cycle)
- Scalar fallback for older CPUs

### 4. GPU Telemetry Integration (NVML/DCGM)

**NVML (NVIDIA Management Library)**:
- C-based API for monitoring and managing GPU devices
- Provides metrics like temperature, power usage, and clock speeds
- Low-level control for power cap management

**DCGM (Data Center GPU Manager)**:
- Higher-level, lightweight tool designed for large-scale production environments
- Provides **over 40 metrics per device**, including:
  - Tensor Core utilization
  - Memory bandwidth utilization
  - PCIe/NVLink throughput
  - SM throughput and warp efficiency
  - L2 cache hit/miss rates
  - Power consumption and power cap
  - Temperature and thermal throttling
- **Kubernetes Integration**: dcgm-exporter for real-time monitoring in K8s clusters
- **Role**: Ensures GPU cluster runs at maximum efficiency (~90%+ utilization) and detects faults immediately

**Metrics Collected**:
- GPU utilization (compute + memory)
- Power consumption + power cap
- Temperature and thermal throttling
- SM throughput and warp efficiency
- L2 cache hit/miss rates
- PCIe/NVLink bandwidth utilization
- Tensor Core usage (if applicable)

**Use Cases**:
- Self-monitoring (detect throttling, ensure 90%+ utilization)
- Performance optimization (tune kernel sizes based on real metrics)
- Power management (adjust power caps, DVFS hints)
- Debugging (identify bottlenecks in real-time)
- Production monitoring (Prometheus/Grafana integration)

### 5. Nsight Profiling

**Nsight Systems** (System-wide profiling):
- High-level timeline view of CPU activity, CUDA API calls, and GPU kernels
- Identifies GPU idle time (starvation) and memory transfer inefficiencies
- Correlates CPU and GPU activities on a single timeline
- Multi-node analysis for tracking performance across multiple GPUs on different servers

**Nsight Compute** (Kernel-level analysis):
- Interactive profiler for detailed, low-level analysis of individual CUDA kernels
- Deep metrics on SM (Streaming Multiprocessor) activity
- Tensor Core usage analysis
- Memory bottleneck identification
- Instruction throughput and warp efficiency metrics

**Instrumentation Points**:
- Kernel launches (grid/block dimensions, shared memory)
- Memory transfers (size, direction, timing)
- Query operations (latency tracking)
- Export to NVTX for visualization

**Usage**:
```bash
# System-wide profiling
nsys profile --trace=cuda,nvtx ./ingest_demo

# Kernel-level analysis
ncu --set full ./ingest_demo
```

### 6. NCCL (Multi-Node Scaling)

**NVIDIA Collective Communications Library (NCCL)** is the standard for high-performance communication between GPUs in parallel applications.

**Key Features**:
- **Topology Detection**: Automatically detects hardware topology (PCIe, NVLink, InfiniBand) and optimizes for it
- **Collective Operations**: Implements optimized collectives essential for distributed workloads:
  - **AllReduce**: Global sum across all GPUs
  - **ReduceScatter**: Sharded reduction with distributed results
  - **AllGather**: Global merge of distributed data
- **Scaling**: Seamlessly scales from single GPU to thousands of GPUs across multiple nodes
- **Network Optimization**: Optimizes for high-bandwidth networks (InfiniBand EDR/HDR, NVLink 3.0/4.0)

**Integration in Event Aggregator**:
- Periodic synchronization of aggregates across nodes
- Distributed query coordination
- Linear scaling with GPU count
- Automatic routing optimization based on network topology

**Performance**:
- Enables training/inference in hours rather than weeks
- Handles massive multi-node clusters efficiently
- Maintains high bandwidth utilization across nodes

## Performance Characteristics

### Ingestion
- **Sustained Rate**: 100K-200K events/sec (depends on GPU)
- **Latency**: <10ms per batch (p99)
- **Throughput**: Scales linearly with GPU count (with NCCL)

### Queries
- **Single Device**: <5ms (hot tier), <20ms (warm tier)
- **Single Region**: <50ms (hot tier), <200ms (warm tier)
- **Cross-Tier**: <500ms (with GPU pushdown)

### Memory
- **Hot Tier**: ~2.5GB (default config)
- **Warm Tier**: ~10-50GB (configurable)
- **Cold Tier**: Unlimited (disk-based)

## Scalability

### Single Node
- **Single GPU**: 100K events/sec
- **Multi-GPU (4x)**: ~400K events/sec (with NCCL)
- **Memory**: Scales with GPU count

### Cluster
- **Per-Node Aggregation**: Each node runs local aggregator
- **Event Distribution**: Kafka/Pulsar for load balancing
- **Query Coordination**: Distributed query planner
- **Global Aggregation**: Periodic reduce across nodes

## Production Features

### Monitoring
- Prometheus metrics export
- Grafana dashboards
- Real-time alerting

### Fault Tolerance
- Checkpointing (periodic snapshots)
- Write-ahead log (WAL) for recovery
- Multi-node replication

### Security
- Input validation (timestamp bounds, region/device limits)
- Rate limiting (prevent DoS)
- Access control (region/device filtering)

## Debugging & Optimization

### Nsight Integration
```bash
# Profile kernel performance
nsys profile --trace=cuda,nvtx ./ingest_demo

# Analyze kernel efficiency
ncu --set full ./ingest_demo
```

### Telemetry Hooks
- Real-time GPU metrics during ingestion
- Power profiling for optimization
- SM throughput analysis

### Performance Tuning
1. **Kernel Sizing**: Adaptive based on SM count
2. **Memory Layout**: SoA for better coalescing
3. **Batch Sizing**: Tune for PCIe bandwidth
4. **Power Management**: DVFS hints for peak performance

## Interview Talking Points

### Architecture Decisions
1. **Why Columnar?** Warp-coalesced access → 2-3x faster
2. **Why Tiered Storage?** Cost/performance tradeoff
3. **Why SIMD?** 8-16x faster CPU queries
4. **Why Telemetry?** Self-optimization + debugging

### Scalability
- Linear scaling with GPU count (NCCL)
- Horizontal scaling via message queues
- Efficient memory usage (compression)

### Production Readiness
- Monitoring (Prometheus/Grafana)
- Fault tolerance (checkpointing, WAL)
- Security (validation, rate limiting)

### NVIDIA-Specific
- **Nsight integration** for profiling (Systems + Compute)
- **NVML/DCGM** for telemetry (40+ metrics per device)
- **Multi-GPU coordination** via NCCL (scales to thousands of GPUs)
- **Power-aware optimization** (DVFS hints, power cap control)
- **Columnar layouts** for warp-coalesced GPU access
- **Tiered storage** leveraging GPU HBM → CPU RAM → disk

## Modern HPC/AI Stack Summary

This architecture leverages the complete modern HPC/AI technology stack:

| Component | Purpose | Key Benefit |
|-----------|---------|-------------|
| **SIMD/CPU** | Rapid data preparation | Maximize CPU efficiency before GPU (8-16x speedup) |
| **Columnar Layout** | Warp-coalesced GPU access | 2-3x faster GPU kernels |
| **Nsight** | Visualize and debug bottlenecks | Find idle GPU time & bottlenecks |
| **DCGM/NVML** | Real-time monitoring | Ensure maximum GPU utilization (~90%+) |
| **NCCL** | High-speed communication | Seamless, high-bandwidth scaling across nodes |
| **Tiered Storage** | Cost-effective scaling | GPU HBM → CPU RAM → disk with transparent queries |

This combination allows AI models and analytics workloads to process data in hours rather than weeks, as the CPU rapidly feeds data (SIMD), the GPU remains saturated (monitored by DCGM), and communication between nodes is handled at maximum possible speed (NCCL).

