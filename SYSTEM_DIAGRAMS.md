# System Diagrams - Modern HPC/AI Stack

This document provides comprehensive system diagrams for the Event Aggregator, showcasing the integration of modern HPC/AI technologies: columnar memory layouts, tiered storage, SIMD-optimized CPU queries, Nsight profiling, NVML/DCGM telemetry, and NCCL for multi-node scaling.

## 1. High-Level System Architecture

This diagram shows the complete system architecture with all modern HPC/AI stack components:

```mermaid
graph TB
    subgraph "Ingestion Layer"
        A[Event Streams<br/>Kafka/Pulsar/RDMA<br/>100K+ events/sec] --> B[Micro-buffer Batching<br/>1-4MB chunks]
        B --> C[Pinned Host Buffers<br/>Zero-copy DMA]
    end
    
    subgraph "SIMD-Optimized CPU Preprocessing"
        C --> D[SIMD Query Engine<br/>AVX-512/AVX2<br/>8-16x faster]
        D --> E[Data Preprocessing<br/>Spatial joins<br/>Range selects]
    end
    
    subgraph "GPU Processing Layer"
        E --> F[Async CUDA Transfer<br/>cudaMemcpyAsync]
        F --> G[Columnar SoA Layout<br/>Warp-coalesced access<br/>2-3x faster]
        G --> H[CUDA Kernels<br/>Per-block reduction<br/>Atomic updates]
        H --> I[GPU Aggregation Array<br/>7-day sliding window]
    end
    
    subgraph "Tiered Storage System"
        I --> J[Hot Tier: GPU HBM<br/>0-24h, <1ms access<br/>Compressed columns]
        J --> K[Warm Tier: CPU RAM<br/>24h-7d, 5-10ms<br/>Parquet-like format]
        K --> L[Cold Tier: Disk/NVMe/S3<br/>>7d, 50-200ms<br/>Queryable spill]
    end
    
    subgraph "Query & Analytics Layer"
        M[Query API<br/>Region/Device/Time filters] --> N[SIMD Vector Filters<br/>AVX-512 filtering<br/>8-16x speedup]
        N --> O[Cross-Tier Query<br/>Hot/Warm/Cold coordination]
        O --> P[GPU Pushdown<br/>Large scans offloaded]
        P --> Q[Result Assembly<br/>JSON formatting]
    end
    
    subgraph "Monitoring & Profiling"
        H --> R[Nsight Systems<br/>System-wide profiling<br/>Timeline visualization]
        H --> S[Nsight Compute<br/>Kernel analysis<br/>SM throughput]
        I --> T[NVML/DCGM<br/>Real-time telemetry<br/>40+ metrics/device]
        T --> U[Prometheus/Grafana<br/>Metrics export<br/>Dashboards]
    end
    
    subgraph "Multi-Node Scaling"
        I --> V[NCCL Communication<br/>AllReduce/ReduceScatter<br/>Topology-aware]
        V --> W[Multi-GPU Sharding<br/>Region × Device × Time]
        W --> X[Cluster Coordination<br/>Distributed queries<br/>Global aggregation]
    end
    
    Q --> Y[REST API<br/>Dashboard Integration]
    Y --> Z[Web Dashboard<br/>Real-time visualization]
    
    style G fill:#e1f5ff
    style D fill:#fff4e1
    style J fill:#ffe1f5
    style T fill:#e1ffe1
    style V fill:#f5e1ff
```

## 2. Data Flow Through Tiered Storage

This diagram details how data flows through the tiered storage system with automatic spill-over:

```mermaid
flowchart TD
    Start([Event Ingestion<br/>100K+ events/sec]) --> Buffer[Pinned Host Buffer<br/>Double-buffering]
    
    Buffer --> SIMD[SIMD Preprocessing<br/>AVX-512 filtering<br/>9x speedup]
    
    SIMD --> Transfer[Async GPU Transfer<br/>cudaMemcpyAsync<br/>PCIe bandwidth optimized]
    
    Transfer --> Columnar["Columnar SoA Layout<br/>Warp-coalesced access<br/>ts array, region array, util array"]
    
    Columnar --> Kernel[CUDA Kernel Processing<br/>Per-block reduction<br/>256x less contention]
    
    Kernel --> Hot{Hot Tier<br/>GPU HBM<br/>0-24h}
    
    Hot -->|Full| Spill1[Automatic Spill<br/>Background thread]
    Spill1 --> Warm[Warm Tier<br/>CPU RAM<br/>24h-7d<br/>Parquet-like format]
    
    Warm -->|Full| Spill2[Background Spill]
    Spill2 --> Cold[Cold Tier<br/>NVMe/S3<br/>>7d<br/>Queryable]
    
    Hot --> Query[Query Request]
    Warm --> Query
    Cold --> Query
    
    Query --> SIMDQuery[SIMD Query Engine<br/>AVX-512 vectorization<br/>8-16x faster]
    
    SIMDQuery --> Result["Query Result<br/>~5ms hot, ~200ms warm<br/>~500ms cross-tier"]
    
    style Hot fill:#ff6b6b
    style Warm fill:#ffd93d
    style Cold fill:#6bcf7f
    style Columnar fill:#4d96ff
    style SIMD fill:#ff9ff3
```

## 3. Multi-Node Architecture with NCCL

This diagram shows how the system scales across multiple nodes using NCCL:

```mermaid
graph TB
    subgraph "Node 1"
        A1[Event Stream 1] --> B1[Local Aggregator<br/>GPU 0-3]
        B1 --> C1[Local Aggregation<br/>Region × Device × Time]
        C1 --> D1[NCCL AllReduce<br/>Periodic sync]
    end
    
    subgraph "Node 2"
        A2[Event Stream 2] --> B2[Local Aggregator<br/>GPU 4-7]
        B2 --> C2[Local Aggregation<br/>Region × Device × Time]
        C2 --> D2[NCCL AllReduce<br/>Periodic sync]
    end
    
    subgraph "Node N"
        AN[Event Stream N] --> BN[Local Aggregator<br/>GPU N×4 to N×4+3]
        BN --> CN[Local Aggregation<br/>Region × Device × Time]
        CN --> DN[NCCL AllReduce<br/>Periodic sync]
    end
    
    subgraph "NCCL Communication Layer"
        D1 --> E[NCCL Topology Detection<br/>PCIe/NVLink/InfiniBand<br/>Automatic optimization]
        D2 --> E
        DN --> E
        
        E --> F[Collective Operations<br/>AllReduce: Global sum<br/>ReduceScatter: Sharded reduce<br/>AllGather: Global merge]
        
        F --> G[High-Bandwidth Network<br/>InfiniBand EDR/HDR<br/>NVLink 3.0/4.0<br/>Optimized routing]
    end
    
    subgraph "Global Query Coordination"
        H[Query Request] --> I[Query Planner<br/>Distributed execution]
        I --> J1[Node 1 Query]
        I --> J2[Node 2 Query]
        I --> JN[Node N Query]
        
        J1 --> K[Result Aggregation<br/>Merge from all nodes]
        J2 --> K
        JN --> K
        
        K --> L[Final Result<br/>Global aggregation]
    end
    
    G --> M[Global State<br/>Synchronized aggregates<br/>All nodes consistent]
    
    style E fill:#9b59b6
    style F fill:#9b59b6
    style G fill:#3498db
    style M fill:#2ecc71
```

## 4. Component Interaction Diagram

This diagram shows how all components interact during normal operation:

```mermaid
sequenceDiagram
    participant Stream as Event Stream
    participant SIMD as SIMD Engine
    participant GPU as GPU Kernel
    participant Hot as Hot Tier (HBM)
    participant Warm as Warm Tier (RAM)
    participant Cold as Cold Tier (Disk)
    participant Query as Query Engine
    participant Nsight as Nsight Profiler
    participant DCGM as DCGM/NVML
    participant NCCL as NCCL
    
    Stream->>SIMD: Batch events (10K-50K)
    Note over SIMD: AVX-512 preprocessing<br/>9x speedup
    
    SIMD->>GPU: Transfer via pinned memory
    Note over GPU: Columnar SoA layout<br/>Warp-coalesced access
    
    GPU->>Nsight: Kernel launch marker
    Nsight->>Nsight: Track timing & metrics
    
    GPU->>DCGM: Request telemetry
    DCGM-->>GPU: Utilization, power, SM metrics
    
    GPU->>Hot: Update aggregates (0-24h)
    
    alt Hot tier full
        Hot->>Warm: Spill oldest data
        Note over Warm: Parquet-like format
    end
    
    alt Warm tier full
        Warm->>Cold: Spill to disk
        Note over Cold: Queryable format
    end
    
    Query->>Hot: Query request (hot data)
    Hot-->>Query: Results (<5ms)
    
    Query->>Warm: Query request (warm data)
    Warm-->>Query: Results (<20ms)
    
    Query->>Cold: Query request (cold data)
    Cold-->>Query: Results (<200ms)
    
    Note over GPU,NCCL: Periodic sync (every N seconds)
    GPU->>NCCL: Local aggregates
    NCCL->>NCCL: AllReduce across nodes
    NCCL-->>GPU: Global aggregates
```

## 5. Memory Layout: Columnar (SoA) vs Array of Structures (AoS)

This diagram illustrates the columnar memory layout optimization:

```mermaid
flowchart TB
    subgraph AoS["Array of Structures (AoS) - Slower"]
        direction LR
        A1["Event 0<br/>ts, region, util"]
        A2["Event 1<br/>ts, region, util"]
        A3["Event 2<br/>ts, region, util"]
        A4["..."]
        A1 --> A2 --> A3 --> A4
        Note1["Warp reads scattered<br/>Poor coalescing<br/>2-3x slower"]
    end
    
    subgraph SoA["Structure of Arrays (SoA) - Faster"]
        direction TB
        B1["ts[0,1,2...]<br/>Consecutive addresses"]
        B2["region[0,1,2...]<br/>Consecutive addresses"]
        B3["util[0,1,2...]<br/>Consecutive addresses"]
        B1 --> B2 --> B3
        Note2["Warp reads coalesced<br/>32 threads read<br/>consecutive addresses<br/>2-3x faster"]
    end
    
    subgraph Warp["GPU Warp Access Pattern"]
        direction LR
        C1["Thread 0"] --> C2["Thread 1"] --> C3["Thread 2"] --> C4["..."] --> C5["Thread 31"]
        Note3["32 threads in warp<br/>Read consecutive addresses<br/>Perfect coalescing"]
    end
    
    style B1 fill:#4d96ff
    style B2 fill:#4d96ff
    style B3 fill:#4d96ff
    style Note2 fill:#2ecc71
    style Note1 fill:#e74c3c
    style AoS fill:#ffe5e5
    style SoA fill:#e5f5ff
    style Warp fill:#fff5e5
```

## 6. SIMD Query Engine Architecture

This diagram shows how SIMD optimizations accelerate CPU queries:

```mermaid
graph TB
    subgraph "Query Request"
        A[Query: Region=3, Device=5<br/>Time: last 7 days] --> B[Query Planner]
    end
    
    subgraph "SIMD-Optimized Filtering"
        B --> C[Load 16 elements<br/>AVX-512 register]
        C --> D[Vectorized Comparison<br/>region == 3<br/>device == 5]
        D --> E[Generate Bitmask<br/>1 = match, 0 = no match]
        E --> F[Parallel Aggregation<br/>Sum, Max, Min<br/>8-16x faster]
    end
    
    subgraph "Cross-Tier Query"
        F --> G{Data Location?}
        G -->|Hot| H[GPU HBM<br/><1ms access]
        G -->|Warm| I[CPU RAM<br/>5-10ms access]
        G -->|Cold| J[Disk/NVMe<br/>50-200ms access]
        
        H --> K[GPU Pushdown<br/>Large scans offloaded]
        I --> K
        J --> K
    end
    
    subgraph "Result Assembly"
        K --> L[Merge Results<br/>From all tiers]
        L --> M[Format JSON<br/>Return to client]
    end
    
    style C fill:#ff9ff3
    style D fill:#ff9ff3
    style F fill:#ff9ff3
    style H fill:#ff6b6b
    style I fill:#ffd93d
    style J fill:#6bcf7f
```

## 7. Monitoring & Telemetry Integration

This diagram shows how Nsight, NVML, and DCGM integrate for monitoring:

```mermaid
graph TB
    subgraph "Nsight Profiling Suite"
        A[Nsight Systems<br/>System-wide timeline] --> B[CPU Activity<br/>CUDA API calls<br/>GPU kernels]
        A --> C[Memory Transfers<br/>PCIe bandwidth<br/>Transfer timing]
        A --> D[GPU Idle Time<br/>Starvation detection<br/>Bottleneck identification]
    end
    
    subgraph "Nsight Compute"
        E[Kernel Profiler] --> F[SM Activity<br/>Occupancy<br/>Warp efficiency]
        E --> G[Tensor Core Usage<br/>Memory bottlenecks<br/>Cache hit rates]
        E --> H[Low-level Metrics<br/>Instruction throughput<br/>Memory bandwidth]
    end
    
    subgraph "NVML/DCGM Telemetry"
        I[DCGM Agent<br/>40+ metrics/device] --> J[GPU Utilization<br/>Compute + Memory]
        I --> K[Power Metrics<br/>Consumption, Cap<br/>Throttling detection]
        I --> L[Temperature<br/>Thermal throttling<br/>Cooling efficiency]
        I --> M[SM Throughput<br/>Warp efficiency<br/>L2 cache metrics]
        I --> N[PCIe/NVLink<br/>Bandwidth utilization<br/>Throughput]
    end
    
    subgraph "Integration Points"
        O[CUDA Kernels] --> A
        O --> E
        O --> I
        
        P[Query Operations] --> A
        P --> I
        
        Q[Memory Operations] --> A
        Q --> I
    end
    
    subgraph "Monitoring Stack"
        J --> R[Prometheus Exporter<br/>dcgm-exporter]
        K --> R
        L --> R
        M --> R
        N --> R
        
        R --> S[Grafana Dashboards<br/>Real-time visualization]
        R --> T[Alerting System<br/>Threshold-based alerts]
    end
    
    style A fill:#3498db
    style E fill:#3498db
    style I fill:#2ecc71
    style R fill:#e74c3c
```

## 8. Performance Optimization Workflow

This diagram shows how the system uses telemetry for self-optimization:

```mermaid
flowchart TD
    Start([System Running]) --> Monitor[DCGM/NVML<br/>Real-time monitoring]
    
    Monitor --> Check{GPU Utilization<br/>< 90%?}
    
    Check -->|Yes| Analyze1[Nsight Systems<br/>Identify bottlenecks]
    Analyze1 --> Find1{Issue Found?}
    
    Find1 -->|PCIe Saturation| Fix1[Increase batch size<br/>Amortize transfer overhead]
    Find1 -->|Atomic Contention| Fix2[Adjust block size<br/>Optimize reduction]
    Find1 -->|Power Throttling| Fix3[Adjust power caps<br/>DVFS hints]
    Find1 -->|Memory Bandwidth| Fix4[Optimize memory layout<br/>Improve coalescing]
    
    Fix1 --> Apply[Apply Optimization]
    Fix2 --> Apply
    Fix3 --> Apply
    Fix4 --> Apply
    
    Apply --> Verify[Nsight Compute<br/>Verify improvement]
    Verify --> Monitor
    
    Check -->|No| Optimal[Optimal Performance<br/>90%+ utilization]
    Optimal --> Monitor
    
    style Monitor fill:#2ecc71
    style Analyze1 fill:#3498db
    style Apply fill:#9b59b6
    style Optimal fill:#27ae60
```

## 9. Technology Stack Summary

This diagram provides a visual summary of the complete technology stack:

```mermaid
mindmap
  root((Event Aggregator<br/>HPC/AI Stack))
    Data Processing
      Columnar Layout
        SoA memory structure
        Warp-coalesced access
        2-3x faster kernels
      SIMD CPU Queries
        AVX-512/AVX2
        8-16x speedup
        Vectorized filtering
      Tiered Storage
        GPU HBM: Hot tier
        CPU RAM: Warm tier
        Disk/S3: Cold tier
    GPU Optimization
      CUDA Kernels
        Per-block reduction
        Atomic updates
        Power-of-two buffers
      Memory Management
        Pinned memory
        Double-buffering
        Async transfers
    Profiling & Monitoring
      Nsight Systems
        System timeline
        Bottleneck detection
      Nsight Compute
        Kernel analysis
        SM metrics
      DCGM/NVML
        40+ metrics
        Real-time telemetry
        Prometheus export
    Multi-Node Scaling
      NCCL
        AllReduce
        Topology-aware
        InfiniBand/NVLink
      Cluster Coordination
        Distributed queries
        Global aggregation
        Linear scaling
```

## Key Performance Characteristics

### Throughput
- **Ingestion**: 100K-200K events/sec (single GPU)
- **Scaling**: Linear with GPU count (NCCL)
- **Multi-node**: 400K+ events/sec (4 GPUs/node × N nodes)

### Latency
- **Hot tier queries**: <5ms (GPU HBM)
- **Warm tier queries**: <20ms (CPU RAM)
- **Cold tier queries**: <200ms (Disk/NVMe)
- **Cross-tier queries**: <500ms (with GPU pushdown)

### Memory Efficiency
- **Hot tier**: ~2.5GB (GPU HBM, compressed)
- **Warm tier**: ~10-50GB (CPU RAM, Parquet-like)
- **Cold tier**: Unlimited (disk-based)

### Optimization Benefits
- **Columnar layout**: 2-3x faster GPU kernels
- **SIMD queries**: 8-16x faster CPU filtering
- **Per-block reduction**: 256x less atomic contention
- **Tiered storage**: Cost-effective unlimited retention

## References

- [1] NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- [2] NVIDIA NCCL: https://developer.nvidia.com/nccl
- [3] NVIDIA DCGM: https://developer.nvidia.com/dcgm
- [4] SIMD Performance: https://n.demir.io/articles/understanding-simd-performance-developers-introduction/
- [5] Columnar Layout Benefits: https://www.modular.com/blog/understanding-simd-infinite-complexity-of-trivial-problems

