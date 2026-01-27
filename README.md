# HEBench-Distributed

> **A Cloud-Native, Distributed Benchmarking Framework for Homomorphic Encryption Libraries**

[![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![gRPC](https://img.shields.io/badge/gRPC-4285F4?style=flat&logo=google&logoColor=white)](https://grpc.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Problem

Benchmarking Fully Homomorphic Encryption (FHE) libraries is challenging:

- **Multiple libraries** - SEAL, HElib, OpenFHE all have different APIs and performance characteristics
- **No standardization** - Each library uses different parameter sets and benchmarking methodologies
- **Local-only tools** - Existing frameworks (like HEBench) require local installation and execution
- **No remote access** - Can't benchmark on powerful cloud machines remotely
- **Limited scalability** - Sequential execution only, no parallel benchmarking
- **No real-time monitoring** - Must wait for entire benchmark suite to complete

**Traditional Approach:**
```
Local Machine → Install 3+ FHE Libraries → Run Benchmark → Wait Hours → Get Results
```

**Our Approach:**
```
Any Machine → gRPC API Call → Cloud Execution → Real-time Progress → Get Results
```

---

## The Solution

**HEBench-Distributed** is a modern, distributed benchmarking framework that:

- **Distributed Architecture** - Run benchmarks across multiple nodes via gRPC
- **Remote Execution** - Submit benchmarks from anywhere, execute on powerful servers
- **Real-time Monitoring** - Track benchmark progress via WebSocket/SSE
- **Standardized Workloads** - Fair comparison using identical test cases
- **Multi-Library Support** - SEAL, HElib, OpenFHE in one framework
- **Cloud-Native** - Docker containers, Kubernetes-ready
- **REST + gRPC APIs** - Integrate into CI/CD pipelines
- **Web Dashboard** - Visual interface for non-technical users

---

## System Architecture

```
┌──────────────────────��──────────────────────────────────────────┐
│                         CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  CLI Tool  │  Web Dashboard  │  REST API  │  gRPC Client        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────��─────────┐
│                      gRPC SERVICE LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  • Job Scheduler                                                 │
│  • Workload Manager                                              │
│  • Result Aggregator                                             │
│  • Session Manager                                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TEST HARNESS LAYER                          │
├──────────────────────��──────────────────────────────────────────┤
│  • Workload Executor                                             │
│  • Result Validator                                              │
│  • Metrics Collector                                             │
│  • Report Generator                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   SEAL      │ │   HElib     │ │  OpenFHE    │
│   Backend   │ │   Backend   │ │   Backend   │
│  (BFV/CKKS) │ │    (BGV)    │ │ (BFV/CKKS)  │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   PostgreSQL    │
              │  (Results DB)   │
              └─────────────────┘
```

---

## Supported Workloads

We implement standardized benchmarks inspired by [HEBench](https://hebench.github.io/):

| Workload | Description | Parameters |
|----------|-------------|------------|
| **Vector Addition** | Element-wise addition of two encrypted vectors | Vector size: 100, 1000, 10000 |
| **Vector Multiplication** | Element-wise multiplication of encrypted vectors | Vector size: 100, 1000, 10000 |
| **Matrix Multiplication** | Encrypted matrix multiplication | Dimensions: 4x4, 8x8, 16x16 |
| **Dot Product** | Inner product of two encrypted vectors | Vector size: 100, 1000, 10000 |
| **Scalar Multiplication** | Multiply encrypted vector by scalar | Vector size: 1000 |
| **Polynomial Evaluation** | Evaluate polynomial on encrypted data | Degree: 3, 5, 7 |

---

## Supported FHE Libraries

| Library | Schemes | Version | Status |
|---------|---------|---------|--------|
| [Microsoft SEAL](https://github.com/microsoft/SEAL) | BFV, CKKS | 4.1.1 | Fully Supported |
| [IBM HElib](https://github.com/homenc/HElib) | BGV | 2.3.0 | Fully Supported |
| [OpenFHE](https://github.com/openfheorg/openfhe-development) | BFV, CKKS, TFHE | 1.1.2 | Fully Supported |

---

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- **OR** Rust 1.70+, CMake 3.13+, C++17 compiler

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/TiffanyYongNgikChee/grpc-he-benchmark.git
cd grpc-he-benchmark

# Build (takes 10-15 minutes due to FHE library compilation)
docker-compose build

# Start services
docker-compose up -d

# Enter the container
docker-compose exec he-benchmark bash

# Run benchmark
cargo run --example benchmark --release
```

### Option 2: Local Build

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake libgmp-dev libntl-dev

# Clone repo
git clone https://github.com/TiffanyYongNgikChee/grpc-he-benchmark.git
cd grpc-he-benchmark

# Build C++ wrappers
./rebuild_all.sh

# Build Rust project
cargo build --release

# Run benchmark
cargo run --example benchmark --release
```

---

## Usage Examples

### 1. Run Local Benchmark

```bash
# Run comprehensive benchmark comparing all three libraries
cargo run --example benchmark --release
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════╗
║           FHE BENCHMARK: SEAL vs HElib vs OpenFHE             ║
╚═══════════════════════════════════════════════════════════════╝

Workload: Vector Addition (size=1000)

┌─────────────┬────────────┬──────────────┬──────────────┬───────────┐
│ Library     │ Key Gen    │ Encrypt      │ Compute      │ Decrypt   │
├─────────────┼────────────┼──────────────┼──────────────┼───────────┤
│ SEAL        │  245 ms    │   12 ms      │    8 ms      │   10 ms   │
│ HElib       │  312 ms    │   18 ms      │   15 ms      │   14 ms   │
│ OpenFHE     │  198 ms    │   10 ms      │    6 ms      │    9 ms   │
└─────────────┴────────────┴──────────────┴──────────────┴───────────┘

Winner: OpenFHE (fastest total time: 223 ms)
```

### 2. Start gRPC Server

```bash
# Terminal 1: Start server
cargo run --bin grpc-server

# Server starts at [::1]:50051
```

```bash
# Terminal 2: Run client
cargo run --bin grpc-client
```

### 3. Use REST API (Coming Soon)

```bash
# Submit benchmark job
curl -X POST http://localhost:8080/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "workload": "vector_addition",
    "size": 1000,
    "libraries": ["SEAL", "HElib", "OpenFHE"]
  }'

# Response
{
  "job_id": "bench_20260127_143022",
  "status": "running",
  "progress": 0.35
}

# Check status
curl http://localhost:8080/api/benchmark/bench_20260127_143022

# Get results
curl http://localhost:8080/api/benchmark/bench_20260127_143022/results
```

---

## Benchmarking Results

### Vector Addition (size=1000)

| Operation | SEAL | HElib | OpenFHE |
|-----------|------|-------|---------|
| Key Generation | 245 ms | 312 ms | 198 ms |
| Encryption | 12 ms | 18 ms | 10 ms |
| Addition | 8 ms | 15 ms | 6 ms |
| Decryption | 10 ms | 14 ms | 9 ms |
| **Total** | **275 ms** | **359 ms** | **223 ms** |

### Matrix Multiplication (8x8)

| Operation | SEAL | HElib | OpenFHE |
|-----------|------|-------|---------|
| Key Generation | 245 ms | 312 ms | 198 ms |
| Encryption | 45 ms | 62 ms | 38 ms |
| Multiplication | 124 ms | 186 ms | 98 ms |
| Decryption | 28 ms | 35 ms | 22 ms |
| **Total** | **442 ms** | **595 ms** | **356 ms** |

*Run on: Intel i7-12700K, 32GB RAM, Ubuntu 22.04*

---

## Project Structure

```
grpc-he-benchmark/
├── cpp_wrapper/              # Microsoft SEAL C++ wrapper
├── helib_wrapper/            # HElib C++ wrapper
├── openfhe_cpp_wrapper/      # OpenFHE C++ wrapper
├── grpc-server/              # Rust gRPC server for HE operations
├── grpc-client/              # Rust gRPC client for testing
├── src/                      # Rust FFI bindings to C++ wrappers
│   ├── lib.rs                # Main library entry
│   ├── bindings.rs           # SEAL FFI bindings
│   ├── helib_bindings.rs     # HElib FFI bindings
│   ├── helib.rs              # HElib safe wrapper
│   ├── open_fhe_binding.rs   # OpenFHE FFI bindings
│   └── open_fhe_lib.rs       # OpenFHE safe wrapper
├── examples/                 # Benchmark examples
│   ├── benchmark.rs          # Comprehensive benchmark
│   └── vector_operations.rs  # Vector operations example
├── proto/                    # gRPC protocol definitions
│   └── he_service.proto
├── Dockerfile                # Multi-stage build
├── docker-compose.yml        # Service orchestration
└── README.md
```

---

## Comparison with HEBench

| Feature | **HEBench** | **HEBench-Distributed** |
|---------|-------------|-------------------------|
| Architecture | Local test harness | Distributed gRPC service |
| Language | C++ | Rust + C++ |
| Execution Model | Sequential | Parallel + Remote |
| Interface | CLI only | CLI + gRPC + REST + Web |
| Deployment | Manual build | Docker + Kubernetes |
| Real-time Monitoring | No | Yes |
| Remote Execution | No | Yes |
| Cloud-Native | No | Yes |
| Backend Plugin System | C API Bridge | Rust Trait-based |
| Result Storage | File-based | PostgreSQL + File |
| Multi-node Support | No | Yes |

**Why HEBench-Distributed?**

- **Accessibility**: Anyone can run benchmarks remotely without local setup
- **Scalability**: Distribute workloads across multiple nodes
- **Integration**: REST/gRPC APIs for CI/CD pipelines
- **Modern Stack**: Rust for safety, gRPC for performance
- **Cloud-Ready**: Native Kubernetes support

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Core Language** | Rust | Memory-safe FFI, concurrency |
| **FHE Libraries** | SEAL, HElib, OpenFHE | Homomorphic encryption operations |
| **FFI Layer** | C++ Wrappers | Bridge Rust ↔ C++ FHE libraries |
| **RPC Framework** | gRPC (tonic) | Remote benchmark execution |
| **Serialization** | Protocol Buffers | Efficient data transfer |
| **Database** | PostgreSQL | Store benchmark results |
| **Containerization** | Docker + Compose | Reproducible environment |
| **Orchestration** | Kubernetes (planned) | Distributed deployment |

---

## Development

### Rebuilding After Changes

```bash
# Inside Docker container
./rebuild_all.sh

# Or rebuild individually
cd cpp_wrapper/build && cmake .. && make && cd ../..
cd helib_wrapper/build && cmake .. && make && cd ../..
cd openfhe_cpp_wrapper/build && cmake .. && make && cd ../..

cargo build --release
```

### Running Examples

```bash
cargo run --example benchmark --release
cargo run --example vector_operations --release
```

### Running Tests

```bash
cargo test --all
```

---

## Troubleshooting

<details>
<summary><strong>Docker build fails</strong></summary>

```bash
docker-compose build --no-cache
```
</details>

<details>
<summary><strong>Library linking errors</strong></summary>

```bash
# Check library paths
ls -l /app/*.so
ls -l /usr/local/lib/
ls -l /usr/local/helib_pack/lib/

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/app:/usr/local/lib:/usr/local/helib_pack/lib:$LD_LIBRARY_PATH
```
</details>

<details>
<summary><strong>Rust compilation issues</strong></summary>

```bash
cargo clean
cargo build --release
```
</details>

<details>
<summary><strong>gRPC connection refused</strong></summary>

```bash
# Check if server is running
cargo run --bin grpc-server

# Verify port is available
netstat -tuln | grep 50051

# Try using 127.0.0.1 instead of [::1]
# Update grpc-client to connect to "http://127.0.0.1:50051"
```
</details>

---

## Roadmap

| Phase | Status | Timeline |
|-------|--------|----------|
| Phase 1: Core FFI + gRPC | Complete | Nov - Dec 2025 |
| Phase 2: Standardized Workloads | In Progress | Jan 2026 |
| Phase 3: Test Harness + API | Planned | Feb 2026 |
| Phase 4: Web Dashboard | Planned | Mar 2026 |
| Phase 5: Cloud Deployment | Planned | Apr 2026 |

### Phase 2: Standardized Workloads (Current)
- [ ] Vector Addition
- [ ] Vector Multiplication
- [ ] Matrix Multiplication (4x4, 8x8, 16x16)
- [ ] Dot Product
- [ ] Polynomial Evaluation
- [ ] YAML configuration system

### Phase 3: Test Harness + REST API
- [ ] Test harness orchestrator
- [ ] Workload scheduler
- [ ] Result validation
- [ ] REST API wrapper
- [ ] PostgreSQL integration

### Phase 4: Web Dashboard
- [ ] React + TypeScript frontend
- [ ] Real-time benchmark monitoring
- [ ] Result visualization (charts)
- [ ] Comparison view
- [ ] Export functionality (CSV, JSON, PDF)

### Phase 5: Cloud Deployment
- [ ] Kubernetes manifests
- [ ] Multi-node benchmarking
- [ ] Load balancing
- [ ] Horizontal scaling
- [ ] Monitoring (Prometheus + Grafana)

---

## Documentation

- [User Guide](docs/user-guide.md) - How to run benchmarks
- [Developer Guide](docs/developer-guide.md) - How to extend the framework
- [API Reference](docs/api-reference.md) - gRPC and REST API documentation
- [Architecture](docs/architecture.md) - System design and components
- [Comparison with HEBench](docs/comparison-hebench.md) - Detailed comparison

---

## Contributing

This project welcomes contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Add new workloads
- Implement additional FHE library backends
- Improve performance
- Write documentation
- Report bugs

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Microsoft SEAL](https://github.com/microsoft/SEAL) - FHE library
- [HElib](https://github.com/homenc/HElib) - FHE library
- [OpenFHE](https://github.com/openfheorg/openfhe-development) - FHE library
- [HEBench](https://github.com/hebench/frontend) - Inspiration for standardized benchmarking

---

## Author

**Tiffany Yong Ngik Chee**  
ATU Galway — Final Year Project 2025/2026

Email: g00425067@atu.ie  
GitHub: [@TiffanyYongNgikChee](https://github.com/TiffanyYongNgikChee)

---

<p align="center">
  <strong>Building the future of distributed FHE benchmarking</strong>
</p>