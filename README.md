# Encrypted Machine Learning Benchmark Framework

> **An end-to-end empirical study of privacy-preserving neural network inference using Fully Homomorphic Encryption — quantifying the real latency, noise, and accuracy trade-offs of computing on encrypted data.**

[![CI](https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework/actions)
[![Rust](https://img.shields.io/badge/Rust-1.75+-000000?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![OpenFHE](https://img.shields.io/badge/OpenFHE-1.2.2-blueviolet)](https://www.openfhe.org/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-hexplore--neon.vercel.app-4f46e5)](https://hexplore-neon.vercel.app)

---

## Abstract

Privacy-preserving machine learning demands encryption that survives computation, yet the practical performance cost of Fully Homomorphic Encryption (FHE) remains poorly characterised outside specialist research. Existing tools such as HEBench require local installation, benchmark only primitive operations, and provide no cross-library comparison or end-to-end inference evaluation. The seminal CryptoNets work demonstrated FHE-based neural network classification in 2016, but no accessible, interactive platform has since emerged to let developers observe and compare the real cost of computing on encrypted data.

This project presents **HEXplorer** — a full-stack benchmarking platform that fills this gap. HEXplorer integrates three production FHE libraries (Microsoft SEAL, IBM HElib, and OpenFHE) behind a three-tier architecture (React → Spring Boot → Rust gRPC → C++ HE wrappers) and offers three interactive modes: live encrypted inference on a user-drawn digit, batch accuracy testing on ten MNIST images, and a head-to-head library comparison across five core operations. A LeNet-5 convolutional neural network, retrained with degree-2 polynomial (x²) activations, classifies 28×28 handwritten digits **entirely on ciphertext** using the BFV scheme at 128-bit security. Per-layer inference timings are streamed live to the browser via Server-Sent Events as each layer completes, making bottlenecks visible in real time without post-processing.

**Core findings:** At 128-bit security with degree-2 activations and scale factor 1,000, encrypted CNN inference achieves **88.86% accuracy** — identical to the plaintext baseline — with a latency of ~5–8 seconds per image on an AWS EC2 `r6i.large` instance (32 vCPUs, 256 GB RAM). This represents a ~2,000× overhead compared to plaintext inference and constitutes the first open, reproducible, browser-accessible measurement of end-to-end FHE neural network inference across three libraries on a standard benchmark dataset.

![Project Overview](images/project_overview.png)
---

## Key Results

![Benchmark Result](images/benchmark_result.png)

---

## System Architecture

![System Architecture](images/system_architecture.png)

HEXplorer is built as a **four-layer pipeline**, where each layer has one clear job:

```
[ React Frontend ]
       ↓  draws digit / clicks RUN
[ Spring Boot REST API ]
       ↓  translates HTTP → gRPC, streams SSE back to browser
[ Rust gRPC Server ]
       ↓  calls C++ HE libraries via FFI
[ C++ HE Wrappers ]  ←  OpenFHE · SEAL · HElib
```

| Layer | Technology | What it does |
|---|---|---|
| **Frontend** | React 19 | Provides the drawing canvas, live pipeline animation, and library comparison charts. Communicates with the backend over REST and Server-Sent Events (SSE). |
| **REST Gateway** | Spring Boot 3.2 | Receives HTTP requests from the browser, translates them into gRPC calls, and streams per-layer progress events back to the frontend as SSE. |
| **Inference Engine** | Rust + Tonic gRPC | Orchestrates the CNN forward pass: loads quantised weights, calls the C++ HE library through FFI, and reports each layer's timing as it completes. |
| **HE Libraries** | OpenFHE · SEAL · HElib | C++ libraries that perform the actual homomorphic operations (key generation, encryption, convolution, activation, pooling, decryption). All three are compiled into the same Docker image. |

> **Why this layering?** The browser cannot call C++ directly, and Rust cannot serve HTTP/SSE natively with the same ease as Spring Boot. Each layer uses the best tool for its job, connected by well-defined interfaces (REST → gRPC → FFI).

---

### AWS Deployment Architecture

![AWS diagram](images/aws_architecture.png)

The live deployment splits the system across **two AWS instances** to separate the lightweight frontend-serving concern from the memory-hungry HE computation:

| Component | Instance | Why |
|---|---|---|
| React frontend | Vercel CDN | Always online, zero cost, global CDN — no need to pay for EC2 uptime for static files |
| Spring Boot API | EC2 `t3.small` | Lightweight Java process; just translates HTTP↔gRPC |
| Rust gRPC + HE | EC2 `r6i.large` | OpenFHE key generation at 128-bit security requires ~6 GB RAM; this instance provides headroom |

> **Vercel proxy trick:** Vercel rewrites `/api/*` requests server-side to the EC2 IP. This means the browser only ever talks to `https://hexplore-neon.vercel.app` — no mixed-content (HTTPS→HTTP) browser errors, and the EC2 IP never needs to be exposed publicly.

**Deployment:**
- Frontend → [Vercel](https://hexplore-neon.vercel.app) (CDN, always online)
- Backend → AWS EC2 `r6i.large` (Docker Compose, online during demos)
- Vercel proxies `/api/*` to EC2 server-side, eliminating browser mixed-content restrictions

---

## CNN Pipeline on Encrypted Data

![CNN pipeline](images/digit_inference.png)

**Why x² instead of ReLU?** ReLU requires a comparison to zero — a non-polynomial operation that cannot be evaluated homomorphically without a prohibitively expensive polynomial approximation. The degree-2 polynomial x² is evaluable with a single ciphertext multiplication and is sufficient to introduce the non-linearity required for multi-layer classification (Fan & Vercauteren, 2012; Cheon et al., 2018).

---

## Repository Structure

```
Encrypted-Machine-Learning-Benchmark-Framework/
│
├── proto/                              # Protocol Buffer schema (source of truth)
│   └── he_service.proto                #   All gRPC service + message definitions
│
├── src/                                # Rust core library (HE bindings + inference engine)
│   ├── lib.rs                          #   Crate root; feature flags
│   ├── open_fhe_binding.rs             #   Raw extern "C" FFI declarations for OpenFHE
│   ├── open_fhe_lib.rs                 #   Safe Rust wrappers (OpenFHEContext, Ciphertext, etc.)
│   ├── encrypted_inference.rs          #   Full HE-CNN pipeline (encrypt → forward → decrypt)
│   ├── weight_loader.rs                #   Load quantised CSV weights into BFV plaintexts
│   ├── helib_bindings.rs               #   Raw extern "C" FFI declarations for HElib
│   └── helib.rs                        #   Safe Rust wrappers for HElib
│
├── grpc_server/                        # Rust gRPC server (Tonic framework)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                     #   HEService RPC implementations; listens on :50051
│
├── openfhe_cpp_wrapper/                # C++ layer — OpenFHE (primary inference library)
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── openfhe_wrapper.h           #   BFV context, key generation, encrypt/decrypt API
│   │   └── openfhe_cnn_ops.h           #   conv2d, avgpool, matmul, polynomial activation
│   └── src/
│       ├── openfhe_wrapper.cpp
│       └── openfhe_cnn_ops.cpp
│
├── cpp_wrapper/                        # C++ layer — Microsoft SEAL (micro-benchmarks)
│   ├── CMakeLists.txt
│   ├── include/seal_wrapper.h
│   └── src/
│
├── helib_wrapper/                      # C++ layer — IBM HElib (micro-benchmarks)
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
│
├── spring-boot-api/                    # Java REST gateway (Spring Boot 3.2)
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/com/fyp/hebench/
│       ├── HeBenchApplication.java     #   Application entry point
│       ├── controller/
│       │   └── BenchmarkController.java #  REST endpoints + SSE streaming
│       ├── service/
│       │   └── GrpcClientService.java  #   gRPC stub, protobuf marshalling
│       └── model/                      #   Request/Response POJOs
│
├── frontend/                           # React 19 interactive dashboard
│   ├── Dockerfile                      #   Nginx-served production build
│   ├── nginx.conf                      #   /api/* proxy to Spring Boot
│   ├── vercel.json                     #   Vercel rewrite rules (SPA + API proxy)
│   ├── public/
│   │   └── benchmark_data.json         #   Pre-computed batch benchmark results
│   └── src/
│       ├── api/
│       │   └── client.js               #   REST + SSE fetch wrappers
│       └── workbench/
│           ├── Workbench.js            #   Main page (canvas, pipeline, results)
│           ├── MiniCanvas.js           #   28×28 drawing canvas
│           ├── CnnPipeline.js          #   Animated layer-by-layer diagram
│           ├── LiveStatusFeed.js       #   Real-time SSE log display
│           ├── OutputPanel.js          #   Prediction result + confidence bar
│           ├── MetricsStrip.js         #   Per-layer timing breakdown
│           ├── LibraryComparison.js    #   SEAL vs HElib vs OpenFHE benchmarks
│           ├── MnistBatchBenchmark.js  #   100-image batch result table
│           ├── NeuralHero.js           #   Animated hero section
│           ├── CnnClassroom.js         #   Guided tutorial explainer
│           └── useInferenceProgress.js #   SSE stream state hook
│
├── mnist_training/                     # PyTorch training pipeline
│   ├── train_mnist.py                  #   Train HE_CNN (degree 2 / 3 / 4 activations)
│   ├── verify_plaintext_cnn.py         #   Validate weights reproduce expected accuracy
│   ├── export_test_images.py           #   Export MNIST test images as JSON pixel arrays
│   ├── find_safe_params.py             #   Search for plaintext moduli that avoid overflow
│   ├── check_overflow.py               #   Simulate BFV overflow given scale + modulus
│   ├── requirements.txt
│   ├── weights/                        #   Default weights (symbolic link → weights_deg2/)
│   ├── weights_deg2/                   #   Quantised CSV weights for x² activation
│   ├── weights_deg3/                   #   Quantised CSV weights for cubic activation
│   └── weights_deg4/                   #   Quantised CSV weights for quartic activation
│
├── scripts/
│   ├── run_all_benchmarks.sh           #   Run complete benchmark suite end-to-end
│   ├── csv_to_json.py                  #   Convert raw CSV results → frontend JSON
│   └── setup_new_experiments.sh        #   Provision EC2 and start Docker Compose
│
├── docs/
│   ├── grpc-api.md                     #   Full gRPC API reference
│   └── project-overview.md             #   Architecture decision record
│
├── examples/                           #   Standalone Rust example binaries
│   ├── benchmark.rs
│   ├── mnist_inference.rs
│   └── mnist_benchmark.rs
│
├── Dockerfile                          #   Multi-stage: compile SEAL+HElib+OpenFHE → slim runtime
├── docker-compose.yml                  #   Default: he-grpc-server + spring-boot-api
├── docker-compose.frontend.yml         #   EC2 variant: frontend Nginx + spring-boot-api
├── docker-compose.compute.yml          #   EC2 variant: Rust gRPC server only
├── Cargo.toml                          #   Workspace manifest
└── build.rs                            #   Rust build script (links C++ .so files)
```

---

## Quick Start

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| [Docker](https://docs.docker.com/get-docker/) | 24+ | Required for all-in-one setup |
| [Docker Compose](https://docs.docker.com/compose/) | 2.x | Bundled with Docker Desktop |
| Rust | 1.75+ | Only for local development |
| Java | 17+ | Only for local development |
| Node.js | 20+ | Only for frontend development |
| Python | 3.10+ | Only for model training |

> **Note:** First Docker build compiles OpenFHE, SEAL, and HElib from source. This takes **10–20 minutes** and requires ≥8 GB RAM available to Docker. Subsequent builds use cache.

---

### 1. Clone and Build

```bash
git clone https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework.git
cd Encrypted-Machine-Learning-Benchmark-Framework

# Build all services (takes 10–20 min on first run)
docker compose build
```

### 2. Start the Backend

```bash
# Start the Rust gRPC server and Spring Boot REST API
docker compose up -d he-grpc-server spring-boot-api

# Confirm the API is healthy
curl http://localhost:8080/api/health
# Expected: OK
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```

Or to run the full stack (frontend served by Nginx on port 80):

```bash
docker compose -f docker-compose.frontend.yml up -d
# Opens http://localhost
```

---

## Usage Guide

### Interactive Inference (Browser)

1. Navigate to [http://localhost:3000](http://localhost:3000) (or the [live demo](https://hexplore-neon.vercel.app))
2. Draw a digit (0–9) on the canvas in the **Workbench** panel
3. Click **RUN ▶** — the server encrypts your pixels, runs the 12-layer CNN on ciphertext, and streams per-layer timing back in real time
4. The **Output** panel shows the predicted digit, confidence score, and decrypted logits
5. Use the **Library Comparison** section to benchmark SEAL vs HElib vs OpenFHE on identical operations
6. The **MNIST Batch Benchmark** section shows pre-computed results for 100 real test images

### REST API

**Predict a digit (single image)**

```bash
# Generate 784 zeros as a placeholder; substitute real pixel values
PIXELS=$(python3 -c "print(','.join(['0']*784))")

curl -s -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"pixels\":[$PIXELS],\"scaleFactor\":1000,\"securityLevel\":0,\"activationDegree\":2}" \
  | python3 -m json.tool
```

**Response:**
```json
{
  "predictedDigit": 0,
  "confidence": 0.91,
  "logits": [842, -31, 12, -5, 19, -22, 5, 3, -8, 102],
  "encryptionMs": 118.4,
  "conv1Ms": 812.3,
  "bias1Ms": 4.1,
  "act1Ms": 405.1,
  "pool1Ms": 198.2,
  "conv2Ms": 304.6,
  "bias2Ms": 3.0,
  "act2Ms": 102.4,
  "pool2Ms": 51.3,
  "fcMs": 78.9,
  "biasFcMs": 1.8,
  "decryptionMs": 28.7,
  "totalMs": 5832.7,
  "floatModelAccuracy": 88.86,
  "securityLevelLabel": "128-bit",
  "activationDegree": 2,
  "status": "success"
}
```

**Stream layer-by-layer progress (Server-Sent Events)**

```bash
curl -N -s -X POST http://localhost:8080/api/predict/stream \
  -H "Content-Type: application/json" \
  -d "{\"pixels\":[$PIXELS],\"scaleFactor\":1000,\"securityLevel\":0}"
```

Each event fires as a layer completes:
```
data: {"eventType":"layer_done","layer":"encrypt","layerMs":118.4,"elapsedMs":118.4}
data: {"eventType":"layer_done","layer":"conv1","layerMs":812.3,"elapsedMs":930.7}
data: {"eventType":"layer_done","layer":"act1","layerMs":405.1,"elapsedMs":1335.8}
...
data: {"eventType":"complete","result":{"predictedDigit":0,"totalMs":5832.7,...}}
```

**Run a library benchmark**

```bash
# Benchmark OpenFHE on 10 operations (add, multiply, keygen)
curl -s -X POST http://localhost:8080/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"library":"OpenFHE","numOperations":10}' | python3 -m json.tool

# Compare all three libraries simultaneously
curl -s -X POST http://localhost:8080/api/benchmark/compare \
  -H "Content-Type: application/json" \
  -d '{"numOperations":10}' | python3 -m json.tool
```

### gRPC API (Direct)

Defined in [`proto/he_service.proto`](proto/he_service.proto). Full reference: [`docs/grpc-api.md`](docs/grpc-api.md).

| RPC | Streaming | Description |
|---|---|---|
| `PredictDigit` | Unary | Encrypted CNN inference, returns full result |
| `PredictDigitStream` | Server-streaming | Same, but streams per-layer progress events |
| `RunBenchmark` | Unary | Single-library micro-benchmark |
| `RunComparisonBenchmark` | Unary | All three libraries, parallel |
| `Encrypt` / `Decrypt` | Unary | Low-level encrypt/decrypt a vector |
| `Add` / `Multiply` | Unary | Homomorphic addition / multiplication |

### Train Your Own Model

```bash
cd mnist_training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train degree-2 model (recommended — ~10 min on CPU, ~2 min on GPU)
python train_mnist.py --degree 2

# Trains degree-3 and degree-4 variants as well
# Exports quantised integer weights to weights_deg{2,3,4}/*.csv
```

Then regenerate the batch benchmark JSON:

```bash
python scripts/csv_to_json.py
# Writes: frontend/public/benchmark_data.json
```

---

## Parameter Reference

These parameters control the fundamental accuracy/security/performance trade-off in BFV-based FHE:

| Parameter | What it controls | Values tested | Finding |
|---|---|---|---|
| **Security level** | Ring dimension n; hardness of Ring-LWE problem | 128-bit (n=4096), 192-bit (n=8192), 256-bit (n=16384) | Only 128-bit is feasible on 16 GB RAM. Higher levels OOM during key generation. |
| **Activation degree** | Polynomial approximation of non-linearity | x² (deg 2), x³ (deg 3), x⁴ (deg 4) | x² is the only validated option. Higher degrees overflow the plaintext modulus at inference time. |
| **Scale factor S** | Integer quantisation of floating-point weights: `round(w × S)` | 100, 1,000, 10,000 | S=1000 is the sweet spot — S=100 loses too much weight precision; S=10,000 causes intermediate overflow. |
| **Plaintext modulus p** | Ceiling for all intermediate integer arithmetic | Primes near 10⁸ satisfying p ≡ 1 (mod 2n) | p=100,073,473 provides sufficient headroom for the entire pipeline at S=1,000. |
| **Multiplication depth** | Maximum sequential ciphertext multiplications before noise overwhelms signal | 6 (fixed for this CNN) | Conv1×Act1×Conv2×Act2×FC×spare = depth 6. Determines minimum viable n. |

---

## Tech Stack

| Layer | Technology | Version | Rationale |
|---|---|---|---|
| **HE (primary)** | [OpenFHE](https://www.openfhe.org/) | 1.2.2 | Actively maintained, BFV + CKKS + TFHE, clean C++ API, NIST-standard parameters |
| **HE (benchmarks)** | [Microsoft SEAL](https://github.com/microsoft/SEAL) | 4.1.1 | Industry-standard BFV/CKKS; reference implementation |
| **HE (benchmarks)** | [IBM HElib](https://github.com/homenc/HElib) | 2.3.0 | BGV scheme; bootstrapping support; academic origin |
| **Core runtime** | [Rust](https://www.rust-lang.org/) + [Tonic](https://github.com/hyperium/tonic) | 1.75 / 0.11 | Memory-safe FFI across C++ boundary; zero-cost abstractions; async gRPC |
| **FFI bridge** | C++ `extern "C"` wrappers | C++17 | Only viable way to call OpenFHE/SEAL/HElib from Rust |
| **API gateway** | [Spring Boot](https://spring.io/projects/spring-boot) + [gRPC-Java](https://grpc.io/) | 3.2 / 1.60 | REST + SSE for browser clients; automatic protobuf stub generation |
| **Serialisation** | [Protocol Buffers](https://protobuf.dev/) | 3.25 | Typed cross-language schema; streaming support |
| **Frontend** | [React](https://react.dev/) + [Framer Motion](https://www.framer.com/motion/) | 19 / 12 | Live SSE visualisation; drawing canvas; animated pipeline |
| **ML training** | [PyTorch](https://pytorch.org/) | 2.x | Train HE-compatible CNN; export integer weights as CSV |
| **Container** | [Docker](https://www.docker.com/) multi-stage + Compose | 24+ | Reproducible build compiling all three HE libraries from source |
| **Deployment** | AWS EC2 + [Vercel](https://vercel.com/) | — | EC2 for compute; Vercel for always-on frontend CDN |

---

## Development

### Build C++ Wrappers Locally

> These steps assume OpenFHE, SEAL, and HElib are installed. Use Docker if they are not.

```bash
# OpenFHE wrapper
cd openfhe_cpp_wrapper
mkdir -p build && cd build
cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE
make -j$(nproc)

# SEAL wrapper
cd ../../cpp_wrapper
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# HElib wrapper
cd ../../helib_wrapper
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/helib_pack/helib_pack
make -j$(nproc)
```

### Build Rust

```bash
# From repo root — links against the .so files built above
cargo build --release
```

### Run gRPC Server

```bash
cd grpc_server
cargo run --release
# gRPC server listening on [::]:50051
```

### Run Spring Boot API

```bash
cd spring-boot-api
./mvnw spring-boot:run
# REST API at http://localhost:8080
```

### Run Frontend (dev server)

```bash
cd frontend
npm install
npm start
# http://localhost:3000  (proxies /api/* → localhost:8080)
```

### Run Tests

```bash
# Rust unit tests
cargo test

# Spring Boot tests (unit + mock integration)
cd spring-boot-api && ./mvnw test

# React tests
cd frontend && npm test -- --watchAll=false
```

---

## Deployment

### Vercel (Frontend) + AWS EC2 (Backend)

The frontend is deployed to Vercel as a static site. All `/api/*` requests are transparently proxied by Vercel to the EC2 backend — this eliminates browser mixed-content restrictions without requiring SSL on EC2.

```
Browser → HTTPS → Vercel (proxy) → HTTP → EC2:8080
```

**`vercel.json`** (included in `frontend/`):
```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "http://<EC2-IP>:8080/api/:path*" },
    { "source": "/(.*)",       "destination": "/index.html" }
  ]
}
```

**Redeploy after changes:**
```bash
cd frontend && vercel --prod
```

**Backend auto-start on EC2 reboot:**
```bash
# On EC2 — ensure Docker starts on boot
sudo systemctl enable docker

# Start backend services
docker compose up -d he-grpc-server spring-boot-api
```

> Make sure **port 8080** is open in your EC2 Security Group (inbound TCP from `0.0.0.0/0`).

---

## Troubleshooting

<details>
<summary><strong>Docker build fails — out of memory</strong></summary>

OpenFHE and HElib compilation requires substantial RAM. Ensure Docker Desktop is allocated ≥8 GB, or reduce parallelism:

```bash
# Edit Dockerfile: replace -j$(nproc) with -j2
docker compose build --no-cache
```
</details>

<details>
<summary><strong>Undefined symbol / library linking error at runtime</strong></summary>

```bash
# Verify .so files exist
ls -la /app/lib/*.so
ls -la /usr/local/lib/libOPENFHE*.so

# Refresh the dynamic linker cache
sudo ldconfig

# Set LD_LIBRARY_PATH if running outside Docker
export LD_LIBRARY_PATH=/app/lib:/usr/local/lib:/usr/local/helib_pack/helib_pack/lib:$LD_LIBRARY_PATH
```
</details>

<details>
<summary><strong>Spring Boot cannot connect to gRPC server</strong></summary>

In Docker Compose, Spring Boot connects to the service name `he-grpc-server:50051`. For local (non-Docker) runs, override in `application.properties`:

```properties
grpc.server.host=localhost
grpc.server.port=50051
```
</details>

<details>
<summary><strong>Encrypted prediction returns wrong digit or 0% accuracy</strong></summary>

1. Verify weights are present: `ls mnist_training/weights/*.csv`
2. Confirm scale factor matches training: `scale_factor` must equal the value used in `train_mnist.py` (default: 1000)
3. Confirm pixel values are integers in range 0–255, not floats in 0–1
4. Run plaintext validation: `cd mnist_training && python verify_plaintext_cnn.py`
5. Run overflow diagnostic: `python check_overflow.py` — prints whether intermediate values exceed the plaintext modulus at your chosen scale
</details>

<details>
<summary><strong>Frontend shows "Offline" on Vercel</strong></summary>

The backend EC2 instance may be stopped. Start it:

```bash
# SSH into EC2
docker compose up -d he-grpc-server spring-boot-api

# Verify
curl http://localhost:8080/api/health  # → OK
```
Also verify port 8080 is open in the EC2 Security Group inbound rules.
</details>

---

## References

### Foundational Cryptography

1. R. Rivest, L. Adleman, and M. Dertouzos, "On data banks and privacy homomorphisms," in *Foundations of Secure Computation*, 1978, pp. 169–180. — First theoretical proposal of homomorphic encryption.

2. C. Gentry, "A fully homomorphic encryption scheme," Ph.D. dissertation, Stanford University, 2009. [Online]. Available: https://crypto.stanford.edu/craig — First construction of fully homomorphic encryption.

3. O. Regev, "On lattices, learning with errors, random linear codes, and cryptography," *Journal of the ACM*, vol. 56, no. 6, pp. 1–40, Sep. 2009. — LWE hardness assumption underlying BFV security.

4. Z. Brakerski, C. Gentry, and V. Vaikuntanathan, "(Leveled) fully homomorphic encryption without bootstrapping," in *Proc. ITCS 2012*, pp. 309–325. IACR ePrint 2011/277. — BGV scheme; basis of HElib.

5. J. Fan and F. Vercauteren, "Somewhat practical fully homomorphic encryption," IACR ePrint 2012/144, 2012. [Online]. Available: https://eprint.iacr.org/2012/144 — Specification of the BFV scheme implemented in this project.

6. J. H. Cheon, A. Kim, M. Kim, and Y. Song, "Homomorphic encryption for arithmetic of approximate numbers," in *Proc. ASIACRYPT 2017*, Lecture Notes in Computer Science, vol. 10624, pp. 409–437. IACR ePrint 2016/421. — CKKS scheme for floating-point HE.

### Encrypted Machine Learning

7. R. Gilad-Bachrach, N. Dowlin, K. Laine, K. Lauter, M. Naehrig, and J. Wernsing, "CryptoNets: Applying neural networks to encrypted data with high throughput and accuracy," in *Proc. ICML 2016*, vol. 48, pp. 201–210. — Seminal work on HE-based CNN inference; introduced polynomial activation functions in FHE.

8. H. Chabanne, A. de Wargny, J. Milgram, C. Morel, and E. Prouff, "Privacy-preserving classification on deep neural network," IACR ePrint 2017/035, 2017. [Online]. Available: https://eprint.iacr.org/2017/035 — Batch-normalised HE inference; noise budget management.

9. W. Zhang, Y. Wang, Z. Zheng, Z. Chen, A. Bader, and E. Ng, "BatchCrypt: Efficient homomorphic encryption for Cross-Silo federated learning," in *Proc. USENIX ATC 2020*, pp. 493–506. — HE for federated learning across organisations.

10. M. Blatt, A. Gusev, Y. Polyakov, K. Rohloff, and V. Vaikuntanathan, "Secure large-scale genome-wide association studies using homomorphic encryption," *Proceedings of the National Academy of Sciences*, vol. 117, no. 21, pp. 11608–11613, 2020. — FHE applied to genomic data.

11. C. Boura, N. Gama, M. Georgieva, and D. Jetchev, "CHIMERA: Combining ring-LWE-based fully homomorphic encryption schemes," *Journal of Mathematical Cryptology*, vol. 14, no. 1, pp. 316–338, 2020. — Polynomial activation degree trade-offs in HE-ML.

12. H. Soni and R. Kumar, "Secure and efficient data storage in mobile edge computing using fully homomorphic encryption," *IEEE Internet of Things Journal*, vol. 8, no. 16, pp. 12604–12615, Aug. 2021. — FHE for secure aggregation in healthcare IoT.

13. P. Boura, N. Gama, M. Georgieva, and D. Jetchev, "Illuminating the dark or how to recover what should not be there," *Journal of Mathematical Cryptology*, 2018. — Finance/fraud detection use case for HE deep learning.

### HE Libraries

14. A. Al Badawi, J. Bates, F. Bergamaschi, D. B. Cousins, S. Erabelli, N. Genise, S. Halevi, H. Hunt, A. Kim, Y. Lee, Z. Liu, D. Micciancio, I. Quah, Y. Polyakov, S. R. V., K. Rohloff, J. Saylor, D. Suponitsky, M. Triplett, V. Vaikuntanathan, and V. Zucca, "OpenFHE: Open-source fully homomorphic encryption library," in *Proc. WAHC 2022*, Nov. 2022, pp. 53–63. IACR ePrint 2022/915. [Online]. Available: https://eprint.iacr.org/2022/915 — Primary HE library used for encrypted inference.

15. S. Halevi and V. Shoup, "Bootstrapping for HElib," *Journal of Cryptology*, vol. 34, no. 7, 2021. [Online]. Available: https://github.com/homenc/HElib — IBM HElib; BGV scheme used in micro-benchmarks.

16. H. Chen, K. Laine, and R. Player, "Simple encrypted arithmetic library – SEAL v2.1," in *Proc. Financial Cryptography Workshops 2017*, Lecture Notes in Computer Science, vol. 10323, pp. 3–18, 2017. — Microsoft SEAL BFV implementation used in micro-benchmarks.

### Standards and Benchmarking

17. HomomorphicEncryption.org, "Homomorphic Encryption Standard," 2018. [Online]. Available: https://homomorphicencryption.org/standard/ — Security parameter guidelines (ring dimension, plaintext modulus) for BFV parameter selection.

18. HEBench Project, "HE benchmarking framework," 2022. [Online]. Available: https://hebench.github.io/ — Standardised HE workload specification; primary motivation for HEXplorer's benchmark design.

### Machine Learning and Datasets

19. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. — LeNet-5 architecture used as the CNN baseline in this project.

20. X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proc. AISTATS 2010*, vol. 9, pp. 249–256. — Activation function analysis supporting polynomial activation choice.

### Infrastructure and Protocols

21. gRPC Authors, "gRPC: A high-performance, open-source universal RPC framework," 2016. [Online]. Available: https://grpc.io — gRPC transport layer used between Spring Boot API and Rust inference engine.

22. Google LLC, "Protocol Buffers (protobuf) language guide," 2023. [Online]. Available: https://developers.google.com/protocol-buffers — Schema definition language for gRPC service contracts.

23. M. Belshe, R. Peon, and M. Thomson, "Hypertext Transfer Protocol Version 2 (HTTP/2)," IETF RFC 7540, May 2015. [Online]. Available: https://tools.ietf.org/html/rfc7540 — Underlying transport for gRPC streaming.

24. Pivotal Software, "Spring Boot reference documentation," 2023. [Online]. Available: https://docs.spring.io/spring-boot/docs/current/reference/html/ — Java middleware framework for REST-to-gRPC bridging.

25. L. Lerche and F. LeCuyer, "Tonic: A Rust implementation of gRPC," 2023. [Online]. Available: https://github.com/hyperium/tonic — Rust gRPC library used in the inference server.

26. React Authors, "React: A JavaScript library for building user interfaces," 2023. [Online]. Available: https://react.dev — Frontend framework for the HEXplorer interactive dashboard.

27. Docker Inc., "Docker compose: Multi-container orchestration," 2023. [Online]. Available: https://docs.docker.com/compose/ — Used for multi-stage build and deployment of all HE libraries and services.

### Privacy Regulations and Cloud Standards

28. European Parliament, "General Data Protection Regulation (GDPR)," Regulation (EU) 2016/679, Official Journal of the European Union, May 2018. [Online]. Available: https://gdpr-info.eu — Data protection regulation motivating privacy-preserving inference.

29. P. Mell and T. Grance, "The NIST definition of cloud computing," NIST Special Publication 800-145, Sep. 2011. [Online]. Available: https://csrc.nist.gov/publications/detail/sp/800-145/final — Cloud computing security model referenced for deployment architecture.

---

## License

This project is released under the [MIT License](LICENSE).
<p align="center">
  Built with Rust · C++ · Java · React · PyTorch · OpenFHE · SEAL · HElib · Docker · AWS<br/>
  <sub>Final Year Project 2025–2026</sub>
</p>

