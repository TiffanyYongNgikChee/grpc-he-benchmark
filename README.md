# Homomorphic Encryption Benchmark Framework

A comprehensive benchmarking framework comparing three major homomorphic encryption libraries: Microsoft SEAL, HElib, and OpenFHE.

## 🚀 Quick Start with Docker

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) installed

### Clone and Run

```bash
# Clone the repository
git clone https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework.git
cd Encrypted-Machine-Learning-Benchmark-Framework

# Build the Docker image (this takes 10-15 minutes due to HE library compilation)
docker-compose build

# Start the container
docker-compose up -d

# Enter the container
docker-compose exec he-benchmark bash

# Inside the container, build the Rust project
cargo build --release

# Run benchmarks
cargo run --example benchmark --release
```

## 📦 What's Inside

This project includes:
- **Microsoft SEAL v4.1.1**: BFV and CKKS schemes
- **HElib v2.3.0**: BGV scheme
- **OpenFHE v1.2.3**: Multiple schemes support
- **Rust Bindings**: Safe FFI wrappers for all three libraries
- **Benchmarking Tools**: Performance comparison utilities

## 🏗️ Architecture

```
├── cpp_wrapper/          # Microsoft SEAL C++ wrapper
├── helib_wrapper/        # HElib C++ wrapper
├── openfhe_cpp_wrapper/  # OpenFHE C++ wrapper
├── src/                  # Rust FFI bindings
├── examples/             # Example programs and benchmarks
├── Dockerfile            # Multi-stage build for all HE libraries
└── docker-compose.yml    # Easy orchestration
```

## 🔧 Development Workflow

### Rebuilding After Changes

If you modify the C++ wrappers or Rust code:

```bash
# Inside the Docker container

# Clean and rebuild everything
./rebuild_all.sh

# Or rebuild individually:
cd cpp_wrapper/build && cmake .. && make && cd ../..
cd helib_wrapper/build && cmake .. && make && cd ../..
cd openfhe_cpp_wrapper/build && cmake .. && make && cd ../..

# Then rebuild Rust
cargo build --release
```

### Running Examples

```bash
# Inside the Docker container
cargo run --example benchmark --release
cargo run --example medical_data --release
```

### Exiting the Container

```bash
# Exit the shell
exit

# Stop the container
docker-compose down
```

## 📊 Benchmarking

The benchmark suite tests:
- ✅ Encryption/Decryption performance
- ✅ Homomorphic operations (addition, multiplication)
- ✅ Memory usage
- ✅ Key generation time
- ✅ Relinearization overhead

## 🐛 Troubleshooting

### Build Fails in Docker

```bash
# Rebuild without cache
docker-compose build --no-cache
```

### Library Linking Errors

Make sure the `.so` files are in the correct locations:
```bash
ls -l /app/*.so
ls -l /usr/local/lib/
ls -l /usr/local/helib_pack/lib/
```

### Rust Compilation Issues

```bash
# Clean Cargo build cache
cargo clean
cargo build --release
```

## 🔍 Library Paths

The Docker container sets up the following paths:
- **SEAL**: `/usr/local/lib/libseal-4.1.so`
- **HElib**: `/usr/local/helib_pack/lib/libhelib.so`
- **OpenFHE**: `/usr/local/lib/libOPENFHE*.so`
- **Wrappers**: `/app/libseal_wrapper.so`, `/app/libhelib_wrapper.so`, `/app/libopenfhe_wrapper.so`

## 📝 Environment Variables

The following are pre-configured in the Docker container:
```bash
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/helib_pack/lib
RUST_BACKTRACE=1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test inside Docker container
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [HElib](https://github.com/homenc/HElib)
- [OpenFHE](https://github.com/openfheorg/openfhe-development)

## 📧 Contact

Tiffany Yong - [Your contact info]
