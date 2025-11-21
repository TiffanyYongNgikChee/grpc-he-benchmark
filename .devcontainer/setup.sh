#!/bin/bash
# filepath: /workspaces/he-benchmark-spike/.devcontainer/setup.sh

set -e

echo " Starting HE Benchmark Dev Container Setup..."

# Update package lists
echo " Updating package lists..."
sudo apt-get update

# Install build essentials and dependencies
echo " Installing build dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgmp-dev \
    libntl-dev \
    m4 \
    autoconf \
    libtool \
    pkg-config \
    clang \
    lldb

# Install SEAL (Microsoft SEAL)
echo " Building Microsoft SEAL..."
if [ ! -d "/tmp/SEAL" ]; then
    cd /tmp
    git clone https://github.com/microsoft/SEAL.git
    cd SEAL
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEAL_USE_INTEL_HEXL=OFF
    cmake --build build -j$(nproc)
    sudo cmake --install build
    echo " SEAL installed successfully"
else
    echo "⏭  SEAL already exists, skipping..."
fi

# Install HElib
echo " Building HElib..."
if [ ! -d "/tmp/HElib" ]; then
    cd /tmp
    git clone https://github.com/homenc/HElib.git
    cd HElib
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED=ON ..
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    echo " HElib installed successfully"
else
    echo "⏭  HElib already exists, skipping..."
fi

# Build project's C++ wrappers
echo "🔨 Building C++ wrappers..."
cd /workspaces/he-benchmark-spike

if [ -d "cpp_wrapper/build" ]; then
    cd cpp_wrapper/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    echo " SEAL wrapper built"
fi

if [ -d "helib_wrapper/build" ]; then
    cd /workspaces/he-benchmark-spike/helib_wrapper/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    echo " HElib wrapper built"
fi

# Verify Rust installation
echo " Verifying Rust installation..."
rustc --version
cargo --version

# Build Rust project
echo " Building Rust project..."
cd /workspaces/he-benchmark-spike
cargo build --release

# Run quick verification
echo " Running verification tests..."
cargo test --lib || echo "  Some tests failed, but continuing..."

echo ""
echo "✨ Setup complete! You can now:"
echo "   • Run benchmarks: cargo run --example benchmark"
echo "   • Run tests: cargo test"
echo "   • Build release: cargo build --release"
echo ""