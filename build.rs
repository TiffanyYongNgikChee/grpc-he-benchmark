// build.rs - Cross-platform C++ library linking
//
// Feature flags control which HE libraries are linked:
//   --features seal    → link SEAL wrapper
//   --features helib   → link HElib wrapper
//   --features openfhe → link OpenFHE wrapper
//   --features all_he  → link all three (Docker build)
//
// Default: only OpenFHE (for local development)
// Docker:  cargo build --features all_he

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();

    // Check which libraries to link via feature flags
    let link_seal = std::env::var("CARGO_FEATURE_SEAL").is_ok()
        || std::env::var("CARGO_FEATURE_ALL_HE").is_ok();
    let link_helib = std::env::var("CARGO_FEATURE_HELIB").is_ok()
        || std::env::var("CARGO_FEATURE_ALL_HE").is_ok();
    let link_openfhe = std::env::var("CARGO_FEATURE_OPENFHE").is_ok()
        || std::env::var("CARGO_FEATURE_ALL_HE").is_ok()
        // Default: always link OpenFHE even without explicit feature
        || (!link_seal && !link_helib);

    if target_os == "macos" {
        // ============================================
        // macOS Configuration
        // ============================================

        // Common search paths
        println!("cargo:rustc-link-search=native=/usr/local/lib");

        // SEAL
        if link_seal {
            println!("cargo:rustc-link-search=native=cpp_wrapper/build");
            println!("cargo:rustc-link-lib=dylib=seal_wrapper");
            println!("cargo:rustc-link-lib=dylib=seal-4.1");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/cpp_wrapper/build");
        }

        // HElib
        if link_helib {
            println!("cargo:rustc-link-search=native=helib_wrapper/build");
            println!("cargo:rustc-link-lib=dylib=helib_wrapper");
            println!("cargo:rustc-link-search=native=/usr/local/helib_pack/helib_pack/lib");
            println!("cargo:rustc-link-lib=dylib=helib");
            println!("cargo:rustc-link-lib=dylib=ntl");
            println!("cargo:rustc-link-lib=dylib=gmp");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/helib_wrapper/build");
            println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/helib_pack/helib_pack/lib");
        }

        // OpenFHE
        if link_openfhe {
            println!("cargo:rustc-link-search=native=openfhe_cpp_wrapper/build");
            println!("cargo:rustc-link-lib=dylib=openfhe_wrapper");
            println!("cargo:rustc-link-lib=dylib=OPENFHEcore");
            println!("cargo:rustc-link-lib=dylib=OPENFHEpke");
            println!("cargo:rustc-link-lib=dylib=OPENFHEbinfhe");
        }

        // Runtime paths
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    } else {
        // ============================================
        // Linux Configuration (Docker)
        // ============================================

        // Common search paths
        println!("cargo:rustc-link-search=native=/app/lib");
        println!("cargo:rustc-link-search=native=/usr/local/lib");

        // SEAL
        if link_seal {
            println!("cargo:rustc-link-lib=seal_wrapper");
            println!("cargo:rustc-link-lib=seal-4.1");
        }

        // HElib
        if link_helib {
            println!("cargo:rustc-link-search=native=/usr/local/helib_pack/helib_pack/lib");
            println!("cargo:rustc-link-lib=helib_wrapper");
            println!("cargo:rustc-link-lib=helib");
            println!("cargo:rustc-link-lib=ntl");
            println!("cargo:rustc-link-lib=gmp");
            println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/helib_pack/helib_pack/lib");
        }

        // OpenFHE
        if link_openfhe {
            println!("cargo:rustc-link-lib=openfhe_wrapper");
            println!("cargo:rustc-link-lib=OPENFHEcore");
            println!("cargo:rustc-link-lib=OPENFHEpke");
            println!("cargo:rustc-link-lib=OPENFHEbinfhe");
        }

        // Runtime paths
        println!("cargo:rustc-link-arg=-Wl,-rpath,/app/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    }

    // ============================================
    // Common Dependencies
    // ============================================
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    if link_openfhe {
        println!("cargo:rustc-link-lib=gomp");  // GNU OpenMP (needed for OpenFHE)
    }

    // ============================================
    // Rerun Triggers
    // ============================================
    println!("cargo:rerun-if-changed=cpp_wrapper/src/seal_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp_wrapper/include/seal_wrapper.h");
    println!("cargo:rerun-if-changed=helib_wrapper/src/helib_wrapper.cpp");
    println!("cargo:rerun-if-changed=helib_wrapper/include/helib_wrapper.h");
    println!("cargo:rerun-if-changed=openfhe_cpp_wrapper/src/openfhe_wrapper.cpp");
    println!("cargo:rerun-if-changed=openfhe_cpp_wrapper/include/openfhe_wrapper.h");
    println!("cargo:rerun-if-changed=openfhe_cpp_wrapper/src/openfhe_cnn_ops.cpp");
    println!("cargo:rerun-if-changed=openfhe_cpp_wrapper/include/openfhe_cnn_ops.h");
}