// build.rs - Cross-platform C++ library linking

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    if target_os == "macos" {
        // ============================================
        // macOS Configuration
        // ============================================
        // SEAL
        println!("cargo:rustc-link-search=native=cpp_wrapper/build");
        println!("cargo:rustc-link-lib=dylib=seal_wrapper");
        
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=dylib=seal-4.1");
        
        // HElib
        println!("cargo:rustc-link-search=native=helib_wrapper/build");
        println!("cargo:rustc-link-lib=dylib=helib_wrapper");
        
        println!("cargo:rustc-link-search=native=/usr/local/helib_pack/helib_pack/lib");
        println!("cargo:rustc-link-lib=dylib=helib");
        println!("cargo:rustc-link-lib=dylib=ntl");
        println!("cargo:rustc-link-lib=dylib=gmp");

        // OpenFHE
        println!("cargo:rustc-link-search=native=openfhe_cpp_wrapper/build");
        println!("cargo:rustc-link-lib=dylib=openfhe_wrapper");
        println!("cargo:rustc-link-lib=dylib=OPENFHEcore");
        println!("cargo:rustc-link-lib=dylib=OPENFHEpke");
        println!("cargo:rustc-link-lib=dylib=OPENFHEbinfhe");
        
        // Runtime paths
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/cpp_wrapper/build");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/helib_wrapper/build");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/helib_pack/helib_pack/lib");
    } else {
        // ============================================
        // Linux Configuration (Docker)
        // ============================================
        // All wrapper libraries are in /app/lib
        println!("cargo:rustc-link-search=native=/app/lib");
        println!("cargo:rustc-link-lib=seal_wrapper");
        println!("cargo:rustc-link-lib=helib_wrapper");
        println!("cargo:rustc-link-lib=openfhe_wrapper");
        
        // System HE libraries
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=seal-4.1");
        println!("cargo:rustc-link-lib=OPENFHEcore");
        println!("cargo:rustc-link-lib=OPENFHEpke");
        println!("cargo:rustc-link-lib=OPENFHEbinfhe");
        
        // HElib libraries
        println!("cargo:rustc-link-search=native=/usr/local/helib_pack/helib_pack/lib");
        println!("cargo:rustc-link-lib=helib");
        println!("cargo:rustc-link-lib=ntl");
        println!("cargo:rustc-link-lib=gmp");
        
        // Runtime paths (rpath) - simplified for runtime image
        println!("cargo:rustc-link-arg=-Wl,-rpath,/app/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/helib_pack/helib_pack/lib");
    }
    
    // ============================================
    // Common Dependencies
    // ============================================
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=gomp");  // GNU OpenMP (needed for OpenFHE)
   
    // ============================================
    // Rerun Triggers
    // ============================================
    println!("cargo:rerun-if-changed=cpp_wrapper/src/seal_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp_wrapper/include/seal_wrapper.h");
    println!("cargo:rerun-if-changed=helib_wrapper/src/helib_wrapper.cpp");
    println!("cargo:rerun-if-changed=helib_wrapper/include/helib_wrapper.h");
}