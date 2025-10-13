// build.rs - Tells Cargo how to link C++ library (macOS version)

fn main() {
    // Link to wrapper library in cpp_wrapper/build
    println!("cargo:rustc-link-search=native=cpp_wrapper/build");
    println!("cargo:rustc-link-lib=dylib=seal_wrapper");
    
    // Link to SEAL static library
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=seal-4.1");  // Changed to static!
    
    // Link C++ standard library (needed for static linking)
    println!("cargo:rustc-link-lib=c++");
    
    // Add rpath for wrapper library
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    println!("cargo:rustc-link-arg=-Wl,-rpath,cpp_wrapper/build");
    
    // Rerun if wrapper changes
    println!("cargo:rerun-if-changed=cpp_wrapper/src/seal_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp_wrapper/include/seal_wrapper.h");
}