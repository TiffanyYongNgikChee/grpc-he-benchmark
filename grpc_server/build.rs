// grpc_server/build.rs
// This compiles the .proto files into Rust code

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the proto file
    tonic_build::compile_protos("../proto/he_service.proto")?;
    
    println!("cargo:rerun-if-changed=../proto/he_service.proto");
    
    Ok(())
}
