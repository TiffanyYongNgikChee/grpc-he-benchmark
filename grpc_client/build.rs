fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../proto/he_service.proto")?;
    println!("cargo:rerun-if-changed=../proto/he_service.proto");
    Ok(())
}
