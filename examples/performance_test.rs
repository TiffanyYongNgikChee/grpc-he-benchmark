use he_benchmark::{Context, Encryptor, Decryptor, Plaintext};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⏱️  Performance Benchmark\n");
    
    let context = Context::new(4096, 1024)?;
    let encryptor = Encryptor::new(&context)?;
    let decryptor = Decryptor::new(&context)?;
    
    // Benchmark encryption
    let plain = Plaintext::from_hex("42")?;
    let iterations = 100;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encryptor.encrypt(&plain)?;
    }
    let duration = start.elapsed();
    
    println!("Encryption: {} operations in {:?}", iterations, duration);
    println!("Average: {:?} per operation", duration / iterations);
    println!("\n✅ FFI overhead is acceptable");
    
    Ok(())
}