// grpc_client/src/main.rs
use tonic::Request;

pub mod he_service {
    tonic::include_proto!("he_service");
}

use he_service::{he_service_client::HeServiceClient, GenerateKeysRequest, EncryptRequest, BenchmarkRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔌 Connecting to HE gRPC Server...");
    
    // Connect to the server
    let mut client = HeServiceClient::connect("http://[::1]:50051").await?;
    
    println!("✓ Connected!\n");

    // Test 1: Generate Keys
    println!("═══════════════════════════════════════");
    println!("Test 1: Generating SEAL keys...");
    println!("═══════════════════════════════════════");
    
    let request = Request::new(GenerateKeysRequest {
        library: "SEAL".to_string(),
        poly_modulus_degree: 8192,
    });

    let response = client.generate_keys(request).await?;
    let keys_response = response.into_inner();
    
    println!("✓ Session ID: {}", keys_response.session_id);
    println!("✓ Public Key (bytes): {} bytes", keys_response.public_key.len());
    println!("✓ Status: {}\n", keys_response.status);

    let session_id = keys_response.session_id;

    // Test 2: Encrypt
    println!("═══════════════════════════════════════");
    println!("Test 2: Encrypting values [10, 20, 30]");
    println!("═══════════════════════════════════════");
    
    let request = Request::new(EncryptRequest {
        session_id: session_id.clone(),
        values: vec![10, 20, 30],
    });

    let response = client.encrypt(request).await?;
    let encrypt_response = response.into_inner();
    
    println!("✓ Ciphertext: {} bytes", encrypt_response.ciphertext.len());
    println!("✓ Status: {}\n", encrypt_response.status);

    // Test 3: Run Benchmark
    println!("═══════════════════════════════════════");
    println!("Test 3: Running benchmark for SEAL");
    println!("═══════════════════════════════════════");
    
    let request = Request::new(BenchmarkRequest {
        library: "SEAL".to_string(),
        num_operations: 100,
    });

    let response = client.run_benchmark(request).await?;
    let benchmark = response.into_inner();
    
    println!("📊 Benchmark Results:");
    println!("  Key Generation:  {:.2} ms", benchmark.key_gen_time_ms);
    println!("  Encryption:      {:.2} ms", benchmark.encryption_time_ms);
    println!("  Addition:        {:.2} ms", benchmark.addition_time_ms);
    println!("  Multiplication:  {:.2} ms", benchmark.multiplication_time_ms);
    println!("  Decryption:      {:.2} ms", benchmark.decryption_time_ms);
    println!("  Status: {}\n", benchmark.status);

    println!("✅ All tests completed successfully!");

    Ok(())
}
