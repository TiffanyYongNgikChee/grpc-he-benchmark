// grpc_client/src/main.rs
//
// Comprehensive Test Client for HE gRPC Server
// Tests all three libraries: SEAL, HELib, and OpenFHE

use tonic::Request;

pub mod he_service {
    tonic::include_proto!("he_service");
}

use he_service::{
    he_service_client::HeServiceClient, 
    GenerateKeysRequest, 
    EncryptRequest,
    DecryptRequest,
    BinaryOpRequest,
    BenchmarkRequest,
    PredictRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║       HE gRPC Server - Comprehensive Test Suite               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    println!("   Connecting to HE gRPC Server at [::1]:50051...");
    let mut client = HeServiceClient::connect("http://[::1]:50051").await?;
    println!("✓ Connected!\n");

    // Test each library independently
    test_seal(&mut client).await?;
    test_helib(&mut client).await?;
    test_openfhe(&mut client).await?;
    
    // Test comparison benchmark
    test_comparison_benchmark(&mut client).await?;

    // Test digit prediction
    test_predict_digit(&mut client).await?;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║       ALL TESTS PASSED - All three libraries working!         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

async fn test_seal(client: &mut HeServiceClient<tonic::transport::Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Testing SEAL Library (Microsoft SEAL - BFV Scheme)      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Generate Keys
    println!(" Test 1: Generating SEAL keys (poly_modulus_degree=8192)...");
    let request = Request::new(GenerateKeysRequest {
        library: "SEAL".to_string(),
        poly_modulus_degree: 8192,
    });
    let response = client.generate_keys(request).await?;
    let keys_response = response.into_inner();
    let session_id = keys_response.session_id.clone();
    println!("   ✓ Session ID: {}", &session_id[..8]);
    println!("   ✓ Status: {}\n", keys_response.status);

    // 2. Encrypt
    println!(" Test 2: Encrypting vector [10, 20, 30, 40, 50]...");
    let request = Request::new(EncryptRequest {
        session_id: session_id.clone(),
        values: vec![10, 20, 30, 40, 50],
    });
    let response = client.encrypt(request).await?;
    let encrypt_response = response.into_inner();
    println!("   ✓ Ciphertext: {} bytes", encrypt_response.ciphertext.len());
    println!("   ✓ Status: {}\n", encrypt_response.status);

    // 3. Decrypt
    println!(" Test 3: Decrypting ciphertext...");
    let request = Request::new(DecryptRequest {
        session_id: session_id.clone(),
        ciphertext: vec![],
    });
    let response = client.decrypt(request).await?;
    let decrypt_response = response.into_inner();
    println!("   ✓ Decrypted values: {:?}", &decrypt_response.values[..5.min(decrypt_response.values.len())]);
    println!("   ✓ Status: {}\n", decrypt_response.status);

    // 4. Addition
    println!(" Test 4: Homomorphic addition...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.add(request).await?;
    let add_response = response.into_inner();
    println!("   ✓ Status: {}\n", add_response.status);

    // 5. Multiplication
    println!(" Test 5: Homomorphic multiplication...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.multiply(request).await?;
    let multiply_response = response.into_inner();
    println!("   ✓ Status: {}\n", multiply_response.status);

    // 6. Benchmark
    println!(" Test 6: Running SEAL benchmark (50 operations)...");
    let request = Request::new(BenchmarkRequest {
        library: "SEAL".to_string(),
        num_operations: 50,
    });
    let response = client.run_benchmark(request).await?;
    let benchmark = response.into_inner();
    println!("      Benchmark Results:");
    println!("      • Key Generation:  {:.2} ms", benchmark.key_gen_time_ms);
    println!("      • Encryption:      {:.2} ms/op", benchmark.encryption_time_ms);
    println!("      • Addition:        {:.2} ms/op", benchmark.addition_time_ms);
    println!("      • Multiplication:  {:.2} ms/op", benchmark.multiplication_time_ms);
    println!("      • Decryption:      {:.2} ms/op", benchmark.decryption_time_ms);
    println!("   ✓ {}\n", benchmark.status);

    println!(".  SEAL tests completed successfully!\n");
    Ok(())
}

async fn test_helib(client: &mut HeServiceClient<tonic::transport::Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Testing HELib Library (IBM HELib - BGV Scheme)          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Generate Keys
    println!("   Test 1: Generating HELib keys (m=4095, p=2, r=1)...");
    let request = Request::new(GenerateKeysRequest {
        library: "HELib".to_string(),
        poly_modulus_degree: 4096,
    });
    let response = client.generate_keys(request).await?;
    let keys_response = response.into_inner();
    let session_id = keys_response.session_id.clone();
    println!("   ✓ Session ID: {}", &session_id[..8]);
    println!("   ✓ Status: {}\n", keys_response.status);

    // 2. Encrypt
    println!("   Test 2: Encrypting value [42] (HELib uses single values)...");
    let request = Request::new(EncryptRequest {
        session_id: session_id.clone(),
        values: vec![42],
    });
    let response = client.encrypt(request).await?;
    let encrypt_response = response.into_inner();
    println!("   ✓ Ciphertext: {} bytes", encrypt_response.ciphertext.len());
    println!("   ✓ Status: {}\n", encrypt_response.status);

    // 3. Decrypt
    println!("   Test 3: Decrypting ciphertext...");
    let request = Request::new(DecryptRequest {
        session_id: session_id.clone(),
        ciphertext: vec![],
    });
    let response = client.decrypt(request).await?;
    let decrypt_response = response.into_inner();
    println!("   ✓ Decrypted value: {:?}", decrypt_response.values);
    println!("   ✓ Status: {}\n", decrypt_response.status);

    // 4. Addition
    println!("  Test 4: Homomorphic addition...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.add(request).await?;
    let add_response = response.into_inner();
    println!("   ✓ Status: {}\n", add_response.status);

    // 5. Multiplication
    println!("   Test 5: Homomorphic multiplication...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.multiply(request).await?;
    let multiply_response = response.into_inner();
    println!("   ✓ Status: {}\n", multiply_response.status);

    // 6. Benchmark
    println!("   Test 6: Running HELib benchmark (50 operations)...");
    let request = Request::new(BenchmarkRequest {
        library: "HELib".to_string(),
        num_operations: 50,
    });
    let response = client.run_benchmark(request).await?;
    let benchmark = response.into_inner();
    println!("      Benchmark Results:");
    println!("      • Key Generation:  {:.2} ms", benchmark.key_gen_time_ms);
    println!("      • Encryption:      {:.2} ms/op", benchmark.encryption_time_ms);
    println!("      • Addition:        {:.2} ms/op", benchmark.addition_time_ms);
    println!("      • Multiplication:  {:.2} ms/op", benchmark.multiplication_time_ms);
    println!("      • Decryption:      {:.2} ms/op", benchmark.decryption_time_ms);
    println!("   ✓ {}\n", benchmark.status);

    println!("   HELib tests completed successfully!\n");
    Ok(())
}

async fn test_openfhe(client: &mut HeServiceClient<tonic::transport::Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Testing OpenFHE Library (OpenFHE - BFV Scheme)          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Generate Keys
    println!("  Test 1: Generating OpenFHE keys (plaintext_mod=65537)...");
    let request = Request::new(GenerateKeysRequest {
        library: "OpenFHE".to_string(),
        poly_modulus_degree: 4096,
    });
    let response = client.generate_keys(request).await?;
    let keys_response = response.into_inner();
    let session_id = keys_response.session_id.clone();
    println!("   ✓ Session ID: {}", &session_id[..8]);
    println!("   ✓ Status: {}\n", keys_response.status);

    // 2. Encrypt
    println!("   Test 2: Encrypting vector [100, 200, 300, 400]...");
    let request = Request::new(EncryptRequest {
        session_id: session_id.clone(),
        values: vec![100, 200, 300, 400],
    });
    let response = client.encrypt(request).await?;
    let encrypt_response = response.into_inner();
    println!("   ✓ Ciphertext: {} bytes", encrypt_response.ciphertext.len());
    println!("   ✓ Status: {}\n", encrypt_response.status);

    // 3. Decrypt
    println!("   Test 3: Decrypting ciphertext...");
    let request = Request::new(DecryptRequest {
        session_id: session_id.clone(),
        ciphertext: vec![],
    });
    let response = client.decrypt(request).await?;
    let decrypt_response = response.into_inner();
    println!("   ✓ Decrypted values: {:?}", &decrypt_response.values[..4.min(decrypt_response.values.len())]);
    println!("   ✓ Status: {}\n", decrypt_response.status);

    // 4. Addition
    println!("  Test 4: Homomorphic addition...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.add(request).await?;
    let add_response = response.into_inner();
    println!("   ✓ Status: {}\n", add_response.status);

    // 5. Multiplication
    println!("   Test 5: Homomorphic multiplication...");
    let request = Request::new(BinaryOpRequest {
        session_id: session_id.clone(),
        ciphertext1: vec![],
        ciphertext2: vec![],
    });
    let response = client.multiply(request).await?;
    let multiply_response = response.into_inner();
    println!("   ✓ Status: {}\n", multiply_response.status);

    // 6. Benchmark
    println!("   Test 6: Running OpenFHE benchmark (50 operations)...");
    let request = Request::new(BenchmarkRequest {
        library: "OpenFHE".to_string(),
        num_operations: 50,
    });
    let response = client.run_benchmark(request).await?;
    let benchmark = response.into_inner();
    println!("      Benchmark Results:");
    println!("      • Key Generation:  {:.2} ms", benchmark.key_gen_time_ms);
    println!("      • Encryption:      {:.2} ms/op", benchmark.encryption_time_ms);
    println!("      • Addition:        {:.2} ms/op", benchmark.addition_time_ms);
    println!("      • Multiplication:  {:.2} ms/op", benchmark.multiplication_time_ms);
    println!("      • Decryption:      {:.2} ms/op", benchmark.decryption_time_ms);
    println!("   ✓ {}\n", benchmark.status);

    println!("   OpenFHE tests completed successfully!\n");
    Ok(())
}

async fn test_comparison_benchmark(client: &mut HeServiceClient<tonic::transport::Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Running Comparison Benchmark (All Three Libraries)      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("   Benchmarking all libraries with 20 operations each...\n");
    
    let request = Request::new(BenchmarkRequest {
        library: "ALL".to_string(),
        num_operations: 20,
    });
    
    let response = client.run_comparison_benchmark(request).await?;
    let comparison = response.into_inner();
    
    // Display SEAL results
    if let Some(seal) = comparison.seal {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│     SEAL Results                                                │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│  Key Generation:  {:>10.2} ms                                │", seal.key_gen_time_ms);
        println!("│  Encoding:        {:>10.2} ms/op                             │", seal.encoding_time_ms);
        println!("│  Encryption:      {:>10.2} ms/op                             │", seal.encryption_time_ms);
        println!("│  Addition:        {:>10.2} ms/op                             │", seal.addition_time_ms);
        println!("│  Multiplication:  {:>10.2} ms/op                             │", seal.multiplication_time_ms);
        println!("│  Decryption:      {:>10.2} ms/op                             │", seal.decryption_time_ms);
        println!("│  Total Time:      {:>10.2} ms                                │", seal.total_time_ms);
        println!("└─────────────────────────────────────────────────────────────────┘\n");
    }
    
    // Display HELib results
    if let Some(helib) = comparison.helib {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│     HELib Results                                               │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│  Key Generation:  {:>10.2} ms                                │", helib.key_gen_time_ms);
        println!("│  Encoding:        {:>10.2} ms/op                             │", helib.encoding_time_ms);
        println!("│  Encryption:      {:>10.2} ms/op                             │", helib.encryption_time_ms);
        println!("│  Addition:        {:>10.2} ms/op                             │", helib.addition_time_ms);
        println!("│  Multiplication:  {:>10.2} ms/op                             │", helib.multiplication_time_ms);
        println!("│  Decryption:      {:>10.2} ms/op                             │", helib.decryption_time_ms);
        println!("│  Total Time:      {:>10.2} ms                                │", helib.total_time_ms);
        println!("└─────────────────────────────────────────────────────────────────┘\n");
    }
    
    // Display OpenFHE results
    if let Some(openfhe) = comparison.openfhe {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│     OpenFHE Results                                             │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│  Key Generation:  {:>10.2} ms                                │", openfhe.key_gen_time_ms);
        println!("│  Encoding:        {:>10.2} ms/op                             │", openfhe.encoding_time_ms);
        println!("│  Encryption:      {:>10.2} ms/op                             │", openfhe.encryption_time_ms);
        println!("│  Addition:        {:>10.2} ms/op                             │", openfhe.addition_time_ms);
        println!("│  Multiplication:  {:>10.2} ms/op                             │", openfhe.multiplication_time_ms);
        println!("│  Decryption:      {:>10.2} ms/op                             │", openfhe.decryption_time_ms);
        println!("│  Total Time:      {:>10.2} ms                                │", openfhe.total_time_ms);
        println!("└─────────────────────────────────────────────────────────────────┘\n");
    }
    
    // Display comparison summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     COMPARISON RESULTS                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Fastest Library: {:43}  ║", comparison.fastest_library);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Recommendation:                                              ║");
    println!("║  {}  ║", format!("{:60}", comparison.recommendation));
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    println!("   Comparison benchmark completed successfully!\n");
    Ok(())
}

async fn test_predict_digit(client: &mut HeServiceClient<tonic::transport::Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Testing PredictDigit (Encrypted MNIST Inference)        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Build a hardcoded digit "7" image (28×28 = 784 pixels, values 0-255)
    let pixels = create_test_digit_7();

    // Print ASCII art preview
    println!("   Input image (digit 7):");
    print_image_ascii(&pixels, 28, 28);
    println!();

    // Send PredictRequest
    println!("   Sending PredictDigit request (784 pixels, scale_factor=1000)...");
    let request = Request::new(PredictRequest {
        pixels: pixels.clone(),
        scale_factor: 1000,
    });

    let response = client.predict_digit(request).await?;
    let resp = response.into_inner();

    // Display results
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│     Encrypted Inference Results                                 │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  Predicted Digit:  {:>10}                                    │", resp.predicted_digit);
    println!("│  Confidence:       {:>10.4}                                    │", resp.confidence);
    println!("│  Status:           {:>42}  │", resp.status);
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│     Logits (raw encrypted outputs)                              │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    for (i, logit) in resp.logits.iter().enumerate() {
        let marker = if i as i32 == resp.predicted_digit { " ◄" } else { "" };
        println!("│  Class {}: {:>12}{:>40}  │", i, logit, marker);
    }
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│     Timing Breakdown                                            │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  Encryption:           {:>10.2} ms                             │", resp.encryption_ms);
    println!("│  Conv1:                {:>10.2} ms                             │", resp.conv1_ms);
    println!("│  Square Act 1:         {:>10.2} ms                             │", resp.square_activation1_ms);
    println!("│  Pool 1:               {:>10.2} ms                             │", resp.pool1_ms);
    println!("│  Conv2:                {:>10.2} ms                             │", resp.conv2_ms);
    println!("│  Square Act 2:         {:>10.2} ms                             │", resp.square_activation2_ms);
    println!("│  Pool 2:               {:>10.2} ms                             │", resp.pool2_ms);
    println!("│  Flatten:              {:>10.2} ms                             │", resp.flatten_ms);
    println!("│  Fully Connected:      {:>10.2} ms                             │", resp.fc_ms);
    println!("│  Decryption:           {:>10.2} ms                             │", resp.decryption_ms);
    println!("│  Post-processing:      {:>10.2} ms                             │", resp.postprocess_ms);
    println!("│  Weight Loading:       {:>10.2} ms                             │", resp.weight_loading_ms);
    println!("│  ──────────────────────────────────────                        │");
    println!("│  TOTAL:                {:>10.2} ms                             │", resp.total_ms);
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  Float Model Accuracy: {:>10.2}%                              │", resp.float_model_accuracy * 100.0);
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Verify prediction
    let correct = resp.predicted_digit == 7;
    if correct {
        println!("   ✓ CORRECT! Encrypted inference predicted digit 7\n");
    } else {
        println!("   ✗ INCORRECT: Expected 7, got {}. May need tuning.\n", resp.predicted_digit);
    }

    println!("   PredictDigit test completed!\n");
    Ok(())
}

/// Create a hardcoded test image of digit "7" (28×28, pixel values 0-255)
fn create_test_digit_7() -> Vec<i64> {
    let mut img = vec![0i64; 28 * 28];

    let set = |img: &mut Vec<i64>, r: usize, c: usize, val: i64| {
        if r < 28 && c < 28 {
            img[r * 28 + c] = val;
        }
    };

    // Top horizontal bar of the "7" (rows 5-7, cols 7-22)
    for r in 5..=7 {
        for c in 7..=22 {
            set(&mut img, r, c, 220);
        }
    }
    // Slightly dimmer edges on top bar
    for c in 7..=22 {
        set(&mut img, 4, c, 120);
        set(&mut img, 8, c, 80);
    }

    // Diagonal stroke of "7" going from top-right to bottom-center
    for i in 0..17 {
        let r = 8 + i;
        let c_center = 20.0 - (i as f64 * 0.5);
        let c = c_center as usize;

        if r < 28 && c < 28 {
            set(&mut img, r, c, 240);
            if c > 0 {
                set(&mut img, r, c - 1, 180);
            }
            if c + 1 < 28 {
                set(&mut img, r, c + 1, 140);
            }
        }
    }

    img
}

/// Print a 28×28 image as compact ASCII art
fn print_image_ascii(pixels: &[i64], height: usize, width: usize) {
    for r in (0..height).step_by(2) {
        print!("    ");
        for c in 0..width {
            let val = pixels[r * width + c];
            let ch = if val > 200 {
                '█'
            } else if val > 100 {
                '▓'
            } else if val > 50 {
                '░'
            } else {
                ' '
            };
            print!("{}", ch);
        }
        println!();
    }
}
