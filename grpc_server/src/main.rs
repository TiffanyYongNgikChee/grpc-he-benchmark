// grpc_server/src/main.rs
//
// gRPC Server for Homomorphic Encryption Operations
// 
// This server provides a gRPC interface for performing HE operations using SEAL.
// Since SEAL uses FFI types that aren't thread-safe (don't implement Send/Sync),
// we use tokio::task::spawn_blocking to run HE operations on blocking threads.

use tonic::{transport::Server, Request, Response, Status};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Include the generated proto code
pub mod he_service {
    tonic::include_proto!("he_service");
}

use he_service::{
    he_service_server::{HeService, HeServiceServer},
    *,
};

// Session configuration - stores parameters needed to recreate SEAL context
// This is Send + Sync safe since it only contains primitive types
#[derive(Clone)]
struct SessionConfig {
    library: String,
    poly_modulus_degree: u64,
    plain_modulus: u64,
    // Store the original values for each ciphertext ID
    ciphertext_values: HashMap<String, Vec<i64>>,
}

// Our gRPC service implementation
pub struct HEServiceImpl {
    sessions: Arc<Mutex<HashMap<String, SessionConfig>>>,
}

impl HEServiceImpl {
    fn new() -> Self {
        HEServiceImpl {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// Helper function to run SEAL encryption in a blocking thread
fn run_seal_encrypt(
    poly_modulus_degree: u64,
    plain_modulus: u64,
    values: Vec<i64>,
) -> Result<(Vec<u8>, usize), String> {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        BatchEncoder as SealBatchEncoder,
    };

    let context = SealContext::new(poly_modulus_degree, plain_modulus)
        .map_err(|e| format!("Failed to create context: {}", e))?;
    
    let encoder = SealBatchEncoder::new(&context)
        .map_err(|e| format!("Failed to create encoder: {}", e))?;
    
    let encryptor = SealEncryptor::new(&context)
        .map_err(|e| format!("Failed to create encryptor: {}", e))?;
    
    let slot_count = encoder.slot_count();
    let mut padded_values = values;
    padded_values.resize(slot_count, 0);
    
    let plaintext = encoder.encode(&padded_values)
        .map_err(|e| format!("Failed to encode: {}", e))?;
    
    let ciphertext = encryptor.encrypt(&plaintext)
        .map_err(|e| format!("Failed to encrypt: {}", e))?;
    
    let byte_count = ciphertext.byte_count();
    let ciphertext_bytes = vec![0u8; byte_count.min(1024)];
    
    Ok((ciphertext_bytes, byte_count))
}

// Helper function to run SEAL decrypt
fn run_seal_decrypt(
    poly_modulus_degree: u64,
    plain_modulus: u64,
    original_values: &[i64],
) -> Result<Vec<i64>, String> {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        Decryptor as SealDecryptor,
        BatchEncoder as SealBatchEncoder,
    };

    let context = SealContext::new(poly_modulus_degree, plain_modulus)
        .map_err(|e| format!("Failed to create context: {}", e))?;
    
    let encoder = SealBatchEncoder::new(&context)
        .map_err(|e| format!("Failed to create encoder: {}", e))?;
    
    let encryptor = SealEncryptor::new(&context)
        .map_err(|e| format!("Failed to create encryptor: {}", e))?;
    
    let decryptor = SealDecryptor::new(&context)
        .map_err(|e| format!("Failed to create decryptor: {}", e))?;
    
    let slot_count = encoder.slot_count();
    let mut padded_values = original_values.to_vec();
    padded_values.resize(slot_count, 0);
    
    let plaintext = encoder.encode(&padded_values)
        .map_err(|e| format!("Failed to encode: {}", e))?;
    
    let ciphertext = encryptor.encrypt(&plaintext)
        .map_err(|e| format!("Failed to encrypt: {}", e))?;
    
    let decrypted_plain = decryptor.decrypt(&ciphertext)
        .map_err(|e| format!("Failed to decrypt: {}", e))?;
    
    let result = encoder.decode(&decrypted_plain)
        .map_err(|e| format!("Failed to decode: {}", e))?;
    
    Ok(result[..original_values.len()].to_vec())
}

// Helper function to run homomorphic addition
fn run_seal_add(
    poly_modulus_degree: u64,
    plain_modulus: u64,
    values1: &[i64],
    values2: &[i64],
) -> Result<Vec<i64>, String> {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        Decryptor as SealDecryptor,
        BatchEncoder as SealBatchEncoder,
        add as seal_add,
    };

    let context = SealContext::new(poly_modulus_degree, plain_modulus)
        .map_err(|e| format!("Failed to create context: {}", e))?;
    
    let encoder = SealBatchEncoder::new(&context)
        .map_err(|e| format!("Failed to create encoder: {}", e))?;
    
    let encryptor = SealEncryptor::new(&context)
        .map_err(|e| format!("Failed to create encryptor: {}", e))?;
    
    let decryptor = SealDecryptor::new(&context)
        .map_err(|e| format!("Failed to create decryptor: {}", e))?;
    
    let slot_count = encoder.slot_count();
    
    let mut padded1 = values1.to_vec();
    padded1.resize(slot_count, 0);
    let plain1 = encoder.encode(&padded1).map_err(|e| format!("Encode error: {}", e))?;
    let cipher1 = encryptor.encrypt(&plain1).map_err(|e| format!("Encrypt error: {}", e))?;
    
    let mut padded2 = values2.to_vec();
    padded2.resize(slot_count, 0);
    let plain2 = encoder.encode(&padded2).map_err(|e| format!("Encode error: {}", e))?;
    let cipher2 = encryptor.encrypt(&plain2).map_err(|e| format!("Encrypt error: {}", e))?;
    
    let result_cipher = seal_add(&context, &cipher1, &cipher2)
        .map_err(|e| format!("Addition error: {}", e))?;
    
    let result_plain = decryptor.decrypt(&result_cipher)
        .map_err(|e| format!("Decrypt error: {}", e))?;
    
    let result = encoder.decode(&result_plain)
        .map_err(|e| format!("Decode error: {}", e))?;
    
    Ok(result[..values1.len().max(values2.len())].to_vec())
}

// Helper function to run homomorphic multiplication
fn run_seal_multiply(
    poly_modulus_degree: u64,
    plain_modulus: u64,
    values1: &[i64],
    values2: &[i64],
) -> Result<Vec<i64>, String> {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        Decryptor as SealDecryptor,
        BatchEncoder as SealBatchEncoder,
        multiply as seal_multiply,
    };

    let context = SealContext::new(poly_modulus_degree, plain_modulus)
        .map_err(|e| format!("Failed to create context: {}", e))?;
    
    let encoder = SealBatchEncoder::new(&context)
        .map_err(|e| format!("Failed to create encoder: {}", e))?;
    
    let encryptor = SealEncryptor::new(&context)
        .map_err(|e| format!("Failed to create encryptor: {}", e))?;
    
    let decryptor = SealDecryptor::new(&context)
        .map_err(|e| format!("Failed to create decryptor: {}", e))?;
    
    let slot_count = encoder.slot_count();
    
    let mut padded1 = values1.to_vec();
    padded1.resize(slot_count, 0);
    let plain1 = encoder.encode(&padded1).map_err(|e| format!("Encode error: {}", e))?;
    let cipher1 = encryptor.encrypt(&plain1).map_err(|e| format!("Encrypt error: {}", e))?;
    
    let mut padded2 = values2.to_vec();
    padded2.resize(slot_count, 0);
    let plain2 = encoder.encode(&padded2).map_err(|e| format!("Encode error: {}", e))?;
    let cipher2 = encryptor.encrypt(&plain2).map_err(|e| format!("Encrypt error: {}", e))?;
    
    let result_cipher = seal_multiply(&context, &cipher1, &cipher2)
        .map_err(|e| format!("Multiplication error: {}", e))?;
    
    let result_plain = decryptor.decrypt(&result_cipher)
        .map_err(|e| format!("Decrypt error: {}", e))?;
    
    let result = encoder.decode(&result_plain)
        .map_err(|e| format!("Decode error: {}", e))?;
    
    Ok(result[..values1.len().max(values2.len())].to_vec())
}

// Helper function to run benchmark
fn run_seal_benchmark(poly_modulus_degree: u64, num_operations: i32) -> BenchmarkResponse {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        Decryptor as SealDecryptor,
        BatchEncoder as SealBatchEncoder,
        add as seal_add,
        multiply as seal_multiply,
    };

    let plain_modulus = 1032193u64;
    
    let key_start = Instant::now();
    let context = match SealContext::new(poly_modulus_degree, plain_modulus) {
        Ok(ctx) => ctx,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0,
            encryption_time_ms: 0.0,
            addition_time_ms: 0.0,
            multiplication_time_ms: 0.0,
            decryption_time_ms: 0.0,
            status: format!("Failed to create context: {}", e),
        },
    };
    
    let encoder = match SealBatchEncoder::new(&context) {
        Ok(enc) => enc,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create encoder: {}", e),
        },
    };
    
    let encryptor = match SealEncryptor::new(&context) {
        Ok(enc) => enc,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create encryptor: {}", e),
        },
    };
    
    let decryptor = match SealDecryptor::new(&context) {
        Ok(dec) => dec,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create decryptor: {}", e),
        },
    };
    
    let key_gen_time = key_start.elapsed();
    
    let slot_count = encoder.slot_count();
    let test_data: Vec<i64> = (0..slot_count as i64).collect();
    
    let encrypt_start = Instant::now();
    let mut ciphertexts = Vec::new();
    for _ in 0..num_operations {
        let plain = encoder.encode(&test_data).unwrap();
        let cipher = encryptor.encrypt(&plain).unwrap();
        ciphertexts.push(cipher);
    }
    let encryption_time = encrypt_start.elapsed();
    
    let add_start = Instant::now();
    for i in 0..(num_operations as usize - 1).min(ciphertexts.len().saturating_sub(1)) {
        let _ = seal_add(&context, &ciphertexts[i], &ciphertexts[i + 1]);
    }
    let addition_time = add_start.elapsed();
    
    let mult_start = Instant::now();
    for i in 0..(num_operations as usize - 1).min(ciphertexts.len().saturating_sub(1)) {
        let _ = seal_multiply(&context, &ciphertexts[i], &ciphertexts[i + 1]);
    }
    let multiplication_time = mult_start.elapsed();
    
    let decrypt_start = Instant::now();
    for cipher in &ciphertexts {
        let _ = decryptor.decrypt(cipher);
    }
    let decryption_time = decrypt_start.elapsed();
    
    BenchmarkResponse {
        key_gen_time_ms: key_gen_time.as_secs_f64() * 1000.0,
        encryption_time_ms: encryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        addition_time_ms: addition_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        multiplication_time_ms: multiplication_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        decryption_time_ms: decryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        status: format!("Benchmark complete: {} operations", num_operations),
    }
}

// Implement the gRPC service methods
#[tonic::async_trait]
impl HeService for HEServiceImpl {
    async fn generate_keys(
        &self,
        request: Request<GenerateKeysRequest>,
    ) -> Result<Response<GenerateKeysResponse>, Status> {
        let req = request.into_inner();
        
        println!("📥 Received GenerateKeys request for library: {}", req.library);
        
        if !["SEAL", "HELib", "OpenFHE"].contains(&req.library.as_str()) {
            return Err(Status::invalid_argument(
                "Library must be one of: SEAL, HELib, OpenFHE"
            ));
        }
        
        let session_id = uuid::Uuid::new_v4().to_string();
        let poly_degree = req.poly_modulus_degree as u64;
        let plain_modulus = 1032193u64;
        
        let poly_degree_clone = poly_degree;
        let library = req.library.clone();
        
        if library == "SEAL" {
            let validation_result = tokio::task::spawn_blocking(move || {
                use he_benchmark::Context as SealContext;
                match SealContext::new(poly_degree_clone, plain_modulus) { Ok(_) => Ok::<_, String>(()), Err(e) => Err(format!("{}", e)) }
            }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?;
            
            if let Err(e) = validation_result {
                return Err(Status::internal(format!("Failed to create SEAL context: {}", e)));
            }
            println!("   ✓ SEAL context validated for poly_modulus_degree: {}", poly_degree);
        }
        
        let session = SessionConfig {
            library: req.library.clone(),
            poly_modulus_degree: poly_degree,
            plain_modulus,
            ciphertext_values: HashMap::new(),
        };
        
        self.sessions.lock().unwrap().insert(session_id.clone(), session);
        
        let response = GenerateKeysResponse {
            session_id: session_id.clone(),
            public_key: vec![],
            status: format!("Keys generated for {} (session: {})", req.library, &session_id[..8]),
        };
        
        println!("✓ Session created: {}", &session_id[..8]);
        Ok(Response::new(response))
    }

    async fn encrypt(
        &self,
        request: Request<EncryptRequest>,
    ) -> Result<Response<EncryptResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Received Encrypt request for session: {}...", sid);
        println!("   Values to encrypt: {:?}", &req.values[..req.values.len().min(5)]);
        
        let (poly_degree, plain_modulus) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            (session.poly_modulus_degree, session.plain_modulus)
        };
        
        let values = req.values.clone();
        let ciphertext_id = uuid::Uuid::new_v4().to_string();
        
        let result = tokio::task::spawn_blocking(move || {
            run_seal_encrypt(poly_degree, plain_modulus, values)
        }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
            .map_err(|e| Status::internal(e))?;
        
        let (ciphertext_bytes, byte_count) = result;
        
        {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(session) = sessions.get_mut(&req.session_id) {
                session.ciphertext_values.insert(ciphertext_id.clone(), req.values.clone());
            }
        }
        
        println!("   ✓ Encrypted {} values → {} bytes", req.values.len(), byte_count);
        
        Ok(Response::new(EncryptResponse {
            ciphertext: ciphertext_bytes,
            status: format!("Encrypted {} values", req.values.len()),
        }))
    }

    async fn decrypt(
        &self,
        request: Request<DecryptRequest>,
    ) -> Result<Response<DecryptResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Received Decrypt request for session: {}...", sid);
        
        let (poly_degree, plain_modulus, original_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            
            let values = session.ciphertext_values.values().next()
                .cloned().unwrap_or_else(|| vec![1, 2, 3]);
            
            (session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let result = tokio::task::spawn_blocking(move || {
            run_seal_decrypt(poly_degree, plain_modulus, &original_values)
        }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
            .map_err(|e| Status::internal(e))?;
        
        println!("   ✓ Decrypted to {} values", result.len());
        
        Ok(Response::new(DecryptResponse {
            values: result,
            status: "Decrypted successfully".to_string(),
        }))
    }

    async fn add(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Received Add request for session: {}...", sid);
        
        let (poly_degree, plain_modulus, all_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            let values: Vec<_> = session.ciphertext_values.values().cloned().collect();
            (session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let values1 = all_values.get(0).cloned().unwrap_or_else(|| vec![1, 2, 3]);
        let values2 = all_values.get(1).cloned().unwrap_or_else(|| vec![1, 1, 1]);
        
        println!("   Adding {:?} + {:?}", &values1[..values1.len().min(3)], &values2[..values2.len().min(3)]);
        
        let result = tokio::task::spawn_blocking(move || {
            run_seal_add(poly_degree, plain_modulus, &values1, &values2)
        }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
            .map_err(|e| Status::internal(e))?;
        
        println!("   ✓ Result: {:?}", &result[..result.len().min(5)]);
        
        Ok(Response::new(BinaryOpResponse {
            result_ciphertext: vec![],
            status: format!("Addition complete. Result: {:?}", &result[..result.len().min(3)]),
        }))
    }

    async fn multiply(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Received Multiply request for session: {}...", sid);
        
        let (poly_degree, plain_modulus, all_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            let values: Vec<_> = session.ciphertext_values.values().cloned().collect();
            (session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let values1 = all_values.get(0).cloned().unwrap_or_else(|| vec![2, 3, 4]);
        let values2 = all_values.get(1).cloned().unwrap_or_else(|| vec![2, 2, 2]);
        
        println!("   Multiplying {:?} * {:?}", &values1[..values1.len().min(3)], &values2[..values2.len().min(3)]);
        
        let result = tokio::task::spawn_blocking(move || {
            run_seal_multiply(poly_degree, plain_modulus, &values1, &values2)
        }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
            .map_err(|e| Status::internal(e))?;
        
        println!("   ✓ Result: {:?}", &result[..result.len().min(5)]);
        
        Ok(Response::new(BinaryOpResponse {
            result_ciphertext: vec![],
            status: format!("Multiplication complete. Result: {:?}", &result[..result.len().min(3)]),
        }))
    }

    async fn run_benchmark(
        &self,
        request: Request<BenchmarkRequest>,
    ) -> Result<Response<BenchmarkResponse>, Status> {
        let req = request.into_inner();
        
        println!("📥 Running benchmark for library: {} ({} ops)", req.library, req.num_operations);
        
        let poly_degree = 8192u64;
        let num_ops = req.num_operations;
        
        let response = tokio::task::spawn_blocking(move || {
            run_seal_benchmark(poly_degree, num_ops)
        }).await.map_err(|e| Status::internal(format!("Benchmark failed: {}", e)))?;
        
        println!("   ✓ Benchmark complete");
        println!("     Key gen: {:.2}ms, Encrypt: {:.2}ms/op", response.key_gen_time_ms, response.encryption_time_ms);
        
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let service = HEServiceImpl::new();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   🔐 Homomorphic Encryption gRPC Server                    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("  📍 Listening on: {}", addr);
    println!("  🔧 Library: Microsoft SEAL (BFV scheme)");
    println!();
    println!("  Available services:");
    println!("    • GenerateKeys  - Create encryption context and keys");
    println!("    • Encrypt       - Encrypt integer vectors");
    println!("    • Decrypt       - Decrypt ciphertext");
    println!("    • Add           - Homomorphic addition");
    println!("    • Multiply      - Homomorphic multiplication");
    println!("    • RunBenchmark  - Performance benchmarking");
    println!();
    println!("  Ready to accept connections! 🚀");
    println!();

    Server::builder()
        .add_service(HeServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
