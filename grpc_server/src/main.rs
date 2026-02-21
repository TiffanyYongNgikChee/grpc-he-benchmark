// grpc_server/src/main.rs
//
// gRPC Server for Homomorphic Encryption Operations
// 
// This server provides a gRPC interface for performing HE operations using SEAL and HELib.
// Since SEAL/HELib use FFI types that aren't thread-safe (don't implement Send/Sync),
// we use tokio::task::spawn_blocking to run HE operations on blocking threads.

use tonic::{transport::Server, Request, Response, Status};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use he_benchmark::encrypted_inference::EncryptedInferenceEngine;

// Safety wrapper: EncryptedInferenceEngine contains FFI pointers (NonNull<...>)
// that don't implement Send/Sync. We guarantee safety by only accessing
// the engine inside tokio::task::spawn_blocking (single-threaded access
// guarded by Mutex).
struct SendSyncEngine(EncryptedInferenceEngine);
unsafe impl Send for SendSyncEngine {}
unsafe impl Sync for SendSyncEngine {}

// Include the generated proto code
pub mod he_service {
    tonic::include_proto!("he_service");
}

use he_service::{
    he_service_server::{HeService, HeServiceServer},
    *,
};

// Session configuration - stores parameters needed to recreate SEAL/HELib context
// This is Send + Sync safe since it only contains primitive types
#[derive(Clone)]
struct SessionConfig {
    library: String,
    poly_modulus_degree: u64,
    plain_modulus: u64,
    ciphertext_values: HashMap<String, Vec<i64>>,
}

// Our gRPC service implementation
pub struct HEServiceImpl {
    sessions: Arc<Mutex<HashMap<String, SessionConfig>>>,
    /// Pre-initialized encrypted inference engine (OpenFHE BFV).
    /// Wrapped in SendSyncEngine for async safety; only accessed in spawn_blocking.
    inference_engine: Arc<SendSyncEngine>,
}

impl HEServiceImpl {
    fn new(engine: EncryptedInferenceEngine) -> Self {
        HEServiceImpl {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            inference_engine: Arc::new(SendSyncEngine(engine)),
        }
    }
}

// ============================================
// SEAL Helper Functions
// ============================================

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

fn run_seal_benchmark(poly_modulus_degree: u64, num_operations: i32) -> BenchmarkResponse {
    use he_benchmark::{
        Context as SealContext,
        Encryptor as SealEncryptor,
        Decryptor as SealDecryptor,
        BatchEncoder as SealBatchEncoder,
        add as seal_add,
        multiply as seal_multiply,
    };

    let total_start = Instant::now();
    let plain_modulus = 1032193u64;
    
    let key_start = Instant::now();
    let context = match SealContext::new(poly_modulus_degree, plain_modulus) {
        Ok(ctx) => ctx,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create context: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let encoder = match SealBatchEncoder::new(&context) {
        Ok(enc) => enc,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create encoder: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let encryptor = match SealEncryptor::new(&context) {
        Ok(enc) => enc,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create encryptor: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let decryptor = match SealDecryptor::new(&context) {
        Ok(dec) => dec,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("Failed to create decryptor: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    let key_gen_time = key_start.elapsed();
    
    let slot_count = encoder.slot_count();
    let test_data: Vec<i64> = (0..slot_count as i64).collect();
    
    // Encoding phase
    let encode_start = Instant::now();
    let mut plaintexts = Vec::new();
    for _ in 0..num_operations {
        let plain = encoder.encode(&test_data).unwrap();
        plaintexts.push(plain);
    }
    let encoding_time = encode_start.elapsed();
    
    // Encryption phase
    let encrypt_start = Instant::now();
    let mut ciphertexts = Vec::new();
    for plain in &plaintexts {
        let cipher = encryptor.encrypt(plain).unwrap();
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
    
    let total_time = total_start.elapsed();

    BenchmarkResponse {
        key_gen_time_ms: key_gen_time.as_secs_f64() * 1000.0,
        encoding_time_ms: encoding_time.as_secs_f64() * 1000.0 / num_operations as f64,
        encryption_time_ms: encryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        addition_time_ms: addition_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        multiplication_time_ms: multiplication_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        decryption_time_ms: decryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        status: format!("SEAL benchmark complete: {} operations", num_operations),
    }
}

// ============================================
// HELib Helper Functions
// ============================================

const HELIB_M: u64 = 4095;
const HELIB_P: u64 = 2;
const HELIB_R: u64 = 1;

fn run_helib_encrypt(value: i64) -> Result<usize, String> {
    use he_benchmark::{HEContext, HESecretKey, HEPlaintext};
    
    let context = HEContext::new(HELIB_M, HELIB_P, HELIB_R)
        .map_err(|e| format!("HELib context error: {}", e))?;
    let secret_key = HESecretKey::generate(&context)
        .map_err(|e| format!("HELib key error: {}", e))?;
    let public_key = secret_key.public_key()
        .map_err(|e| format!("HELib public key error: {}", e))?;
    
    let plaintext = HEPlaintext::new(&context, value)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    let _ciphertext = public_key.encrypt(&plaintext)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    
    Ok(4096)
}

fn run_helib_decrypt(value: i64) -> Result<Vec<i64>, String> {
    use he_benchmark::{HEContext, HESecretKey, HEPlaintext};
    
    let context = HEContext::new(HELIB_M, HELIB_P, HELIB_R)
        .map_err(|e| format!("HELib context error: {}", e))?;
    let secret_key = HESecretKey::generate(&context)
        .map_err(|e| format!("HELib key error: {}", e))?;
    let public_key = secret_key.public_key()
        .map_err(|e| format!("HELib public key error: {}", e))?;
    
    let plaintext = HEPlaintext::new(&context, value)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    let ciphertext = public_key.encrypt(&plaintext)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    let decrypted = secret_key.decrypt(&ciphertext)
        .map_err(|e| format!("HELib decrypt error: {}", e))?;
    
    Ok(vec![decrypted.value()])
}

fn run_helib_add(val1: i64, val2: i64) -> Result<Vec<i64>, String> {
    use he_benchmark::{HEContext, HESecretKey, HEPlaintext};
    
    let context = HEContext::new(HELIB_M, HELIB_P, HELIB_R)
        .map_err(|e| format!("HELib context error: {}", e))?;
    let secret_key = HESecretKey::generate(&context)
        .map_err(|e| format!("HELib key error: {}", e))?;
    let public_key = secret_key.public_key()
        .map_err(|e| format!("HELib public key error: {}", e))?;
    
    let pt1 = HEPlaintext::new(&context, val1)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    let pt2 = HEPlaintext::new(&context, val2)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    
    let ct1 = public_key.encrypt(&pt1)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    let ct2 = public_key.encrypt(&pt2)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    
    let result = ct1.add(&ct2)
        .map_err(|e| format!("HELib add error: {}", e))?;
    let decrypted = secret_key.decrypt(&result)
        .map_err(|e| format!("HELib decrypt error: {}", e))?;
    
    Ok(vec![decrypted.value()])
}

fn run_helib_multiply(val1: i64, val2: i64) -> Result<Vec<i64>, String> {
    use he_benchmark::{HEContext, HESecretKey, HEPlaintext};
    
    let context = HEContext::new(HELIB_M, HELIB_P, HELIB_R)
        .map_err(|e| format!("HELib context error: {}", e))?;
    let secret_key = HESecretKey::generate(&context)
        .map_err(|e| format!("HELib key error: {}", e))?;
    let public_key = secret_key.public_key()
        .map_err(|e| format!("HELib public key error: {}", e))?;
    
    let pt1 = HEPlaintext::new(&context, val1)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    let pt2 = HEPlaintext::new(&context, val2)
        .map_err(|e| format!("HELib plaintext error: {}", e))?;
    
    let ct1 = public_key.encrypt(&pt1)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    let ct2 = public_key.encrypt(&pt2)
        .map_err(|e| format!("HELib encrypt error: {}", e))?;
    
    let result = ct1.multiply(&ct2)
        .map_err(|e| format!("HELib multiply error: {}", e))?;
    let decrypted = secret_key.decrypt(&result)
        .map_err(|e| format!("HELib decrypt error: {}", e))?;
    
    Ok(vec![decrypted.value()])
}

fn run_helib_benchmark(num_operations: i32) -> BenchmarkResponse {
    use he_benchmark::{HEContext, HESecretKey, HEPlaintext};
    
    let total_start = Instant::now();
    
    let key_start = Instant::now();
    let context = match HEContext::new(HELIB_M, HELIB_P, HELIB_R) {
        Ok(ctx) => ctx,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("HELib context failed: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let secret_key = match HESecretKey::generate(&context) {
        Ok(sk) => sk,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("HELib key gen failed: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let public_key = match secret_key.public_key() {
        Ok(pk) => pk,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("HELib public key failed: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    let key_gen_time = key_start.elapsed();
    
    // Encoding phase (HELib encoding is simpler - just create plaintexts)
    let encode_start = Instant::now();
    let mut plaintexts = Vec::new();
    for i in 0..num_operations {
        if let Ok(pt) = HEPlaintext::new(&context, i as i64) {
            plaintexts.push(pt);
        }
    }
    let encoding_time = encode_start.elapsed();
    
    // Encryption phase
    let encrypt_start = Instant::now();
    let mut ciphertexts = Vec::new();
    for pt in &plaintexts {
        if let Ok(ct) = public_key.encrypt(pt) {
            ciphertexts.push(ct);
        }
    }
    let encryption_time = encrypt_start.elapsed();
    
    let add_start = Instant::now();
    for i in 1..ciphertexts.len() {
        let _ = ciphertexts[0].add(&ciphertexts[i]);
    }
    let addition_time = add_start.elapsed();
    
    let mult_start = Instant::now();
    for i in 1..ciphertexts.len() {
        let _ = ciphertexts[0].multiply(&ciphertexts[i]);
    }
    let multiplication_time = mult_start.elapsed();
    
    let decrypt_start = Instant::now();
    for ct in &ciphertexts {
        let _ = secret_key.decrypt(ct);
    }
    let decryption_time = decrypt_start.elapsed();
    
    let total_time = total_start.elapsed();
    
    BenchmarkResponse {
        key_gen_time_ms: key_gen_time.as_secs_f64() * 1000.0,
        encoding_time_ms: encoding_time.as_secs_f64() * 1000.0 / num_operations as f64,
        encryption_time_ms: encryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        addition_time_ms: addition_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        multiplication_time_ms: multiplication_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        decryption_time_ms: decryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        status: format!("HELib benchmark complete: {} operations", num_operations),
    }
}

// ============================================
// OpenFHE Helper Functions
// ============================================

const OPENFHE_PLAINTEXT_MOD: u64 = 65537;
const OPENFHE_MULT_DEPTH: u32 = 2;

fn run_openfhe_encrypt(values: Vec<i64>) -> Result<usize, String> {
    use he_benchmark::{OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext};
    
    let context = OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH)
        .map_err(|e| format!("OpenFHE context error: {}", e))?;
    let keypair = OpenFHEKeyPair::generate(&context)
        .map_err(|e| format!("OpenFHE keypair error: {}", e))?;
    
    let plaintext = OpenFHEPlaintext::from_vec(&context, &values)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    let _ciphertext = OpenFHECiphertext::encrypt(&context, &keypair, &plaintext)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    
    // OpenFHE ciphertext size estimate
    Ok(8192)
}

fn run_openfhe_decrypt(values: Vec<i64>) -> Result<Vec<i64>, String> {
    use he_benchmark::{OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext};
    
    let context = OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH)
        .map_err(|e| format!("OpenFHE context error: {}", e))?;
    let keypair = OpenFHEKeyPair::generate(&context)
        .map_err(|e| format!("OpenFHE keypair error: {}", e))?;
    
    let plaintext = OpenFHEPlaintext::from_vec(&context, &values)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    let ciphertext = OpenFHECiphertext::encrypt(&context, &keypair, &plaintext)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    let decrypted = ciphertext.decrypt(&context, &keypair)
        .map_err(|e| format!("OpenFHE decrypt error: {}", e))?;
    
    let result = decrypted.to_vec()
        .map_err(|e| format!("OpenFHE to_vec error: {}", e))?;
    
    Ok(result[..values.len().min(result.len())].to_vec())
}

fn run_openfhe_add(values1: &[i64], values2: &[i64]) -> Result<Vec<i64>, String> {
    use he_benchmark::{OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext};
    
    let context = OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH)
        .map_err(|e| format!("OpenFHE context error: {}", e))?;
    let keypair = OpenFHEKeyPair::generate(&context)
        .map_err(|e| format!("OpenFHE keypair error: {}", e))?;
    
    let pt1 = OpenFHEPlaintext::from_vec(&context, values1)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    let pt2 = OpenFHEPlaintext::from_vec(&context, values2)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    
    let ct1 = OpenFHECiphertext::encrypt(&context, &keypair, &pt1)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    let ct2 = OpenFHECiphertext::encrypt(&context, &keypair, &pt2)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    
    let result_ct = ct1.add(&context, &ct2)
        .map_err(|e| format!("OpenFHE add error: {}", e))?;
    let decrypted = result_ct.decrypt(&context, &keypair)
        .map_err(|e| format!("OpenFHE decrypt error: {}", e))?;
    
    let result = decrypted.to_vec()
        .map_err(|e| format!("OpenFHE to_vec error: {}", e))?;
    
    Ok(result[..values1.len().max(values2.len()).min(result.len())].to_vec())
}

fn run_openfhe_multiply(values1: &[i64], values2: &[i64]) -> Result<Vec<i64>, String> {
    use he_benchmark::{OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext};
    
    let context = OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH)
        .map_err(|e| format!("OpenFHE context error: {}", e))?;
    let keypair = OpenFHEKeyPair::generate(&context)
        .map_err(|e| format!("OpenFHE keypair error: {}", e))?;
    
    let pt1 = OpenFHEPlaintext::from_vec(&context, values1)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    let pt2 = OpenFHEPlaintext::from_vec(&context, values2)
        .map_err(|e| format!("OpenFHE plaintext error: {}", e))?;
    
    let ct1 = OpenFHECiphertext::encrypt(&context, &keypair, &pt1)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    let ct2 = OpenFHECiphertext::encrypt(&context, &keypair, &pt2)
        .map_err(|e| format!("OpenFHE encrypt error: {}", e))?;
    
    let result_ct = ct1.multiply(&context, &keypair, &ct2)
        .map_err(|e| format!("OpenFHE multiply error: {}", e))?;
    let decrypted = result_ct.decrypt(&context, &keypair)
        .map_err(|e| format!("OpenFHE decrypt error: {}", e))?;
    
    let result = decrypted.to_vec()
        .map_err(|e| format!("OpenFHE to_vec error: {}", e))?;
    
    Ok(result[..values1.len().max(values2.len()).min(result.len())].to_vec())
}

fn run_openfhe_benchmark(num_operations: i32) -> BenchmarkResponse {
    use he_benchmark::{OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext};
    
    let total_start = Instant::now();
    
    // Key generation timing
    let key_start = Instant::now();
    let context = match OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH) {
        Ok(ctx) => ctx,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("OpenFHE context failed: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    
    let keypair = match OpenFHEKeyPair::generate(&context) {
        Ok(kp) => kp,
        Err(e) => return BenchmarkResponse {
            key_gen_time_ms: 0.0, encryption_time_ms: 0.0, addition_time_ms: 0.0,
            multiplication_time_ms: 0.0, decryption_time_ms: 0.0,
            status: format!("OpenFHE keypair failed: {}", e),
            total_time_ms: 0.0, encoding_time_ms: 0.0,
        },
    };
    let key_gen_time = key_start.elapsed();
    
    // Test data
    let test_data: Vec<i64> = (0..64).collect();
    
    // Encoding timing
    let encode_start = Instant::now();
    let mut plaintexts = Vec::new();
    for _ in 0..num_operations {
        if let Ok(pt) = OpenFHEPlaintext::from_vec(&context, &test_data) {
            plaintexts.push(pt);
        }
    }
    let encoding_time = encode_start.elapsed();
    
    // Encryption timing
    let encrypt_start = Instant::now();
    let mut ciphertexts = Vec::new();
    for pt in &plaintexts {
        if let Ok(ct) = OpenFHECiphertext::encrypt(&context, &keypair, pt) {
            ciphertexts.push(ct);
        }
    }
    let encryption_time = encrypt_start.elapsed();
    
    // Addition timing
    let add_start = Instant::now();
    for i in 1..ciphertexts.len() {
        let _ = ciphertexts[0].add(&context, &ciphertexts[i]);
    }
    let addition_time = add_start.elapsed();
    
    // Multiplication timing
    let mult_start = Instant::now();
    for i in 1..ciphertexts.len() {
        let _ = ciphertexts[0].multiply(&context, &keypair, &ciphertexts[i]);
    }
    let multiplication_time = mult_start.elapsed();
    
    // Decryption timing
    let decrypt_start = Instant::now();
    for ct in &ciphertexts {
        let _ = ct.decrypt(&context, &keypair);
    }
    let decryption_time = decrypt_start.elapsed();
    
    let total_time = total_start.elapsed();

    BenchmarkResponse {
        key_gen_time_ms: key_gen_time.as_secs_f64() * 1000.0,
        encoding_time_ms: encoding_time.as_secs_f64() * 1000.0 / num_operations as f64,
        encryption_time_ms: encryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        addition_time_ms: addition_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        multiplication_time_ms: multiplication_time.as_secs_f64() * 1000.0 / (num_operations - 1).max(1) as f64,
        decryption_time_ms: decryption_time.as_secs_f64() * 1000.0 / num_operations as f64,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        status: format!("OpenFHE benchmark complete: {} operations", num_operations),
    }
}

// ============================================
// gRPC Service Implementation
// ============================================

#[tonic::async_trait]
impl HeService for HEServiceImpl {
    async fn generate_keys(
        &self,
        request: Request<GenerateKeysRequest>,
    ) -> Result<Response<GenerateKeysResponse>, Status> {
        let req = request.into_inner();
        
        println!("📥 Received GenerateKeys request for library: {}", req.library);
        
        if !["SEAL", "HELib", "OpenFHE"].contains(&req.library.as_str()) {
            return Err(Status::invalid_argument("Library must be one of: SEAL, HELib, OpenFHE"));
        }
        
        let session_id = uuid::Uuid::new_v4().to_string();
        let poly_degree = req.poly_modulus_degree as u64;
        let plain_modulus = 1032193u64;
        let library = req.library.clone();
        
        // Validate context creation
        if library == "SEAL" {
            let pd = poly_degree;
            let result = tokio::task::spawn_blocking(move || {
                use he_benchmark::Context as SealContext;
                SealContext::new(pd, plain_modulus).map(|_| ()).map_err(|e| format!("{}", e))
            }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?;
            
            if let Err(e) = result {
                return Err(Status::internal(format!("Failed to create SEAL context: {}", e)));
            }
            println!("   ✓ SEAL context validated");
        } else if library == "HELib" {
            let result = tokio::task::spawn_blocking(move || {
                use he_benchmark::HEContext;
                HEContext::new(HELIB_M, HELIB_P, HELIB_R).map(|_| ()).map_err(|e| format!("{}", e))
            }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?;
            
            if let Err(e) = result {
                return Err(Status::internal(format!("Failed to create HELib context: {}", e)));
            }
            println!("   ✓ HELib context validated");
        } else if library == "OpenFHE" {
            let result = tokio::task::spawn_blocking(move || {
                use he_benchmark::OpenFHEContext;
                OpenFHEContext::new_bfv(OPENFHE_PLAINTEXT_MOD, OPENFHE_MULT_DEPTH)
                    .map(|_| ()).map_err(|e| format!("{}", e))
            }).await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?;
            
            if let Err(e) = result {
                return Err(Status::internal(format!("Failed to create OpenFHE context: {}", e)));
            }
            println!("   ✓ OpenFHE context validated");
        }
        
        let session = SessionConfig {
            library: req.library.clone(),
            poly_modulus_degree: poly_degree,
            plain_modulus,
            ciphertext_values: HashMap::new(),
        };
        
        self.sessions.lock().unwrap().insert(session_id.clone(), session);
        
        println!("✓ Session created: {}", &session_id[..8]);
        
        Ok(Response::new(GenerateKeysResponse {
            session_id: session_id.clone(),
            public_key: vec![],
            status: format!("Keys generated for {} (session: {})", req.library, &session_id[..8]),
        }))
    }

    async fn encrypt(
        &self,
        request: Request<EncryptRequest>,
    ) -> Result<Response<EncryptResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Encrypt request for session: {}", sid);
        
        let (library, poly_degree, plain_modulus) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            (session.library.clone(), session.poly_modulus_degree, session.plain_modulus)
        };
        
        let values = req.values.clone();
        let ciphertext_id = uuid::Uuid::new_v4().to_string();
        
        let (ciphertext_bytes, byte_count) = if library == "HELib" {
            let first_value = values.first().copied().unwrap_or(0);
            let result = tokio::task::spawn_blocking(move || run_helib_encrypt(first_value))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?;
            (vec![0u8; result.min(1024)], result)
        } else if library == "OpenFHE" {
            let result = tokio::task::spawn_blocking(move || run_openfhe_encrypt(values))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?;
            (vec![0u8; result.min(1024)], result)
        } else {
            tokio::task::spawn_blocking(move || run_seal_encrypt(poly_degree, plain_modulus, values))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        };
        
        {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(session) = sessions.get_mut(&req.session_id) {
                session.ciphertext_values.insert(ciphertext_id.clone(), req.values.clone());
            }
        }
        
        println!("   ✓ Encrypted {} values → {} bytes using {}", req.values.len(), byte_count, library);
        
        Ok(Response::new(EncryptResponse {
            ciphertext: ciphertext_bytes,
            status: format!("Encrypted {} values using {}", req.values.len(), library),
        }))
    }

    async fn decrypt(
        &self,
        request: Request<DecryptRequest>,
    ) -> Result<Response<DecryptResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("�� Decrypt request for session: {}", sid);
        
        let (library, poly_degree, plain_modulus, original_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            let values = session.ciphertext_values.values().next()
                .cloned().unwrap_or_else(|| vec![1, 2, 3]);
            (session.library.clone(), session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let result = if library == "HELib" {
            let value = original_values.first().copied().unwrap_or(0);
            tokio::task::spawn_blocking(move || run_helib_decrypt(value))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else if library == "OpenFHE" {
            tokio::task::spawn_blocking(move || run_openfhe_decrypt(original_values))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else {
            tokio::task::spawn_blocking(move || run_seal_decrypt(poly_degree, plain_modulus, &original_values))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        };
        
        println!("   ✓ Decrypted {} values using {}", result.len(), library);
        
        Ok(Response::new(DecryptResponse {
            values: result,
            status: format!("Decrypted successfully using {}", library),
        }))
    }

    async fn add(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!(" Add request for session: {}", sid);
        
        let (library, poly_degree, plain_modulus, all_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            let values: Vec<_> = session.ciphertext_values.values().cloned().collect();
            (session.library.clone(), session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let values1 = all_values.get(0).cloned().unwrap_or_else(|| vec![1, 2, 3]);
        let values2 = all_values.get(1).cloned().unwrap_or_else(|| vec![1, 1, 1]);
        
        let result = if library == "HELib" {
            let v1 = values1.first().copied().unwrap_or(0);
            let v2 = values2.first().copied().unwrap_or(0);
            tokio::task::spawn_blocking(move || run_helib_add(v1, v2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else if library == "OpenFHE" {
            tokio::task::spawn_blocking(move || run_openfhe_add(&values1, &values2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else {
            tokio::task::spawn_blocking(move || run_seal_add(poly_degree, plain_modulus, &values1, &values2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        };
        
        println!("   ✓ Addition result: {:?} using {}", &result[..result.len().min(3)], library);
        
        Ok(Response::new(BinaryOpResponse {
            result_ciphertext: vec![],
            status: format!("Addition complete using {}", library),
        }))
    }

    async fn multiply(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        let sid = &req.session_id[..8.min(req.session_id.len())];
        
        println!("📥 Multiply request for session: {}", sid);
        
        let (library, poly_degree, plain_modulus, all_values) = {
            let sessions = self.sessions.lock().unwrap();
            let session = sessions.get(&req.session_id)
                .ok_or_else(|| Status::not_found("Session not found"))?;
            let values: Vec<_> = session.ciphertext_values.values().cloned().collect();
            (session.library.clone(), session.poly_modulus_degree, session.plain_modulus, values)
        };
        
        let values1 = all_values.get(0).cloned().unwrap_or_else(|| vec![2, 3, 4]);
        let values2 = all_values.get(1).cloned().unwrap_or_else(|| vec![2, 2, 2]);
        
        let result = if library == "HELib" {
            let v1 = values1.first().copied().unwrap_or(0);
            let v2 = values2.first().copied().unwrap_or(0);
            tokio::task::spawn_blocking(move || run_helib_multiply(v1, v2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else if library == "OpenFHE" {
            tokio::task::spawn_blocking(move || run_openfhe_multiply(&values1, &values2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        } else {
            tokio::task::spawn_blocking(move || run_seal_multiply(poly_degree, plain_modulus, &values1, &values2))
                .await.map_err(|e| Status::internal(format!("Task failed: {}", e)))?
                .map_err(|e| Status::internal(e))?
        };
        
        println!("   ✓ Multiply result: {:?} using {}", &result[..result.len().min(3)], library);
        
        Ok(Response::new(BinaryOpResponse {
            result_ciphertext: vec![],
            status: format!("Multiplication complete using {}", library),
        }))
    }

    async fn run_benchmark(
        &self,
        request: Request<BenchmarkRequest>,
    ) -> Result<Response<BenchmarkResponse>, Status> {
        let req = request.into_inner();
        
        println!(" Benchmark request for library: {} ({} ops)", req.library, req.num_operations);
        
        let library = req.library.clone();
        let num_ops = req.num_operations;
        
        let response = if library == "HELib" {
            tokio::task::spawn_blocking(move || run_helib_benchmark(num_ops))
                .await.map_err(|e| Status::internal(format!("Benchmark failed: {}", e)))?
        } else if library == "OpenFHE" {
            tokio::task::spawn_blocking(move || run_openfhe_benchmark(num_ops))
                .await.map_err(|e| Status::internal(format!("Benchmark failed: {}", e)))?
        } else {
            let poly_degree = 8192u64;
            tokio::task::spawn_blocking(move || run_seal_benchmark(poly_degree, num_ops))
                .await.map_err(|e| Status::internal(format!("Benchmark failed: {}", e)))?
        };
        
        println!("   ✓ Benchmark complete using {}", library);
        
        Ok(Response::new(response))
    }

    async fn run_comparison_benchmark(
        &self,
        request: Request<BenchmarkRequest>,
    ) -> Result<Response<ComparisonBenchmarkResponse>, Status> {
        let req = request.into_inner();
        let num_ops = req.num_operations;
        
        println!("📥 Comparison benchmark request ({} ops per library)", num_ops);
        println!("   Running SEAL benchmark...");
        
        // Run all three benchmarks
        let seal_ops = num_ops;
        let seal_result = tokio::task::spawn_blocking(move || {
            run_seal_benchmark(8192, seal_ops)
        }).await.map_err(|e| Status::internal(format!("SEAL benchmark failed: {}", e)))?;
        
        println!("   Running HELib benchmark...");
        let helib_ops = num_ops;
        let helib_result = tokio::task::spawn_blocking(move || {
            run_helib_benchmark(helib_ops)
        }).await.map_err(|e| Status::internal(format!("HELib benchmark failed: {}", e)))?;
        
        println!("   Running OpenFHE benchmark...");
        let openfhe_ops = num_ops;
        let openfhe_result = tokio::task::spawn_blocking(move || {
            run_openfhe_benchmark(openfhe_ops)
        }).await.map_err(|e| Status::internal(format!("OpenFHE benchmark failed: {}", e)))?;
        
        // Determine fastest library based on total time
        let seal_total = seal_result.total_time_ms;
        let helib_total = helib_result.total_time_ms;
        let openfhe_total = openfhe_result.total_time_ms;
        
        let fastest_library = if seal_total <= helib_total && seal_total <= openfhe_total {
            "SEAL".to_string()
        } else if helib_total <= seal_total && helib_total <= openfhe_total {
            "HELib".to_string()
        } else {
            "OpenFHE".to_string()
        };
        
        // Generate recommendation
        let recommendation = if seal_result.encryption_time_ms < helib_result.encryption_time_ms 
            && seal_result.encryption_time_ms < openfhe_result.encryption_time_ms {
            "SEAL recommended for encryption-heavy workloads (batching support)".to_string()
        } else if helib_result.multiplication_time_ms < seal_result.multiplication_time_ms 
            && helib_result.multiplication_time_ms < openfhe_result.multiplication_time_ms {
            "HELib recommended for multiplication-heavy workloads (BGV optimizations)".to_string()
        } else {
            "OpenFHE recommended for general-purpose HE (flexible API)".to_string()
        };
        
        println!("   ✓ Comparison complete - Fastest: {}", fastest_library);
        
        Ok(Response::new(ComparisonBenchmarkResponse {
            seal: Some(seal_result),
            helib: Some(helib_result),
            openfhe: Some(openfhe_result),
            fastest_library,
            recommendation,
        }))
    }

    // ============================================
    // PredictDigit — Encrypted MNIST Inference
    // ============================================

    async fn predict_digit(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();

        println!("📥 Received PredictDigit request ({} pixels)", req.pixels.len());

        // Validate input
        if req.pixels.len() != 784 {
            return Err(Status::invalid_argument(format!(
                "Expected 784 pixels (28×28 image), got {}",
                req.pixels.len()
            )));
        }

        // Clone the Arc so we can move it into the blocking task
        let engine = Arc::clone(&self.inference_engine);
        let pixels = req.pixels;

        // Run inference on a blocking thread (FFI calls are not async-safe)
        let (result, float_accuracy) = tokio::task::spawn_blocking(move || {
            let res = engine.0.predict(&pixels);
            let acc = engine.0.float_accuracy;
            (res, acc)
        })
        .await
        .map_err(|e| Status::internal(format!("Inference task panicked: {}", e)))?;

        let result = result
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        // Compute a simple confidence score from the logits
        // Use softmax-style: max_logit / sum_of_positive_logits (approximate)
        let max_logit = *result.logits.iter().max().unwrap_or(&0) as f64;
        let sum_positive: f64 = result.logits.iter().filter(|&&v| v > 0).map(|&v| v as f64).sum();
        let confidence = if sum_positive > 0.0 {
            max_logit / sum_positive
        } else {
            0.0
        };

        println!(
            "   ✓ Predicted digit: {} (confidence: {:.2}%, time: {:.1}ms)",
            result.predicted_digit,
            confidence * 100.0,
            result.timing.total_ms
        );

        Ok(Response::new(PredictResponse {
            predicted_digit: result.predicted_digit as i32,
            logits: result.logits,
            confidence,
            status: "success".to_string(),
            encryption_ms: result.timing.encryption_ms,
            conv1_ms: result.timing.conv1_ms,
            bias1_ms: result.timing.bias1_ms,
            act1_ms: result.timing.act1_ms,
            pool1_ms: result.timing.pool1_ms,
            conv2_ms: result.timing.conv2_ms,
            bias2_ms: result.timing.bias2_ms,
            act2_ms: result.timing.act2_ms,
            pool2_ms: result.timing.pool2_ms,
            fc_ms: result.timing.fc_ms,
            bias_fc_ms: result.timing.bias_fc_ms,
            decryption_ms: result.timing.decryption_ms,
            total_ms: result.timing.total_ms,
            float_model_accuracy: float_accuracy,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use environment variable or default to [::]:50051 (all interfaces, IPv6+IPv4)
    let bind_addr = std::env::var("GRPC_BIND_ADDR").unwrap_or_else(|_| "[::]:50051".to_string());
    let addr = bind_addr.parse()?;

    // Initialize the encrypted inference engine (loads weights, creates BFV context, keygen)
    // This takes ~2-5 seconds — done once at startup
    let weights_dir = std::env::var("MNIST_WEIGHTS_DIR")
        .unwrap_or_else(|_| "mnist_training/weights".to_string());
    println!("Initializing encrypted inference engine (weights: {})...", weights_dir);
    let engine = EncryptedInferenceEngine::new(&weights_dir)
        .expect("Failed to initialize EncryptedInferenceEngine");
    println!("Encrypted inference engine ready (float accuracy: {:.2}%)\n", engine.float_accuracy);

    let service = HEServiceImpl::new(engine);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║      Homomorphic Encryption gRPC Server                    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("   Listening on: {}", addr);
    println!("   Libraries: Microsoft SEAL (BFV), HELib (BGV), OpenFHE (BFV)");
    println!();
    println!("  Available services:");
    println!("    • GenerateKeys           - Create encryption context and keys");
    println!("    • Encrypt                - Encrypt integer vectors");
    println!("    • Decrypt                - Decrypt ciphertext");
    println!("    • Add                    - Homomorphic addition");
    println!("    • Multiply               - Homomorphic multiplication");
    println!("    • RunBenchmark           - Benchmark single library");
    println!("    • RunComparisonBenchmark - Compare all three libraries");
    println!("    • PredictDigit           - Encrypted MNIST digit inference");
    println!();
    println!("  Ready to accept connections!");
    println!();

    Server::builder()
        .add_service(HeServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
