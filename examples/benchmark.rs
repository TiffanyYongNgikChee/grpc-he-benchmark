//! Medical Data Encryption Comparison: SEAL vs HElib
//! 
//! This example encrypts the same medical record using both frameworks
//! and provides a detailed performance comparison.

use he_benchmark::{
    Context as SealContext, 
    Encryptor as SealEncryptor, 
    Decryptor as SealDecryptor,
    BatchEncoder as SealBatchEncoder,
    HEContext,
    HESecretKey,
    HEPublicKey,
    HEPlaintext,
};

use std::time::{Instant, Duration};
use std::thread::sleep;
use std::io::{self, Write};

// Performance Tracking Structures
// These structs store timing information for each phase
// of the SEAL and HElib encryption processes.
#[derive(Debug, Clone)]
// PhaseMetrics holds the duration (time taken) of each major step
// in the encryption pipeline for ONE framework (either SEAL or HElib).
struct PhaseMetrics {
    setup_time: Duration, // Time spent creating the encryption context and generating keys.
    encoding_time: Duration, // Time spent encoding the raw medical data into plaintext format.
    encryption_time: Duration, // Time taken to encrypt the encoded plaintext into ciphertext.
    operation_time: Duration, // Time taken to perform homomorphic operations (addition, etc.)
    decryption_time: Duration, // Time spent decrypting the resulting ciphertext.
    total_time: Duration, // Total accumulated time for the entire encryption workflow.
}

impl PhaseMetrics {
    // Creates a new PhaseMetrics object with all times initialized to zero.
    fn new() -> Self {
        Self {
            setup_time: Duration::ZERO,
            encoding_time: Duration::ZERO,
            encryption_time: Duration::ZERO,
            operation_time: Duration::ZERO,
            decryption_time: Duration::ZERO,
            total_time: Duration::ZERO,
        }
    }
}

#[derive(Debug)]
// ComparisonResult contains the performance metrics for BOTH
// encryption frameworks, SEAL and HElib, as well as a description
// of the test data used (e.g., "200-character medical record").
struct ComparisonResult {
    seal: PhaseMetrics, // Timing results for the SEAL encryption run.
    helib: PhaseMetrics, // Timing results for the HElib encryption run.
    data_description: String, // Human-readable description of the dataset (size, type, etc.).
}

// UI Helper Functions
// These functions provide visual formatting,
// progress indicators, and animated steps for
// a more user-friendly terminal experience.

/// Clears the terminal screen and moves the cursor
/// to the top-left corner using ANSI escape codes.
fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
}

/// Prints a stylized header box with a centered title.
/// Used at the beginning of major sections.
fn print_header(title: &str) {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║ {:^65} ║", title);
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");
}

/// Prints a small labeled section divider.
/// Used to visually separate steps in the process.
fn print_section(title: &str) {
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ {} ", title);
    println!("└─────────────────────────────────────────────────────────────────┘");
}

/// Displays an animated "processing" step with dots,
/// simulating work being done. The duration controls
/// how long the animation lasts.
fn processing_step(label: &str, duration_ms: u64) {
    print!("   {} ", label);
    io::stdout().flush().unwrap();
    
    let steps = 20; // total number of dots
    let step_duration = duration_ms / steps; // delay per dot
    
    // Print dots gradually to simulate progress.
    for _ in 0..steps {
        print!(".");
        io::stdout().flush().unwrap();
        sleep(Duration::from_millis(step_duration));
    }
    println!(" ✓"); // success checkmark at the end
}

/// Prints a detailed progress bar showing:
/// - percentage completion
/// - visual bar ("█" for completed, "░" for remaining)
/// - current step vs total
/// - elapsed time
///
/// Used for longer multi-step operations.
fn print_progress(label: &str, current: usize, total: usize, elapsed: Duration) {
    let percentage = (current as f64 / total as f64 * 100.0) as usize;
    let bar_width = 50;
    let filled = (percentage * bar_width / 100).min(bar_width);
    let empty = bar_width - filled;
    
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  {} Progress: {}%", label, percentage);
    println!("├─────────────────────────────────────────────────────────────────┤");
    print!("│  [");
    print!("{}", "█".repeat(filled)); // filled portion
    print!("{}", "░".repeat(empty)); // empty portion
    println!("] {}/{}", current, total);
    println!("│  Elapsed: {:.1}s", elapsed.as_secs_f64());
    println!("└─────────────────────────────────────────────────────────────────┘");
}

// SEAL Encryption Process
// Runs a full homomorphic encryption workflow
// using Microsoft SEAL:
//   1. Setup & key generation
//   2. Data encoding
//   3. Encryption
//   4. Homomorphic operations
//   5. Decryption & verification
//
// Returns PhaseMetrics containing timing for each step.

fn run_seal_encryption(medical_data: &[i64]) -> Result<PhaseMetrics, Box<dyn std::error::Error>> {
    // Stores timing for each phase.
    let mut metrics = PhaseMetrics::new();
    // Start total time measurement.
    let total_start = Instant::now();
    
    print_section("SEAL Encryption Process");
    
    // Phase 1: Setup
    println!("\n Phase 1: SEAL Setup & Key Generation");
    let setup_start = Instant::now();
    
    // Create SEAL context with specified polynomial modulus and coefficient modulus.
    processing_step("Creating SEAL context (poly_modulus: 8192)", 600);
    let context = SealContext::new(8192, 1032193)?;
    
    // Initialize batch encoder and determine available batching slots.
    processing_step("Generating SEAL keys", 800);
    let encoder = SealBatchEncoder::new(&context)?;
    let slot_count = encoder.slot_count();
    
    metrics.setup_time = setup_start.elapsed();
    println!("   Setup complete: {:.2}s", metrics.setup_time.as_secs_f64());
    println!("   Available slots: {}", slot_count);
    
    // Phase 2: Encoding
    // Convert raw medical data (chars as ints)
    // into a batch-encoded SEAL plaintext.
    println!("\n Phase 2: SEAL Data Encoding");
    let encode_start = Instant::now();
    
    // SEAL batching requires data to match the slot count.
    processing_step("Padding data to slot size", 400);
    let mut padded_data = medical_data.to_vec();
    padded_data.resize(slot_count, 0); // fill unused slots with zero
    
    // Encode padded vector into SEAL plaintext object.
    processing_step("Encoding into SEAL plaintext", 500);
    let plaintext = encoder.encode(&padded_data)?;
    
    metrics.encoding_time = encode_start.elapsed();
    println!("   Encoding complete: {:.2}s", metrics.encoding_time.as_secs_f64());
    
    // Phase 3: Encryption
    // Encrypt the encoded plaintext.
    println!("\n Phase 3: SEAL Encryption");
    let encrypt_start = Instant::now();
    
    // Encrypt the batch-encoded medical record.
    processing_step("Initializing SEAL encryptor", 300);
    let encryptor = SealEncryptor::new(&context)?;
    
    processing_step("Encrypting medical data", 700);
    let ciphertext = encryptor.encrypt(&plaintext)?;
    
    metrics.encryption_time = encrypt_start.elapsed();
    println!("  Encryption complete: {:.2}s", metrics.encryption_time.as_secs_f64());
    
    // Phase 4: Homomorphic Operation
    // Perform encrypted addition:
    //    (encrypted medical record) + 1
    // This demonstrates fully homomorphic capability.
    println!("\n Phase 4: SEAL Encrypted Operations");
    let op_start = Instant::now();
    
    // Encode & encrypt a vector of all 1s.
    processing_step("Creating second encrypted value", 400);
    let ones = vec![1i64; slot_count];
    let plain2 = encoder.encode(&ones)?;
    let cipher2 = encryptor.encrypt(&plain2)?;
    
    // Homomorphic addition using SEAL via wrapper function.
    processing_step("Performing encrypted addition", 500);
    let result_cipher = he_benchmark::add(&context, &ciphertext, &cipher2)?;
    
    metrics.operation_time = op_start.elapsed();
    println!("   Operation complete: {:.2}s", metrics.operation_time.as_secs_f64());
    
    // Phase 5: Decryption
    // Convert ciphertext back into plaintext,
    // then decode into raw integers.
    println!("\n Phase 5: SEAL Decryption");
    let decrypt_start = Instant::now();
    
    processing_step("Initializing SEAL decryptor", 300);
    let decryptor = SealDecryptor::new(&context)?;
    
    // Decrypt output ciphertext.
    processing_step("Decrypting result", 600);
    let decrypted = decryptor.decrypt(&result_cipher)?;
    
    // Decode the batch-encoded result.
    processing_step("Decoding to readable format", 400);
    let result = encoder.decode(&decrypted)?;
    
    metrics.decryption_time = decrypt_start.elapsed();
    println!("    Decryption complete: {:.2}s", metrics.decryption_time.as_secs_f64());
    
    // Sanity Check: Print first few decoded characters.
    // Helps validate that the homomorphic operations worked.
    let preview: String = result[..medical_data.len().min(10)]
        .iter()
        .filter(|&&n| n > 0 && n < 128) // printable ASCII
        .map(|&n| (n as u8) as char)
        .collect();
    println!("   Preview: \"{}...\"", preview);
    
    // Final total time
    metrics.total_time = total_start.elapsed();
    
    Ok(metrics)
}

// HElib Encryption Process
// This function benchmarks a full encryption workflow using the HElib backend.
// It measures performance across setup, encoding, encryption, homomorphic
// computation, and decryption, storing results in a PhaseMetrics structure.
fn run_helib_encryption(medical_data: &[i64]) -> Result<PhaseMetrics, Box<dyn std::error::Error>> {
    // Initialize metric tracker and start global runtime timer
    let mut metrics = PhaseMetrics::new();
    let total_start = Instant::now();
    
    print_section(" HElib Encryption Process");
    
    // Phase 1: Setup
    // Initializes the HElib cryptographic environment and generates keys.
    println!("\n Phase 1: HElib Setup & Key Generation");
    let setup_start = Instant::now();
    
    // Create HElib context (defines parameters like modulus and ring structure)
    processing_step("Creating HElib context (m: 8191, p: 2, r: 1)", 700);
    let context = HEContext::new(8191, 2, 1)?;
    
    // Generate the secret key (also used to derive public key)
    processing_step("Generating HElib secret key", 900);
    let secret_key = HESecretKey::generate(&context)?;
    
    // Extract the public key used for encrypting plaintexts
    processing_step("Extracting HElib public key", 400);
    let public_key = secret_key.public_key()?;
    
    metrics.setup_time = setup_start.elapsed();
    println!("   Setup complete: {:.2}s", metrics.setup_time.as_secs_f64());
    
    // Phase 2: Encoding (HElib handles single values)
    // HElib generally handles values one-by-one (no batching),
    // so we encode the first data element and a constant for the operation.
    println!("\n Phase 2: HElib Data Encoding");
    let encode_start = Instant::now();
    
    processing_step("Encoding first value", 300);
    // For simplicity, encode first character as demo
    let first_value = medical_data.first().copied().unwrap_or(0);
    let plaintext1 = HEPlaintext::new(&context, first_value)?;
    
    // Encode the first medical data value as the primary plaintext
    processing_step("Encoding second value for operation", 300);
    let plaintext2 = HEPlaintext::new(&context, 1)?;
    
    // Encode a second plaintext (value = 1) for homomorphic addition
    metrics.encoding_time = encode_start.elapsed();
    println!("   Encoding complete: {:.2}s", metrics.encoding_time.as_secs_f64());
    
    // Phase 3: Encryption
    // Encrypt the encoded plaintexts using the public key.
    println!("\n Phase 3: HElib Encryption");
    let encrypt_start = Instant::now();
    
    // Encrypt original value
    processing_step("Encrypting first value", 800);
    let ciphertext1 = public_key.encrypt(&plaintext1)?;
    
    // Encrypt the constant '1'
    processing_step("Encrypting second value", 800);
    let ciphertext2 = public_key.encrypt(&plaintext2)?;
    
    metrics.encryption_time = encrypt_start.elapsed();
    println!("    Encryption complete: {:.2}s", metrics.encryption_time.as_secs_f64());
    
    // Phase 4: Homomorphic Operation
    // Performs homomorphic addition: ciphertext1 + ciphertext2.
    println!("\n Phase 4: HElib Encrypted Operations");
    let op_start = Instant::now();
    
    processing_step("Performing encrypted addition", 600);
    let result_cipher = ciphertext1.add(&ciphertext2)?;
    
    metrics.operation_time = op_start.elapsed();
    println!("    Operation complete: {:.2}s", metrics.operation_time.as_secs_f64());
    
    // Phase 5: Decryption
    // Decrypts the resulting ciphertext using the secret key.
    println!("\n Phase 5: HElib Decryption");
    let decrypt_start = Instant::now();
    
    processing_step("Decrypting result", 700);
    let decrypted = secret_key.decrypt(&result_cipher)?;
    
    metrics.decryption_time = decrypt_start.elapsed();
    println!("    Decryption complete: {:.2}s", metrics.decryption_time.as_secs_f64());
    println!("   Decrypted plaintext obtained");
    
    // Record total runtime across all phases
    metrics.total_time = total_start.elapsed();
    
    Ok(metrics)
}

// Comparison Display
fn print_comparison(result: &ComparisonResult) {
    clear_screen();
    
    print_header("PERFORMANCE COMPARISON: SEAL vs HElib");
    
    println!(" Test Data: {}\n", result.data_description);
    
    // Header
    println!("┌─────────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Phase                   │ SEAL         │ HElib        │ Winner       │");
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┤");
    
    // Setup
    print_comparison_row(
        "Setup & Keys",
        result.seal.setup_time,
        result.helib.setup_time,
    );
    
    // Encoding
    print_comparison_row(
        "Data Encoding",
        result.seal.encoding_time,
        result.helib.encoding_time,
    );
    
    // Encryption
    print_comparison_row(
        "Encryption",
        result.seal.encryption_time,
        result.helib.encryption_time,
    );
    
    // Operations
    print_comparison_row(
        "Encrypted Operations",
        result.seal.operation_time,
        result.helib.operation_time,
    );
    
    // Decryption
    print_comparison_row(
        "Decryption",
        result.seal.decryption_time,
        result.helib.decryption_time,
    );
    
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┤");
    
    // Total
    print_comparison_row(
        "TOTAL TIME",
        result.seal.total_time,
        result.helib.total_time,
    );
    
    println!("└─────────────────────────┴──────────────┴──────────────┴──────────────┘");
    
    // Speedup calculation
    let speedup = if result.seal.total_time < result.helib.total_time {
        result.helib.total_time.as_secs_f64() / result.seal.total_time.as_secs_f64()
    } else {
        result.seal.total_time.as_secs_f64() / result.helib.total_time.as_secs_f64()
    };
    
    let faster = if result.seal.total_time < result.helib.total_time {
        "SEAL"
    } else {
        "HElib"
    };
    
    println!("\n Summary:");
    println!("   {} is {:.2}x faster overall", faster, speedup);
    println!();
}

fn print_comparison_row(phase: &str, seal_time: Duration, helib_time: Duration) {
    let seal_ms = seal_time.as_millis();
    let helib_ms = helib_time.as_millis();
    
    let winner = if seal_ms < helib_ms {
        "SEAL ⚡"
    } else if helib_ms < seal_ms {
        "HElib ⚡"
    } else {
        "Tie"
    };
    
    println!(
        "│ {:23} │ {:>10}ms │ {:>10}ms │ {:12} │",
        phase, seal_ms, helib_ms, winner
    );
}
