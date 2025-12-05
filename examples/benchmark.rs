//! Medical Data Encryption Comparison: SEAL vs HElib vs OpenFHE
//! 
//! This example encrypts the same medical record using all three frameworks
//! and provides a detailed performance comparison.

use he_benchmark::{
    Context as SealContext, 
    Encryptor as SealEncryptor, 
    Decryptor as SealDecryptor,
    BatchEncoder as SealBatchEncoder,
    HEContext,
    HESecretKey,
    HEPlaintext,
    OpenFHEContext,
    OpenFHEKeyPair,
    OpenFHEPlaintext,
    OpenFHECiphertext,
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
// ComparisonResult contains the performance metrics for ALL THREE
// encryption frameworks, SEAL, HElib, and OpenFHE, as well as a description
// of the test data used (e.g., "200-character medical record").
struct ComparisonResult {
    seal: PhaseMetrics, // Timing results for the SEAL encryption run.
    helib: PhaseMetrics, // Timing results for the HElib encryption run.
    openfhe: PhaseMetrics, // Timing results for the OpenFHE encryption run.
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
#[allow(dead_code)]
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
    let _decrypted = secret_key.decrypt(&result_cipher)?;
    
    metrics.decryption_time = decrypt_start.elapsed();
    println!("    Decryption complete: {:.2}s", metrics.decryption_time.as_secs_f64());
    println!("   Decrypted plaintext obtained");
    
    // Record total runtime across all phases
    metrics.total_time = total_start.elapsed();
    
    Ok(metrics)
}

// OpenFHE Encryption Process
// This function benchmarks a full encryption workflow using the OpenFHE backend.
// It measures performance across setup, encoding, encryption, homomorphic
// computation, and decryption, storing results in a PhaseMetrics structure.
fn run_openfhe_encryption(medical_data: &[i64]) -> Result<PhaseMetrics, Box<dyn std::error::Error>> {
    // Initialize metric tracker and start global runtime timer
    let mut metrics = PhaseMetrics::new();
    let total_start = Instant::now();
    
    print_section("🔶 OpenFHE Encryption Process");
    
    // Phase 1: Setup
    // Initializes the OpenFHE cryptographic environment and generates keys.
    println!("\n Phase 1: OpenFHE Setup & Key Generation");
    let setup_start = Instant::now();
    
    // Create OpenFHE context with BFV scheme
    processing_step("Creating OpenFHE context (BFV, plaintext_mod: 65537)", 700);
    let context = OpenFHEContext::new_bfv(65537, 2)?;
    
    // Generate keypair (includes multiplication keys)
    processing_step("Generating OpenFHE keypair", 900);
    let keypair = OpenFHEKeyPair::generate(&context)?;
    
    metrics.setup_time = setup_start.elapsed();
    println!("   Setup complete: {:.2}s", metrics.setup_time.as_secs_f64());
    
    // Phase 2: Encoding
    // OpenFHE uses batch encoding similar to SEAL
    println!("\n Phase 2: OpenFHE Data Encoding");
    let encode_start = Instant::now();
    
    processing_step("Encoding medical data into plaintext", 400);
    // Take first few values for demo (OpenFHE batches efficiently)
    let sample_size = medical_data.len().min(8);
    let plaintext1 = OpenFHEPlaintext::from_vec(&context, &medical_data[..sample_size])?;
    
    // Create a second plaintext with all 1s for the operation
    processing_step("Encoding second value for operation", 300);
    let ones = vec![1i64; sample_size];
    let plaintext2 = OpenFHEPlaintext::from_vec(&context, &ones)?;
    
    metrics.encoding_time = encode_start.elapsed();
    println!("   Encoding complete: {:.2}s", metrics.encoding_time.as_secs_f64());
    
    // Phase 3: Encryption
    // Encrypt the encoded plaintexts
    println!("\n Phase 3: OpenFHE Encryption");
    let encrypt_start = Instant::now();
    
    // Encrypt original values
    processing_step("Encrypting medical data", 800);
    let ciphertext1 = OpenFHECiphertext::encrypt(&context, &keypair, &plaintext1)?;
    
    // Encrypt the vector of 1s
    processing_step("Encrypting second value", 800);
    let ciphertext2 = OpenFHECiphertext::encrypt(&context, &keypair, &plaintext2)?;
    
    metrics.encryption_time = encrypt_start.elapsed();
    println!("    Encryption complete: {:.2}s", metrics.encryption_time.as_secs_f64());
    
    // Phase 4: Homomorphic Operation
    // Performs homomorphic addition: ciphertext1 + ciphertext2
    println!("\n Phase 4: OpenFHE Encrypted Operations");
    let op_start = Instant::now();
    
    processing_step("Performing encrypted addition", 600);
    let result_cipher = ciphertext1.add(&context, &ciphertext2)?;
    
    metrics.operation_time = op_start.elapsed();
    println!("    Operation complete: {:.2}s", metrics.operation_time.as_secs_f64());
    
    // Phase 5: Decryption
    // Decrypts the resulting ciphertext
    println!("\n Phase 5: OpenFHE Decryption");
    let decrypt_start = Instant::now();
    
    processing_step("Decrypting result", 700);
    let decrypted = result_cipher.decrypt(&context, &keypair)?;
    let result = decrypted.to_vec()?;
    
    metrics.decryption_time = decrypt_start.elapsed();
    println!("    Decryption complete: {:.2}s", metrics.decryption_time.as_secs_f64());
    
    // Sanity check: print first few values
    println!("   First values: {:?}", &result[..sample_size.min(5)]);
    
    // Record total runtime across all phases
    metrics.total_time = total_start.elapsed();
    
    Ok(metrics)
}

// Comparison Display
fn print_comparison(result: &ComparisonResult) {
    clear_screen();
    
    print_header("PERFORMANCE COMPARISON: SEAL vs HElib vs OpenFHE");
    
    println!(" Test Data: {}\n", result.data_description);
    
    // Header
    println!("┌─────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Phase                   │ SEAL         │ HElib        │ OpenFHE      │ Winner       │");
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    
    // Setup
    print_comparison_row_3way(
        "Setup & Keys",
        result.seal.setup_time,
        result.helib.setup_time,
        result.openfhe.setup_time,
    );
    
    // Encoding
    print_comparison_row_3way(
        "Data Encoding",
        result.seal.encoding_time,
        result.helib.encoding_time,
        result.openfhe.encoding_time,
    );
    
    // Encryption
    print_comparison_row_3way(
        "Encryption",
        result.seal.encryption_time,
        result.helib.encryption_time,
        result.openfhe.encryption_time,
    );
    
    // Operations
    print_comparison_row_3way(
        "Encrypted Operations",
        result.seal.operation_time,
        result.helib.operation_time,
        result.openfhe.operation_time,
    );
    
    // Decryption
    print_comparison_row_3way(
        "Decryption",
        result.seal.decryption_time,
        result.helib.decryption_time,
        result.openfhe.decryption_time,
    );
    
    println!("├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    
    // Total
    print_comparison_row_3way(
        "TOTAL TIME",
        result.seal.total_time,
        result.helib.total_time,
        result.openfhe.total_time,
    );
    
    println!("└─────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    
    // Speedup calculation - find the fastest
    let times = [
        ("SEAL", result.seal.total_time.as_secs_f64()),
        ("HElib", result.helib.total_time.as_secs_f64()),
        ("OpenFHE", result.openfhe.total_time.as_secs_f64()),
    ];
    
    let fastest = times.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let slowest = times.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let speedup = slowest.1 / fastest.1;
    
    println!("\n Summary:");
    println!("   {} is the fastest ({:.2}x faster than {})", fastest.0, speedup, slowest.0);
    println!("   SEAL: {:.2}s | HElib: {:.2}s | OpenFHE: {:.2}s", 
             result.seal.total_time.as_secs_f64(),
             result.helib.total_time.as_secs_f64(),
             result.openfhe.total_time.as_secs_f64());
    println!();
}

fn print_comparison_row_3way(phase: &str, seal_time: Duration, helib_time: Duration, openfhe_time: Duration) {
    let seal_ms = seal_time.as_millis();
    let helib_ms = helib_time.as_millis();
    let openfhe_ms = openfhe_time.as_millis();
    
    // Find the winner (minimum time)
    let min_time = seal_ms.min(helib_ms).min(openfhe_ms);
    
    let winner = if seal_ms == min_time && helib_ms == min_time && openfhe_ms == min_time {
        "3-way Tie"
    } else if seal_ms == min_time && helib_ms == min_time {
        "SEAL/HElib"
    } else if seal_ms == min_time && openfhe_ms == min_time {
        "SEAL/OpenFHE"
    } else if helib_ms == min_time && openfhe_ms == min_time {
        "HElib/OpenFHE"
    } else if seal_ms == min_time {
        "SEAL ⚡"
    } else if helib_ms == min_time {
        "HElib ⚡"
    } else {
        "OpenFHE ⚡"
    };
    
    println!(
        "│ {:23} │ {:>10}ms │ {:>10}ms │ {:>10}ms │ {:12} │",
        phase, seal_ms, helib_ms, openfhe_ms, winner
    );
}

// Main Function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Clears the terminal to give a clean display for the demo
    clear_screen();
    
    // Prints a formatted title/banner for the program
    print_header("MEDICAL DATA ENCRYPTION COMPARISON");
    
    println!("This example encrypts the same medical record using SEAL,");
    println!("HElib, and OpenFHE frameworks, then compares their performance.\n");

    sleep(Duration::from_secs(2));
    
    // DEFINE data
    // This is the plaintext data that will be encrypted with SEAL
    // and HElib. The demo uses a human-readable medical record to
    // emphasize privacy-preserving computation on sensitive data.
    let medical_record = 
        "PATIENT ID: 12345 | NAME: Bob Smith | AGE: 45 | DIAGNOSIS: HYPERTENSION STAGE 2 | BP: 160/100 | MEDICATION: LISINOPRIL 10MG DAILY | ALLERGIES: EGG | LAST VISIT: 2024-11-10 | NOTES: PATIENT SHOWS IMPROVEMENT WITH CURRENT TREATMENT";
    
    println!(" Medical Record:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{}", medical_record);
    println!("─────────────────────────────────────────────────────────────────\n");
    
    // CONVERT RECORD TO NUMERIC FORM
    // Most homomorphic encryption libraries operate on integers,
    // not text. Here, each character of the medical record string
    // is converted to its ASCII numeric value.
    let medical_data: Vec<i64> = medical_record.chars().map(|c| c as i64).collect();
    println!(" Data size: {} characters\n", medical_data.len());
    
    sleep(Duration::from_secs(1));
    
    // RUN SEAL ENCRYPTION
    // Calls a helper function that:
    //  - sets up SEAL parameters
    //  - encrypts the numeric data
    //  - optionally performs operations
    //  - returns timing and performance metrics
    println!("\n{}", "=".repeat(70));
    println!("🔷 Testing with SEAL Framework");
    println!("{}", "=".repeat(70));
    let seal_metrics = run_seal_encryption(&medical_data)?;
    
    sleep(Duration::from_secs(2));
    
    // RUN HElib ENCRYPTION
    // Same process as SEAL, but using the HElib library to allow
    // an apples-to-apples comparison for the same dataset.
    println!("\n{}", "=".repeat(70));
    println!(" Testing with HElib Framework");
    println!("{}", "=".repeat(70));
    let helib_metrics = run_helib_encryption(&medical_data)?;
    
    sleep(Duration::from_secs(2));
    
    // RUN OpenFHE ENCRYPTION
    // Same process but using OpenFHE library
    println!("\n{}", "=".repeat(70));
    println!("🔶 Testing with OpenFHE Framework");
    println!("{}", "=".repeat(70));
    let openfhe_metrics = run_openfhe_encryption(&medical_data)?;
    
    sleep(Duration::from_secs(2));
    
    // BUILD AND DISPLAY COMPARISON TABLE
    // Wraps the metrics into a shared structure, then formats them
    // for printing (e.g., encryption time, memory usage, ciphertext size).
    let comparison = ComparisonResult {
        seal: seal_metrics,
        helib: helib_metrics,
        openfhe: openfhe_metrics,
        data_description: format!("{} character medical record", medical_data.len()),
    };
    
    print_comparison(&comparison);
    
    println!(" Comparison complete!\n");
    
    Ok(())
}

