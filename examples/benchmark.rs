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