// An example of an medical data

// This import symbols (functions, structs, etc) from the he_benchmark library
use he_benchmark::{
    Context, Encryptor, Decryptor, BatchEncoder, GaloisKeys,
    Plaintext, add, multiply, rotate_rows
};
// From Rust's standard library (std) -
// Instant (used to record precise timestamps - for measuring elapsed time)
// Duraration - Represents a span of time (e.g. 100 milliseconds).
use std::time::{Instant, Duration};
// Imports the sleep function, which pauses the program for a given Duration.
use std::thread::sleep;
// for input and output capabilities. To flush output immediatelyy (so progress bars update in real time)
use std::io::{self, Write};

/// Clear screen and move cursor to top (for progress animation)
fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H"); // This is an ANSI escape sequence used in terminal control
    io::stdout().flush().unwrap(); // Ensures that the printed escape sequence is actually sent to the terminal immediately (not buffered)
}

// Introduce the entry point (main)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    clear_screen();
    
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           MEDICAL DATA ENCRYPTION - SEAL DEMONSTRATION            ║");
    println!("║                                                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ============================================
    // SIMULATED MEDICAL DATA
    // ============================================
    let medical_record = 
        "PATIENT ID: 12345 | NAME: Bob Smith | AGE: 45 | \
         DIAGNOSIS: HYPERTENSION STAGE 2 | BP: 160/100 | \
         MEDICATION: LISINOPRIL 10MG DAILY | \
         ALLERGIES: EGG | LAST VISIT: 2024-11-10 | \
         NOTES: PATIENT SHOWS IMPROVEMENT WITH CURRENT TREATMENT";
    
    println!("📋 Medical Record to Encrypt:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{}", medical_record);
    println!("─────────────────────────────────────────────────────────────────");
    println!("encrypt it using SEAL homomorphic encryption\n");
    
    sleep(Duration::from_secs(2));

    // SETUP & KEY GENERATION
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           CRYPTOGRAPHIC SETUP                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Creates a timestamp storing the current instant (high-resolution clock) in the variable phase1_start
    let phase1_start = Instant::now(); // Instant::now() is used for measuring elapsed time later.
    
    // Calls helper function processing_step (defined earlier).
    // a label that will be printed.800 is means the simulated duration in milliseconds.
    processing_step("Initializing SEAL encryption context", 800);

    // This actually constructs the encryption context using the: 
    // Context::new function from the he_benchmark crate (a wrapper for SEAL).
    // 8192 — typically the polynomial degree (also called poly_modulus_degree). 
    // This controls the ciphertext polynomial degree and affects performance/security tradeoffs. Larger means more capacity but slower.
    // 1032193 — likely a modulus (coefficient/modulus parameter). The exact meaning depends on he_benchmark API.
    let context = Context::new(8192, 1032193)?;
    

    // The processing_step simulates visible progress; sleep adds extra perceived delay to mimic a heavier real-world key generation operation. 
    // (In real setups key generation can take a noticeable time.)
    // processing_step to simulate and display the “Generating public/private key pair” action over ~1200 ms.
    processing_step("Generating public/private key pair", 1200);
    // sleep(Duration::from_millis(500)) pauses execution for another 500 ms.
    sleep(Duration::from_millis(500));
    
    processing_step("Creating batch encoder for medical data", 600);
    
    // Instantiates a BatchEncoder using the context (passes a reference so the encoder can access encryption parameters.)
    // BatchEncoder is used to pack multiple integers (or bytes converted to integers) into a plaintext polynomial that SEAL can encrypt efficiently (SIMD-style batching).
    let encoder = BatchEncoder::new(&context)?;

    // to find how many separate "slots" (elements) can be packed into a single plaintext polynomial given the chosen parameters.
    let slot_count = encoder.slot_count();
    
    // Simulates creating Galois keys with a longer animation (2500 ms) because generating these special keys is usually more expensive.
    processing_step("Generating Galois keys for data rotation", 2500);
    
    // generates the GaloisKeys object by calling GaloisKeys::generate and passing the context. 
    // will be used whenever the code needs to rotate ciphertext slots while remaining encrypted.
    let galois_keys = GaloisKeys::generate(&context)?;
    
    // capture how long Phase 1 took (initialization, encoder, key generation) for reporting in the summary.
    let phase1_time = phase1_start.elapsed();
    
    println!("\n  Phase 1 Complete!");
    // show the number of slot_count available for batching encoded data.
    println!("   Available encryption slots: {}", slot_count);
    // prints the setup time in seconds to two decimal places:
    println!("   Setup time: {:.2}s\n", phase1_time.as_secs_f64());
    
    sleep(Duration::from_secs(1));

    // DATA PREPARATION & ENCODING
    // converting the text into a numerical representation and encoding it into SEAL’s plaintext polynomial form.
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           DATA PREPARATION                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Captures the current timestamp, so can measure how long this entire phase takes.
    let phase2_start = Instant::now();
    
    // Calls helper function processing_step, which animates a short dot-based progress bar (700 ms).
    processing_step("Converting medical text to numerical format", 700);
    // creates an iterator over every character in the string.
    // Example: "AB" → 'A', 'B'.
    // And then converts each character into its Unicode numeric code
    // We cast to i64 because SEAL encoders operate on integers, not text.
    // gathers all converted numbers into a Vec<i64> (vector of 64-bit integers).
    let data: Vec<i64> = medical_record.chars().map(|c| c as i64).collect();
    // text becomes a list of integers, one per character. So we can perform mathematical operations (encryption, rotation) on this vector.
    
    // to show how many characters were in the original record (data.len() gives the vector length).
    println!("   📊 Record length: {} characters", data.len());
    
    processing_step("Padding data to encryption block size", 500);
    
    // Pads the data vector to match the encoder’s slot capacity.
    // data.clone() → creates a copy of the original data. (because the original data need to be modified later)
    let mut padded_data = data.clone();
    // Ensures the vector’s length equals slot_count (the number of encoding slots retrieved earlier from the encoder).
    // If the data is shorter than slot_count, it appends zeros (0) until the lengths match.
    // *** Why pad? ***
    // Homomorphic encryption encodes data into fixed-size vectors (slots).
    // Every plaintext polynomial can hold exactly slot_count numbers, so we must fill unused slots with zeros.
    padded_data.resize(slot_count, 0);
    
    processing_step("Encoding data into plaintext polynomial", 800);
    
    // Performs the real encoding using the SEAL BatchEncoder.
    // converts the vector of integers into a Plaintext object.
    let plaintext = encoder.encode(&padded_data)?;
    //plaintext now holds the polynomial representation of data text, ready to encrypt.
    
    // Calculates how long data preparation took by subtracting the saved start time from the current time.
    let phase2_time = phase2_start.elapsed();
    
    println!("\n   Phase 2 Complete!");
    println!("   Data encoded: {} → {} slots", data.len(), slot_count);
    println!("   Encoding time: {:.2}s\n", phase2_time.as_secs_f64());
    
    sleep(Duration::from_secs(1));
    
    Ok(())
}

/// Print progress bar
// label - name of the process (e.g. "Encryption")
// current and total - track how far
// elapsed - how much time has passed
fn print_progress(label: &str, current: usize, total: usize, elapsed: Duration) {
    // Calculates progress as a percentage (integer)
    let percentage = (current as f64 / total as f64 * 100.0) as usize;
    // Define the visual width of the bar and how many segments should be filled or empty
    let bar_width = 50;
    let filled = (percentage * bar_width / 100).min(bar_width);
    let empty = bar_width - filled;
    
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  {} Progress: {}%", label, percentage);
    println!("├─────────────────────────────────────────────────────────────────┤");
    print!("│  [");
    print!("{}", "█".repeat(filled));
    print!("{}", "░".repeat(empty));
    println!("] {}/{}", current, total);
    println!("│  Elapsed: {:.1}s", elapsed.as_secs_f64());
    println!("└─────────────────────────────────────────────────────────────────┘");
}

/// Simulates some work being done (e.g., key generation, encoding, etc.).
fn processing_step(label: &str, duration_ms: u64) {
    print!("   {} ", label); // Prints the task name (like "Generating keys") and flushes output.
    io::stdout().flush().unwrap();
    
    // Breaks the simulated process into 20 small substeps (so we can print dots).
    let steps = 20;
    let step_duration = duration_ms / steps;
    
    // Loops 20 times, printing a dot (.) each iteration.
    for _ in 0..steps {
        print!(".");
        io::stdout().flush().unwrap();
        // Sleeps a short time between each to create a progress animation.
        sleep(Duration::from_millis(step_duration));
    }
    println!(" /DONE");

}
