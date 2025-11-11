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
    println!("    Record length: {} characters", data.len());
    
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

    // ENCRYPTION
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           HOMOMORPHIC ENCRYPTION                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Captures the current high-precision timestamp to measure how long Phase 3 (encryption) takes in total.
    let phase3_start = Instant::now(); 
    
    processing_step("Initializing encryptor with public key", 400);
    // Creates an Encryptor instance using the previously defined encryption context
    // &context — provides encryption parameters (moduli, keys, etc.) created during setup.
    let encryptor = Encryptor::new(&context)?;
    // Encryptor is the object responsible for turning plaintexts into ciphertexts using the public key.
    // Once created, can call encryptor.encrypt() to actually encrypt data.

    println!("\n   Encrypting medical record...");
    // Starts a new timer to measure only the encryption process, not the whole phase.
    let encrypt_start = Instant::now();
    
    // Show progress during encryption
    // Simulates a live encryption progress animation.
    // loop five times to represent progress steps (20%, 40%, 60%, 80%, 100%).
    for i in 0..5 {
        // pause for 200 ms to pace the updates.
        sleep(Duration::from_millis(200));
        // print_progress("Encryption", i + 1, 5, encrypt_start.elapsed());
        print_progress("Encryption", i + 1, 5, encrypt_start.elapsed());
        // uses ANSI escape codes to move the terminal cursor up six lines.
        print!("\x1B[6A"); // Move cursor up 6 lines
    }
    // Calls the encrypt method of the Encryptor, passing in the plaintext generated in Phase 2.
    // The encryptor uses the public key (already part of the context) to transform the plaintext polynomial into a ciphertext 
    // — an encrypted polynomial that hides the original data.
    let ciphertext = encryptor.encrypt(&plaintext)?;
    // And now the record is fully encrypted: it can be added, multiplied, or rotated under encryption without exposing raw data.
    
    // Calculates how long the actual encryption call took by comparing the current time to encrypt_start.
    let encrypt_time = encrypt_start.elapsed();
    
    print_progress("Encryption", 5, 5, encrypt_time);
    
    // Computes the total time taken for all of Phase 3 (including initialization, progress display, and encryption).
    let phase3_time = phase3_start.elapsed();
    
    println!("\n    Medical record is now ENCRYPTED!");
    println!("    Data is secure - cannot be read without secret key");
    println!("    Encryption time: {:.2}s\n", encrypt_time.as_secs_f64());
    
    sleep(Duration::from_secs(2));

    // PHASE 4: ENCRYPTED OPERATIONS (DEMO)
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           OPERATIONS ON ENCRYPTED DATA                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("    Demonstrating homomorphic operations:");
    println!("   → Rotating encrypted data without decryption\n");
    
    let phase4_start = Instant::now();
    
    processing_step("Rotating patient data by 5 positions", 1500);
    
    // Homomorphic encryption stores data in slots. 
    // A rotation moves the encrypted data around within these slots 
    // — similar to rotating an array — but without decrypting anything.

    // &context → the encryption context that holds parameters and keys.
    // &ciphertext → the encrypted medical record produced in Phase 3.
    // 5 → the rotation step — shift data by 5 positions.
    // &galois_keys → special keys that enable rotation operations on encrypted data.
    // If the encrypted slots represent [A, B, C, D, E], a rotation by 2 would produce [C, D, E, A, B] — still encrypted, but rearranged.
    let rotated = rotate_rows(&context, &ciphertext, 5, &galois_keys)?;
    
    processing_step("Preparing second encrypted value for demo", 800);
    // Creates a vector of ones (1i64) with length equal to slot_count (the number of available encryption slots determined earlier).
    let offset = vec![1i64; slot_count]; // it will later be encoded and encrypted to form another ciphertext, used to show that addition works on encrypted data.
    
    // This step doesn’t encrypt yet — it just prepares the data.
    // Encodes that vector of ones into a plaintext polynomial (the internal format required for homomorphic encryption).
    let plain2 = encoder.encode(&offset)?;
    // Encrypts that encoded plaintext into another ciphertext (cipher2).
    let cipher2 = encryptor.encrypt(&plain2)?;
    // Both are fully encrypted and cannot be read directly.
    
    processing_step("Adding encrypted values (still encrypted!)", 600);
    // Performs homomorphic addition between two ciphertexts — rotated and cipher2.
    let cipher_sum = add(&context, &rotated, &cipher2)?;
    
    let phase4_time = phase4_start.elapsed();
    
    println!("\n    Encrypted operations complete!");
    println!("    All operations done WITHOUT seeing the data");
    println!("    Operation time: {:.2}s\n", phase4_time.as_secs_f64());
    
    sleep(Duration::from_secs(1));
    
    // DECRYPTION & VERIFICATION
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           DECRYPTION & VERIFICATION                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    
    let phase5_start = Instant::now();
    
    processing_step("Initializing decryptor with secret key", 400);
    // Creates a Decryptor instance using the current encryption context.
    // The Decryptor uses the private (secret) key stored inside the context.
    // Only with this secret key can this transform ciphertexts back into readable plaintexts.
    let decryptor = Decryptor::new(&context)?;
    
    
    println!("\n   Decrypting result...");
    // Starts another timer, this one measuring just the decryption process itself (not the entire phase).
    let decrypt_start = Instant::now();
    
    // Show progress during decryption
    for i in 0..5 {
        sleep(Duration::from_millis(150));
        print_progress("Decryption", i + 1, 5, decrypt_start.elapsed());
        print!("\x1B[6A"); //Uses an ANSI escape code to move the cursor up 6 lines so the next update overwrites the old one, making it look animated.
    }
    
    // This is the actual decryption step.
    // It takes cipher_sum, which was created in Phase 4 (the result of encrypted additions and rotations), and decrypts it back into a Plaintext.
    // The Decryptor uses the secret key to reverse the encryption process.
    // The result, decrypted_plain, is now a plaintext polynomial, still numeric, but readable to the program (not directly human-readable text yet).
    let decrypted_plain = decryptor.decrypt(&cipher_sum)?;
    // Measures how long the decryption step took, in seconds.
    let decrypt_time = decrypt_start.elapsed();
    
    print_progress("Decryption", 5, 5, decrypt_time);
    
    processing_step("Decoding plaintext back to readable format", 600);
    // Converts (decodes) the decrypted plaintext polynomial back into a vector of integers.
    // Each integer represents an ASCII code for one character (from Phase 2’s encoding step).
    // After this step, result_data holds a vector like [80, 65, 84, 73, 69, 78, 84, ...]
    let result_data = encoder.decode(&decrypted_plain)?;
    
    let phase5_time = phase5_start.elapsed();
    
    // Converts the numeric vector into a human-readable text string.
    // result_data[..data.len().min(100)] - Takes at most the first 100 characters (for preview).
    // .iter() — Iterates over the numbers.
    // .filter(|&&n| n > 0 && n < 128) — Keeps only valid ASCII characters (ignore zeros or noise from padding).
    // .map(|&n| (n as u8) as char) — Converts each numeric value back into its character equivalent.
    // .collect() — Combines all characters into a String.
    let result_string: String = result_data[..data.len().min(100)]
        .iter()
        .filter(|&&n| n > 0 && n < 128)
        .map(|&n| (n as u8) as char)
        .collect();
    // Result: a readable text version of the decrypted message — i.e., the medical record.
    println!("\n   Decryption successful!");
    println!("   Decrypted preview: \"{}...\"", &result_string[..50.min(result_string.len())]);
    println!("   Decryption time: {:.2}s\n", phase5_time.as_secs_f64());
    
    sleep(Duration::from_secs(2));
    
    
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
