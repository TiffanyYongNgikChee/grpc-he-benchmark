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
