/// Standalone verification of the weight loader CSV parsing logic.
///
/// This script duplicates the core parsing functions from weight_loader.rs
/// so we can test them WITHOUT linking against HE libraries.
///
/// Run with: rustc examples/verify_weight_loader_standalone.rs -o /tmp/verify_weights && /tmp/verify_weights

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

/// Parse a CSV file into a Vec<i64>, verify count matches expected
fn load_csv(path: &Path, expected: usize) -> Vec<i64> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {:?}: {}", path, e));
    let reader = io::BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        for token in trimmed.split(',') {
            let token = token.trim();
            if !token.is_empty() {
                let val: i64 = token.parse().unwrap_or_else(|e| panic!("Bad int '{}': {}", token, e));
                values.push(val);
            }
        }
    }

    assert_eq!(values.len(), expected,
        "Shape mismatch in {:?}: expected {} values, got {}", path, expected, values.len());
    values
}

/// Extract a string value from JSON content (simple manual parser)
fn json_get_str(content: &str, key: &str) -> String {
    let pattern = format!("\"{}\"", key);
    let pos = content.find(&pattern).unwrap_or_else(|| panic!("Missing key: {}", key));
    let after = &content[pos + pattern.len()..];
    let q1 = after.find('"').unwrap();
    let rest = &after[q1 + 1..];
    let q2 = rest.find('"').unwrap();
    rest[..q2].to_string()
}

/// Extract a numeric value from JSON content
fn json_get_num(content: &str, key: &str) -> f64 {
    let pattern = format!("\"{}\"", key);
    let pos = content.find(&pattern).unwrap_or_else(|| panic!("Missing key: {}", key));
    let after = &content[pos + pattern.len()..];
    let colon = after.find(':').unwrap();
    let rest = after[colon + 1..].trim_start();
    let end = rest.find(|c: char| c == ',' || c == '\n' || c == '}' || c == ']').unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().unwrap_or_else(|e| panic!("Bad number for {}: {}", key, e))
}

fn main() {
    println!("================================================================");
    println!("Step 6a: Standalone Weight Loader Verification");
    println!("================================================================\n");

    // Resolve weights directory
    // When compiled with rustc directly, use relative path
    // When compiled with cargo, CARGO_MANIFEST_DIR would be available
    let weights_dir = Path::new("mnist_training/weights");
    if !weights_dir.is_dir() {
        eprintln!("ERROR: weights directory not found at {:?}", weights_dir);
        eprintln!("  Run from project root, or run train_mnist.py first.");
        std::process::exit(1);
    }

    let dir = &weights_dir;
    println!("Loading from: {:?}\n", dir);

    // ---- Load config ----
    let config_content = fs::read_to_string(dir.join("model_config.json"))
        .expect("Cannot read model_config.json");
    
    let model_name = json_get_str(&config_content, "model_name");
    let scale_factor = json_get_num(&config_content, "scale_factor") as i64;
    let plaintext_modulus = json_get_num(&config_content, "plaintext_modulus") as u64;
    let float_accuracy = json_get_num(&config_content, "float_model");
    let total_parameters = json_get_num(&config_content, "total_parameters") as usize;

    println!("  Model: {} (scale={}, modulus={}, accuracy={:.2}%)\n",
             model_name, scale_factor, plaintext_modulus, float_accuracy);

    // ---- Load all CSV files ----
    let conv1_w = load_csv(&dir.join("conv1_weights.csv"), 25);
    let conv1_b = load_csv(&dir.join("conv1_bias.csv"), 1);
    let conv2_w = load_csv(&dir.join("conv2_weights.csv"), 25);
    let conv2_b = load_csv(&dir.join("conv2_bias.csv"), 1);
    let fc_w    = load_csv(&dir.join("fc_weights.csv"), 160);
    let fc_b    = load_csv(&dir.join("fc_bias.csv"), 10);

    let total = conv1_w.len() + conv1_b.len() + conv2_w.len() + conv2_b.len() + fc_w.len() + fc_b.len();

    // ---- Test 1: Shapes ----
    println!("--- Test 1: Shape Verification ---");
    println!("  ✓ conv1_weights: {} values (5×5 kernel)", conv1_w.len());
    println!("  ✓ conv1_bias:    {} value",  conv1_b.len());
    println!("  ✓ conv2_weights: {} values (5×5 kernel)", conv2_w.len());
    println!("  ✓ conv2_bias:    {} value",  conv2_b.len());
    println!("  ✓ fc_weights:    {} values (10×16 matrix)", fc_w.len());
    println!("  ✓ fc_bias:       {} values", fc_b.len());
    println!("  Total: {} parameters", total);
    assert_eq!(total, 222);
    println!("  ✓ Total matches expected 222\n");

    // ---- Test 2: Config ----
    println!("--- Test 2: Config Verification ---");
    assert_eq!(model_name, "HE_CNN");           println!("  ✓ model_name = HE_CNN");
    assert_eq!(scale_factor, 1000);             println!("  ✓ scale_factor = 1000");
    assert_eq!(plaintext_modulus, 7340033);      println!("  ✓ plaintext_modulus = 7340033");
    assert_eq!(total_parameters, 222);           println!("  ✓ total_parameters = 222");
    assert!(float_accuracy > 80.0);              println!("  ✓ float_accuracy = {:.2}% (> 80%)", float_accuracy);
    println!();

    // ---- Test 3: Value range ----
    println!("--- Test 3: Value Range (BFV modulus check) ---");
    let max_half = (plaintext_modulus / 2) as i64;
    let all: Vec<&i64> = conv1_w.iter().chain(conv1_b.iter())
        .chain(conv2_w.iter()).chain(conv2_b.iter())
        .chain(fc_w.iter()).chain(fc_b.iter()).collect();
    
    let max_abs = all.iter().map(|v| v.abs()).max().unwrap();
    let min_val = **all.iter().min().unwrap();
    let max_val = **all.iter().max().unwrap();
    
    println!("  Range: [{}, {}]", min_val, max_val);
    println!("  Max |value|: {} (limit: ±{})", max_abs, max_half);
    assert!(max_abs <= max_half, "Values exceed plaintext modulus!");
    println!("  ✓ All values fit within BFV modulus\n");

    // ---- Test 4: Spot-check known values ----
    println!("--- Test 4: Spot-Check Known Values ---");
    
    // Conv1 row 0: -151,-198,16,114,115
    assert_eq!(conv1_w[0], -151); assert_eq!(conv1_w[1], -198);
    assert_eq!(conv1_w[2], 16);   assert_eq!(conv1_w[3], 114);
    assert_eq!(conv1_w[4], 115);
    println!("  ✓ Conv1 row 0: {:?}", &conv1_w[0..5]);

    // Conv1 bias: 926
    assert_eq!(conv1_b[0], 926);
    println!("  ✓ Conv1 bias: {:?}", conv1_b);

    // Conv2 row 0: -310,438,51,-120,-349
    assert_eq!(conv2_w[0], -310); assert_eq!(conv2_w[1], 438);
    println!("  ✓ Conv2 row 0: {:?}", &conv2_w[0..5]);

    // Conv2 bias: 629
    assert_eq!(conv2_b[0], 629);
    println!("  ✓ Conv2 bias: {:?}", conv2_b);

    // FC row 0 first value: -34
    assert_eq!(fc_w[0], -34);
    println!("  ✓ FC row 0: {:?}", &fc_w[0..16]);

    // FC bias: -320,1405,-789,-713,734,244,618,466,-1505,-135
    assert_eq!(fc_b[0], -320);  assert_eq!(fc_b[1], 1405);
    assert_eq!(fc_b[8], -1505); assert_eq!(fc_b[9], -135);
    println!("  ✓ FC bias: {:?}", fc_b);

    // ---- Test 5: Row accessor simulation ----
    println!("\n--- Test 5: FC Row Accessor ---");
    for class_idx in 0..10 {
        let start = class_idx * 16;
        let row = &fc_w[start..start + 16];
        assert_eq!(row.len(), 16);
    }
    println!("  ✓ All 10 FC rows have 16 values each");

    // ---- Per-layer statistics ----
    println!("\n--- Layer Statistics ---");
    let layers: Vec<(&str, &[i64])> = vec![
        ("Conv1 weights", &conv1_w),
        ("Conv1 bias",    &conv1_b),
        ("Conv2 weights", &conv2_w),
        ("Conv2 bias",    &conv2_b),
        ("FC weights",    &fc_w),
        ("FC bias",       &fc_b),
    ];
    println!("  {:<16} {:>6} {:>8} {:>8}", "Layer", "Count", "Min", "Max");
    println!("  {:<16} {:>6} {:>8} {:>8}", "─".repeat(16), "─".repeat(6), "─".repeat(8), "─".repeat(8));
    for (name, vals) in &layers {
        let min = vals.iter().copied().min().unwrap();
        let max = vals.iter().copied().max().unwrap();
        println!("  {:<16} {:>6} {:>8} {:>8}", name, vals.len(), min, max);
    }

    // ---- Summary ----
    println!("\n================================================================");
    println!("Step 6a COMPLETE: Weight loader verified ✓");
    println!("================================================================");
    println!("  Files loaded:       6 CSVs + model_config.json");
    println!("  Parameters:         {}", total);
    println!("  Scale factor:       {}", scale_factor);
    println!("  Accuracy (float):   {:.2}%", float_accuracy);
    println!("  BFV range check:    ✓ (all ≤ ±{})", max_half);
    println!("  Spot-checks:        ✓");
    println!("  Ready for Step 6b (encode weights as OpenFHE plaintexts)");
}
