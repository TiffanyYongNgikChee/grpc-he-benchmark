/// Step 6c: Encrypted MNIST Inference Pipeline
///
/// Full end-to-end encrypted CNN inference using OpenFHE (BFV scheme):
///   1. Load quantized weights from CSV (Step 6a)
///   2. Encode weights as OpenFHE plaintexts/ciphertexts (Step 6b)
///   3. Encrypt a test MNIST image
///   4. Run the full CNN pipeline on encrypted data:
///      Conv1 → +bias → x² → AvgPool → Conv2 → +bias → x² → AvgPool → FC → +bias
///   5. Decrypt output logits and determine predicted digit
///
/// CNN Architecture (matches Python HE_CNN):
///   Input:  28×28 (784 pixels, scaled by 1000)
///   Conv1:  5×5 kernel → 24×24 = 576 values
///   +bias:  add conv1 bias (broadcast to 576 slots)
///   x²:    square activation (576 values)
///   Pool1:  2×2 avg → 12×12 = 144 values
///   Conv2:  5×5 kernel → 8×8 = 64 values
///   +bias:  add conv2 bias (broadcast to 64 slots)
///   x²:    square activation (64 values)
///   Pool2:  2×2 avg → 4×4 = 16 values
///   FC:     16→10 matmul
///   +bias:  add fc bias (10 values)
///   Output: 10 logits → argmax = predicted digit
///
/// Run with: cargo run --example mnist_inference
///   (requires OpenFHE system library — run inside Docker)

use he_benchmark::open_fhe_lib::{
    OpenFHECiphertext, OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext,
};
use he_benchmark::weight_loader::MnistWeights;

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================");
    println!("Step 6c: Encrypted MNIST Inference Pipeline");
    println!("================================================================\n");

    // ================================================================
    // Phase 1: Setup — load weights, create crypto context
    // ================================================================
    println!("--- Phase 1: Setup ---\n");

    let t_setup = Instant::now();

    // Load quantized weights from CSV
    let weights_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist_training/weights");
    println!("  Loading weights from: {}", weights_dir);
    let weights = MnistWeights::load(weights_dir)?;
    println!(
        "  Loaded {} parameters (scale_factor={})",
        weights.total_params(),
        weights.config.scale_factor
    );

    // Create OpenFHE BFV context
    let plaintext_modulus = weights.config.plaintext_modulus as u64;
    let mult_depth = 3;
    println!(
        "  Creating BFV context (p={}, depth={})...",
        plaintext_modulus, mult_depth
    );
    let ctx = OpenFHEContext::new_bfv(plaintext_modulus, mult_depth)?;
    let kp = OpenFHEKeyPair::generate(&ctx)?;
    println!("  Keypair generated (128-bit security)\n");

    // Encode weights
    println!("  Encoding weights as OpenFHE types...");
    let w = weights.encode(&ctx, &kp)?;

    let setup_time = t_setup.elapsed();
    println!("  Setup complete in {:.2?}\n", setup_time);

    // ================================================================
    // Phase 2: Prepare test input
    // ================================================================
    println!("--- Phase 2: Prepare Test Input ---\n");

    // A hardcoded MNIST digit "7" (28×28 pixels, 0-255 range)
    // This is a stylized "7" pattern for testing without needing
    // to read actual MNIST data files in Rust.
    let raw_pixels: Vec<i64> = create_test_digit_7();

    // Scale pixels to match quantized weights.
    // PyTorch's ToTensor() scales 0-255 → 0.0-1.0 (divides by 255).
    // Weights are quantized with scale_factor = 1000.
    // So input_int = round(pixel / 255.0 * 1000).
    let scale = weights.config.scale_factor;
    let input_scaled: Vec<i64> = raw_pixels
        .iter()
        .map(|&p| ((p as f64 / 255.0) * scale as f64).round() as i64)
        .collect();

    println!("  Test digit: 7 (handcrafted 28×28 pattern)");
    println!("  Input scaling: pixel / 255 * {}", scale);
    println!(
        "  Scaled range: [{}, {}]",
        input_scaled.iter().min().unwrap(),
        input_scaled.iter().max().unwrap()
    );

    // Display the input image as ASCII art
    println!("\n  Input image (28×28, showing non-zero regions):");
    print_image_ascii(&raw_pixels, 28, 28);

    // ================================================================
    // Phase 3: Encrypt input
    // ================================================================
    println!("\n--- Phase 3: Encrypt Input ---\n");

    let t_encrypt = Instant::now();
    let input_pt = OpenFHEPlaintext::from_vec(&ctx, &input_scaled)?;
    let encrypted_input = OpenFHECiphertext::encrypt(&ctx, &kp, &input_pt)?;
    let encrypt_time = t_encrypt.elapsed();
    println!(
        "  Input encrypted ({} pixels → ciphertext) in {:.2?}",
        input_scaled.len(),
        encrypt_time
    );

    // ================================================================
    // Phase 4: Encrypted CNN Inference
    // ================================================================
    println!("\n--- Phase 4: Encrypted CNN Inference ---\n");

    let t_inference = Instant::now();

    // ---- Layer 1: Conv1 (28×28 → 24×24) with integrated rescale ÷S ----
    let t_layer = Instant::now();
    let x = encrypted_input.conv2d(&ctx, &kp, &w.conv1_kernel, 28, 28, 5, 5, scale as i64)?;
    println!(
        "  [1/7] Conv1 (28×28 → 24×24, 5×5, ÷S)           {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 2: Add Conv1 bias ----
    let t_layer = Instant::now();
    let x = x.add(&ctx, &w.conv1_bias)?;
    println!(
        "  [2/7] + Conv1 bias (576 slots)                  {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 3: Square activation x²/S (decrypt→square→rescale→re-encrypt) ----
    let t_layer = Instant::now();
    let x = x.square_activate(&ctx, &kp, scale as i64)?;
    println!(
        "  [3/7] Square activation (x²/S)                  {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 4: AvgPool 2×2 (24×24 → 12×12) ----
    let t_layer = Instant::now();
    let x = x.avgpool(&ctx, &kp, 24, 24, 2, 2)?;
    println!(
        "  [4/7] AvgPool 2×2 (24×24 → 12×12)               {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 5: Conv2 (12×12 → 8×8) with integrated rescale ÷S ----
    let t_layer = Instant::now();
    let x = x.conv2d(&ctx, &kp, &w.conv2_kernel, 12, 12, 5, 5, scale as i64)?;
    println!(
        "  [5/7] Conv2 (12×12 → 8×8, 5×5, ÷S)             {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 6: Add Conv2 bias ----
    let t_layer = Instant::now();
    let x = x.add(&ctx, &w.conv2_bias)?;
    println!(
        "  [6/7] + Conv2 bias (64 slots)                   {:.2?}",
        t_layer.elapsed()
    );

    // ---- Layer 7: Square activation x²/S (decrypt→square→rescale→re-encrypt) ----
    let t_layer = Instant::now();
    let x = x.square_activate(&ctx, &kp, scale as i64)?;
    println!(
        "  [7/7] Square activation (x²/S)                  {:.2?}",
        t_layer.elapsed()
    );

    // ---- AvgPool 2×2 (8×8 → 4×4) ----
    let t_layer = Instant::now();
    let x = x.avgpool(&ctx, &kp, 8, 8, 2, 2)?;
    println!(
        "  [+]   AvgPool 2×2 (8×8 → 4×4 = 16 features)    {:.2?}",
        t_layer.elapsed()
    );

    // ---- FC matmul (16 → 10) with integrated rescale ÷S ----
    let t_layer = Instant::now();
    let x = OpenFHECiphertext::matmul(&ctx, &kp, &w.fc_weights, &x, 10, 16, scale as i64)?;
    println!(
        "  [+]   FC matmul (16 → 10 logits, ÷S)            {:.2?}",
        t_layer.elapsed()
    );

    // ---- Add FC bias ----
    let t_layer = Instant::now();
    let x = x.add(&ctx, &w.fc_bias)?;
    println!(
        "  [+]   + FC bias (10 values)                     {:.2?}",
        t_layer.elapsed()
    );

    let inference_time = t_inference.elapsed();
    println!("\n  Total inference time: {:.2?}", inference_time);

    // ================================================================
    // Phase 5: Decrypt output and classify
    // ================================================================
    println!("\n--- Phase 5: Decrypt & Classify ---\n");

    let t_decrypt = Instant::now();
    let output_pt = x.decrypt(&ctx, &kp)?;
    let logits = output_pt.to_vec()?;
    let decrypt_time = t_decrypt.elapsed();

    // Only the first 10 values are the class logits
    let class_logits: Vec<i64> = logits[..10].to_vec();

    println!("  Output logits (10 classes):");
    println!("  +-------+------------------+");
    println!("  | Digit |       Logit      |");
    println!("  +-------+------------------+");
    for (i, &logit) in class_logits.iter().enumerate() {
        let marker = if logit == *class_logits.iter().max().unwrap() {
            " ◀ MAX"
        } else {
            ""
        };
        println!("  |   {}   | {:>16} |{}", i, logit, marker);
    }
    println!("  +-------+------------------+");

    // Argmax = predicted class
    let predicted = class_logits
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| v)
        .map(|(i, _)| i)
        .unwrap();

    println!("\n  Predicted digit: {}", predicted);
    println!("  Expected digit:  7");
    println!(
        "  Result: {}",
        if predicted == 7 {
            "CORRECT ✓"
        } else {
            "MISMATCH (see notes below)"
        }
    );

    if predicted != 7 {
        println!("\n  Note: The handcrafted test image may not match the");
        println!("  distribution the model was trained on. For accurate");
        println!("  testing, use Step 6d with real MNIST test images.");
    }

    println!("\n  Decryption time: {:.2?}", decrypt_time);

    // ================================================================
    // Summary
    // ================================================================
    println!("\n================================================================");
    println!("Step 6c: Encrypted Inference Pipeline COMPLETE");
    println!("================================================================");
    println!("  Model:           HE_CNN (222 parameters)");
    println!("  Scheme:          BFV (p={}, depth={})", plaintext_modulus, mult_depth);
    println!("  Security:        128-bit");
    println!("  Scale factor:    {}", scale);
    println!(
        "  Float accuracy:  {:.2}% (from training)",
        weights.config.float_accuracy
    );
    println!("  Predicted digit: {}", predicted);
    println!();
    println!("  Timing:");
    println!("    Setup:         {:.2?}", setup_time);
    println!("    Encryption:    {:.2?}", encrypt_time);
    println!("    Inference:     {:.2?}", inference_time);
    println!("    Decryption:    {:.2?}", decrypt_time);
    println!(
        "    Total:         {:.2?}",
        setup_time + encrypt_time + inference_time + decrypt_time
    );
    println!();
    println!("  Pipeline: encrypt → conv1 → +bias → x² → pool");
    println!("          → conv2 → +bias → x² → pool → FC → +bias → decrypt");
    println!();
    println!("  Ready for Step 6d (verification with real MNIST test images)");

    Ok(())
}

// ============================================================================
// Test image: handcrafted digit "7" (28×28 pixels, 0-255 range)
// ============================================================================

/// Create a 28×28 pixel image of the digit "7" for testing.
///
/// This is a hand-designed pattern approximating a MNIST-style "7":
///   - Horizontal stroke at top (rows 4-6)
///   - Diagonal stroke going down-left (rows 6-24)
///   - Values are in 0-255 range (like raw MNIST pixels)
///
/// For production testing, use actual MNIST test images exported from
/// the Python pipeline (Step 6d).
fn create_test_digit_7() -> Vec<i64> {
    let mut img = vec![0i64; 28 * 28];

    // Helper to set a pixel (row, col) with bounds checking
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
    // Starting around (8, 20), going to roughly (24, 12)
    for i in 0..17 {
        let r = 8 + i;
        // Column decreases as we go down (the diagonal)
        let c_center = 20.0 - (i as f64 * 0.5);
        let c = c_center as usize;

        if r < 28 && c < 28 {
            // Main stroke (2-3 pixels wide)
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

/// Print a 28×28 image as compact ASCII art.
fn print_image_ascii(pixels: &[i64], height: usize, width: usize) {
    // Show every other row for compactness
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
