/// Encrypted Inference Verification with Real MNIST Images
///
/// Runs the full encrypted CNN pipeline (from Step 6c) on real MNIST
/// test images exported by `export_test_images.py`, and compares the
/// encrypted predictions against ground-truth labels.
///
/// Pipeline per image:
///   1. Scale raw pixels: round(pixel / 255 * 1000)
///   2. Encrypt scaled image
///   3. Run full CNN: Conv1→+bias→x²→Pool→Conv2→+bias→x²→Pool→FC→+bias
///   4. Decrypt output logits
///   5. Argmax → predicted digit
///   6. Compare with ground-truth label
///
/// Prerequisites:
///   - Run `train_mnist.py` to train and export weights
///   - Run `export_test_images.py` to export test images
///   - Build inside Docker where OpenFHE is installed
///
/// Run with: cargo run --example mnist_verify

use he_benchmark::open_fhe_lib::{
    OpenFHECiphertext, OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext,
};
use he_benchmark::weight_loader::{load_test_images, MnistWeights};

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================");
    println!("Step 6d: Encrypted Inference Verification");
    println!("================================================================\n");

    let t_total = Instant::now();

    // ================================================================
    // Phase 1: Load weights and test images
    // ================================================================
    println!("--- Phase 1: Load Weights & Test Images ---\n");

    let weights_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist_training/weights");

    let weights = MnistWeights::load(weights_dir)?;
    println!();

    let test_images = load_test_images(weights_dir)?;
    println!();

    if test_images.is_empty() {
        println!("  ERROR: No test images found!");
        println!("  Run: cd mnist_training && python export_test_images.py");
        return Ok(());
    }

    // ================================================================
    // Phase 2: Initialize encryption
    // ================================================================
    println!("--- Phase 2: Initialize Encryption ---\n");

    let t_setup = Instant::now();
    let plaintext_modulus = weights.config.plaintext_modulus as u64;
    let mult_depth = 3;

    let ctx = OpenFHEContext::new_bfv(plaintext_modulus, mult_depth)?;
    let kp = OpenFHEKeyPair::generate(&ctx)?;
    println!("  BFV context created (p={}, depth={})", plaintext_modulus, mult_depth);
    println!("  Keypair generated (128-bit security)");

    // Encode weights
    let w = weights.encode(&ctx, &kp)?;
    let setup_time = t_setup.elapsed();
    println!("  Setup time: {:.2?}\n", setup_time);

    // ================================================================
    // Phase 3: Run encrypted inference on all test images
    // ================================================================
    println!("--- Phase 3: Encrypted Inference (all test images) ---\n");

    let scale = weights.config.scale_factor;
    let num_images = test_images.len();
    let mut correct = 0;
    let mut results: Vec<(usize, usize, usize, std::time::Duration)> = Vec::new();

    println!(
        "  {:>5}  {:>5}  {:>5}  {:>8}  {:>6}",
        "Image", "True", "Pred", "Time", "Match"
    );
    println!(
        "  {:>5}  {:>5}  {:>5}  {:>8}  {:>6}",
        "─────", "─────", "─────", "────────", "──────"
    );

    for (i, img) in test_images.iter().enumerate() {
        let t_image = Instant::now();

        // Scale pixels
        let input_scaled = img.scaled_pixels(scale);

        // Encrypt
        let input_pt = OpenFHEPlaintext::from_vec(&ctx, &input_scaled)?;
        let encrypted_input = OpenFHECiphertext::encrypt(&ctx, &kp, &input_pt)?;

        // Conv1 block
        // conv2d with integrated rescale ÷S: input(∝S) × kernel(∝S) / S → ∝S
        let x = encrypted_input.conv2d(&ctx, &kp, &w.conv1_kernel, 28, 28, 5, 5, scale as i64)?;
        let x = x.add(&ctx, &w.conv1_bias)?;
        // Square activation with integrated rescale: x²/S keeps values in range
        let x = x.square_activate(&ctx, &kp, scale as i64)?;
        let x = x.avgpool(&ctx, &kp, 24, 24, 2, 2)?;

        // Conv2 block
        let x = x.conv2d(&ctx, &kp, &w.conv2_kernel, 12, 12, 5, 5, scale as i64)?;
        let x = x.add(&ctx, &w.conv2_bias)?;
        let x = x.square_activate(&ctx, &kp, scale as i64)?;
        let x = x.avgpool(&ctx, &kp, 8, 8, 2, 2)?;

        // FC layer with integrated rescale ÷S
        let x = OpenFHECiphertext::matmul(&ctx, &kp, &w.fc_weights, &x, 10, 16, scale as i64)?;
        let x = x.add(&ctx, &w.fc_bias)?;

        // Decrypt and classify
        let output_pt = x.decrypt(&ctx, &kp)?;
        let logits = output_pt.to_vec()?;
        let class_logits: Vec<i64> = logits[..10].to_vec();

        let predicted = class_logits
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| v)
            .map(|(idx, _)| idx)
            .unwrap();

        let image_time = t_image.elapsed();
        let is_correct = predicted == img.label;
        if is_correct {
            correct += 1;
        }

        let marker = if is_correct { "✓" } else { "✗" };
        println!(
            "  {:>5}  {:>5}  {:>5}  {:>7.2?}  {:>6}",
            i, img.label, predicted, image_time, marker
        );

        results.push((img.label, predicted, i, image_time));
    }

    let total_inference = t_total.elapsed();

    // ================================================================
    // Phase 4: Summary
    // ================================================================
    println!("\n--- Phase 4: Results Summary ---\n");

    let accuracy = (correct as f64 / num_images as f64) * 100.0;

    println!("  Encrypted inference accuracy: {}/{} ({:.1}%)", correct, num_images, accuracy);
    println!("  Float model accuracy (train): {:.2}%", weights.config.float_accuracy);
    println!();

    // Show any misclassifications
    let mismatches: Vec<_> = results
        .iter()
        .filter(|(label, pred, _, _)| label != pred)
        .collect();

    if mismatches.is_empty() {
        println!("  All predictions match ground truth! ✓");
    } else {
        println!("  Misclassifications:");
        for (label, pred, idx, _) in &mismatches {
            println!("    Image {}: true={}, predicted={}", idx, label, pred);
        }
        println!();
        println!("  Note: Some misclassifications may occur due to:");
        println!("    - Quantization error (float → int rounding)");
        println!("    - Scale accumulation through layers");
        println!("    - Integer overflow in intermediate computations");
        println!("    - BFV modular arithmetic wrapping");
    }

    // Timing statistics
    let times: Vec<f64> = results.iter().map(|(_, _, _, d)| d.as_secs_f64()).collect();
    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("  Timing:");
    println!("    Setup:                {:.2?}", setup_time);
    println!("    Avg per image:        {:.3}s", avg_time);
    println!("    Min per image:        {:.3}s", min_time);
    println!("    Max per image:        {:.3}s", max_time);
    println!("    Total ({} images):    {:.2?}", num_images, total_inference);

    // ================================================================
    // Final report
    // ================================================================
    println!("\n================================================================");
    println!("Step 6d: Verification COMPLETE");
    println!("================================================================");
    println!("  Model:              HE_CNN (222 parameters)");
    println!("  Scheme:             BFV (p={}, depth={})", plaintext_modulus, mult_depth);
    println!("  Security:           128-bit");
    println!("  Scale factor:       {}", scale);
    println!("  Test images:        {}", num_images);
    println!(
        "  Encrypted accuracy: {}/{} ({:.1}%)",
        correct, num_images, accuracy
    );
    println!(
        "  Float accuracy:     {:.2}% (from training)",
        weights.config.float_accuracy
    );
    println!();

    if accuracy >= 80.0 {
        println!("  PASS: Encrypted inference matches expected accuracy range ✓");
    } else if accuracy >= 50.0 {
        println!("  WARN: Encrypted accuracy is lower than expected");
        println!("        Check scale factor and intermediate value ranges");
    } else {
        println!("  FAIL: Encrypted accuracy is too low");
        println!("        Likely integer overflow or modular wrapping issue");
    }

    println!();
    println!("  Pipeline verified:");
    println!("    [✓] Weight loading (CSV → MnistWeights)");
    println!("    [✓] Weight encoding (MnistWeights → OpenFHE types)");
    println!("    [✓] Input encryption (pixels → ciphertext)");
    println!("    [✓] Encrypted CNN inference (9 HE operations)");
    println!("    [✓] Output decryption (ciphertext → logits → argmax)");
    println!("    [✓] Accuracy verification (vs ground truth)");

    Ok(())
}
