/// Automated MNIST Benchmark: 100 images × configurable activation degree
///
/// Runs the full encrypted CNN inference pipeline on 100 test images
/// (10 per digit) and outputs detailed CSV results for dissertation analysis.
///
/// This binary runs DIRECTLY on the compute server (EC2 #2), bypassing
/// the gRPC/Spring Boot/frontend layers for maximum throughput.
///
/// # Usage
///
/// ```bash
/// # Default: use weights/ directory (deg2, 128-bit)
/// cargo run --release --example mnist_benchmark
///
/// # Specify weight directory explicitly (deg3):
/// cargo run --release --example mnist_benchmark -- --weights mnist_training/weights_deg3
///
/// # Specify output CSV path:
/// cargo run --release --example mnist_benchmark -- --output results_deg2.csv
///
/// # Full example:
/// cargo run --release --example mnist_benchmark -- \
///     --weights mnist_training/weights_deg2 \
///     --output mnist_training/fhe_benchmark_deg2.csv
/// ```
///
/// # Output CSV columns
///
/// image_index, true_label, predicted_digit, correct, encryption_ms, conv1_ms,
/// bias1_ms, act1_ms, pool1_ms, conv2_ms, bias2_ms, act2_ms, pool2_ms,
/// fc_ms, bias_fc_ms, decryption_ms, total_ms, logits

use he_benchmark::encrypted_inference::EncryptedInferenceEngine;
use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut weights_dir = String::from("mnist_training/weights");
    let mut output_path = String::new(); // auto-generated if empty

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "-w" => {
                i += 1;
                if i < args.len() {
                    weights_dir = args[i].clone();
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = args[i].clone();
                }
            }
            "--help" | "-h" => {
                println!("Usage: mnist_benchmark [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --weights, -w <DIR>   Weight directory (default: mnist_training/weights)");
                println!("  --output,  -o <FILE>  Output CSV path (default: auto-generated)");
                println!("  --help,    -h         Show this help");
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("================================================================");
    println!("  MNIST Encrypted Inference Benchmark");
    println!("================================================================\n");

    // ================================================================
    // Phase 1: Load test images
    // ================================================================
    let images_path = Path::new(&weights_dir).join("test_images_100.csv");
    let labels_path = Path::new(&weights_dir).join("test_labels_100.csv");

    if !images_path.exists() {
        eprintln!("ERROR: {} not found", images_path.display());
        eprintln!("Run: cd mnist_training && python export_test_images_100.py");
        std::process::exit(1);
    }

    println!("  Loading test images from: {}", images_path.display());
    let (images, labels) = load_test_data(&images_path, &labels_path)?;
    let num_images = images.len();
    println!("  Loaded {} images with {} labels\n", num_images, labels.len());

    // ================================================================
    // Phase 2: Initialize engine
    // ================================================================
    println!("  Initializing EncryptedInferenceEngine...");
    let t_init = Instant::now();
    let engine = EncryptedInferenceEngine::new(&weights_dir)?;
    let init_secs = t_init.elapsed().as_secs_f64();
    println!(
        "  Engine ready in {:.2}s (activation_degree={}, security=128-bit)\n",
        init_secs, engine.activation_degree
    );

    // Auto-generate output path if not specified
    if output_path.is_empty() {
        output_path = format!(
            "mnist_training/fhe_benchmark_deg{}_128bit.csv",
            engine.activation_degree
        );
    }

    // ================================================================
    // Phase 3: Run benchmark
    // ================================================================
    println!("  Starting benchmark: {} images...", num_images);
    println!("  Output: {}\n", output_path);

    // Open CSV writer
    let file = fs::File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "image_index,true_label,predicted_digit,correct,encryption_ms,conv1_ms,\
         bias1_ms,act1_ms,pool1_ms,conv2_ms,bias2_ms,act2_ms,pool2_ms,\
         fc_ms,bias_fc_ms,decryption_ms,total_ms,logits"
    )?;

    let mut correct_count = 0;
    let mut total_inference_ms = 0.0;
    let t_benchmark = Instant::now();

    for (idx, (pixels, &true_label)) in images.iter().zip(labels.iter()).enumerate() {
        let img_num = idx + 1;
        print!("  [{:3}/{}] Digit {} ... ", img_num, num_images, true_label);
        io::stdout().flush()?;

        let t_img = Instant::now();
        let result = engine.predict(pixels)?;
        let wall_ms = t_img.elapsed().as_secs_f64() * 1000.0;

        let is_correct = result.predicted_digit == true_label as usize;
        if is_correct {
            correct_count += 1;
        }
        total_inference_ms += result.timing.total_ms;

        // Print progress
        let status = if is_correct { "✓" } else { "✗" };
        println!(
            "predicted={} {} ({:.1}ms)",
            result.predicted_digit, status, wall_ms
        );

        // Write CSV row
        let logits_str: Vec<String> = result.logits.iter().map(|v| v.to_string()).collect();
        writeln!(
            writer,
            "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},\"[{}]\"",
            idx,
            true_label,
            result.predicted_digit,
            if is_correct { 1 } else { 0 },
            result.timing.encryption_ms,
            result.timing.conv1_ms,
            result.timing.bias1_ms,
            result.timing.act1_ms,
            result.timing.pool1_ms,
            result.timing.conv2_ms,
            result.timing.bias2_ms,
            result.timing.act2_ms,
            result.timing.pool2_ms,
            result.timing.fc_ms,
            result.timing.bias_fc_ms,
            result.timing.decryption_ms,
            result.timing.total_ms,
            logits_str.join(",")
        )?;
        writer.flush()?;
    }

    let benchmark_secs = t_benchmark.elapsed().as_secs_f64();

    // ================================================================
    // Phase 4: Summary
    // ================================================================
    let accuracy = correct_count as f64 / num_images as f64 * 100.0;
    let avg_ms = total_inference_ms / num_images as f64;

    println!("\n================================================================");
    println!("  Benchmark Results");
    println!("================================================================");
    println!("  Activation degree:   {}", engine.activation_degree);
    println!("  Security level:      128-bit");
    println!("  Images processed:    {}", num_images);
    println!("  Correct predictions: {}/{}", correct_count, num_images);
    println!("  Accuracy:            {:.1}%", accuracy);
    println!("  Average inference:   {:.1}ms ({:.2}s)", avg_ms, avg_ms / 1000.0);
    println!("  Total benchmark:     {:.1}s", benchmark_secs);
    println!("  Engine init:         {:.1}s", init_secs);
    println!("  Output saved to:     {}", output_path);
    println!("================================================================\n");

    // Also write a summary line to a separate file
    let summary_path = format!(
        "mnist_training/fhe_benchmark_summary_deg{}.txt",
        engine.activation_degree
    );
    let mut summary = fs::File::create(&summary_path)?;
    writeln!(summary, "MNIST Encrypted Inference Benchmark Summary")?;
    writeln!(summary, "============================================")?;
    writeln!(summary, "Activation degree:   {}", engine.activation_degree)?;
    writeln!(summary, "Security level:      128-bit")?;
    writeln!(summary, "Images:              {}", num_images)?;
    writeln!(summary, "Correct:             {}/{}", correct_count, num_images)?;
    writeln!(summary, "Accuracy:            {:.1}%", accuracy)?;
    writeln!(summary, "Avg inference (ms):  {:.1}", avg_ms)?;
    writeln!(summary, "Total benchmark (s): {:.1}", benchmark_secs)?;
    writeln!(summary, "Engine init (s):     {:.1}", init_secs)?;
    writeln!(summary, "Float model acc:     {:.2}%", engine.float_accuracy)?;
    println!("  Summary saved to: {}", summary_path);

    Ok(())
}

/// Load test images and labels from CSV files.
///
/// - `images_path`: CSV with N rows × 784 columns (0-255 pixel values)
/// - `labels_path`: CSV with N rows × 1 column (digit labels)
///
/// Returns (Vec<Vec<i64>>, Vec<i64>) — images and labels
fn load_test_data(
    images_path: &Path,
    labels_path: &Path,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), Box<dyn std::error::Error>> {
    // Load images
    let images_content = fs::read_to_string(images_path)?;
    let images: Vec<Vec<i64>> = images_content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap_or(0.0) as i64)
                .collect()
        })
        .collect();

    // Validate image dimensions
    for (i, img) in images.iter().enumerate() {
        if img.len() != 784 {
            return Err(format!(
                "Image {} has {} pixels, expected 784",
                i,
                img.len()
            )
            .into());
        }
    }

    // Load labels
    let labels_content = fs::read_to_string(labels_path)?;
    let labels: Vec<i64> = labels_content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap_or(0.0) as i64)
                .collect::<Vec<_>>()
        })
        .collect();

    if images.len() != labels.len() {
        return Err(format!(
            "Mismatch: {} images but {} labels",
            images.len(),
            labels.len()
        )
        .into());
    }

    Ok((images, labels))
}
