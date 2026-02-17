/// Step 6a Verification: Test MNIST Weight Loader
///
/// This example loads the quantized weights from CSV files and verifies:
///   1. All 6 CSV files + model_config.json are readable
///   2. Each layer has the correct number of values
///   3. All values fit within BFV plaintext modulus (±3,670,016)
///   4. Scale factor and config metadata match expectations
///   5. Specific known values match the exported CSVs
///
/// Run with: cargo run --example test_weight_loader

use he_benchmark::weight_loader::MnistWeights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================");
    println!("Step 6a: MNIST Weight Loader Verification");
    println!("================================================================\n");

    // ---- Load weights ----
    let weights_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist_training/weights");
    println!("Loading from: {}\n", weights_dir);

    let weights = MnistWeights::load(weights_dir)?;

    // ---- Test 1: Shape verification ----
    println!("\n--- Test 1: Shape Verification ---");
    let checks = [
        ("conv1_weights", weights.conv1_weights.len(), 25),
        ("conv1_bias",    weights.conv1_bias.len(),     1),
        ("conv2_weights", weights.conv2_weights.len(), 25),
        ("conv2_bias",    weights.conv2_bias.len(),     1),
        ("fc_weights",    weights.fc_weights.len(),   160),
        ("fc_bias",       weights.fc_bias.len(),       10),
    ];

    let mut all_pass = true;
    for (name, actual, expected) in &checks {
        let ok = actual == expected;
        let marker = if ok { "✓" } else { "✗" };
        println!("  {} {}: {} values (expected {})", marker, name, actual, expected);
        if !ok { all_pass = false; }
    }

    assert!(all_pass, "Shape verification failed!");
    println!("  Total: {} parameters ✓", weights.total_params());

    // ---- Test 2: Config verification ----
    println!("\n--- Test 2: Config Verification ---");
    println!("  Model name:       {}", weights.config.model_name);
    println!("  Scale factor:     {}", weights.config.scale_factor);
    println!("  Plaintext modulus: {}", weights.config.plaintext_modulus);
    println!("  Float accuracy:   {:.2}%", weights.config.float_accuracy);
    println!("  Total parameters: {}", weights.config.total_parameters);
    println!("  Input shape:      {:?}", weights.config.input_shape);
    println!("  Output shape:     {:?}", weights.config.output_shape);

    assert_eq!(weights.config.model_name, "HE_CNN", "Model name mismatch");
    assert_eq!(weights.config.scale_factor, 1000, "Scale factor mismatch");
    assert_eq!(weights.config.plaintext_modulus, 7340033, "Plaintext modulus mismatch");
    assert_eq!(weights.config.total_parameters, 222, "Total params mismatch");
    assert_eq!(weights.config.input_shape, [1, 1, 28, 28], "Input shape mismatch");
    assert_eq!(weights.config.output_shape, [1, 10], "Output shape mismatch");
    assert!(weights.config.float_accuracy > 80.0, "Accuracy too low: {}", weights.config.float_accuracy);
    println!("  All config checks passed ✓");

    // ---- Test 3: Value range (fits in BFV modulus) ----
    println!("\n--- Test 3: Value Range Check ---");
    let max_half = (weights.config.plaintext_modulus / 2) as i64;

    let all_values: Vec<i64> = weights.conv1_weights.iter()
        .chain(weights.conv1_bias.iter())
        .chain(weights.conv2_weights.iter())
        .chain(weights.conv2_bias.iter())
        .chain(weights.fc_weights.iter())
        .chain(weights.fc_bias.iter())
        .copied()
        .collect();

    let max_abs = all_values.iter().map(|v| v.abs()).max().unwrap();
    let min_val = *all_values.iter().min().unwrap();
    let max_val = *all_values.iter().max().unwrap();

    println!("  Value range: [{}, {}]", min_val, max_val);
    println!("  Max |value|: {}", max_abs);
    println!("  BFV limit:   ±{}", max_half);
    assert!(max_abs <= max_half, "Values exceed plaintext modulus!");
    println!("  All values fit within modulus ✓");

    // ---- Test 4: Spot-check known values ----
    println!("\n--- Test 4: Spot-Check Known Values ---");

    // Conv1 first row (from CSV): -151,-198,16,114,115
    assert_eq!(weights.conv1_weights[0], -151, "conv1[0] mismatch");
    assert_eq!(weights.conv1_weights[1], -198, "conv1[1] mismatch");
    assert_eq!(weights.conv1_weights[2], 16,   "conv1[2] mismatch");
    assert_eq!(weights.conv1_weights[3], 114,  "conv1[3] mismatch");
    assert_eq!(weights.conv1_weights[4], 115,  "conv1[4] mismatch");
    println!("  Conv1 row 0: {:?} ✓", &weights.conv1_weights[0..5]);

    // Conv1 bias: 926
    assert_eq!(weights.conv1_bias[0], 926, "conv1 bias mismatch");
    println!("  Conv1 bias:  {:?} ✓", weights.conv1_bias);

    // Conv2 first row (from CSV): -310,438,51,-120,-349
    assert_eq!(weights.conv2_weights[0], -310, "conv2[0] mismatch");
    assert_eq!(weights.conv2_weights[1], 438,  "conv2[1] mismatch");
    println!("  Conv2 row 0: {:?} ✓", &weights.conv2_weights[0..5]);

    // Conv2 bias: 629
    assert_eq!(weights.conv2_bias[0], 629, "conv2 bias mismatch");
    println!("  Conv2 bias:  {:?} ✓", weights.conv2_bias);

    // FC bias (from CSV): -320,1405,-789,-713,734,244,618,466,-1505,-135
    assert_eq!(weights.fc_bias[0], -320,  "fc_bias[0] mismatch");
    assert_eq!(weights.fc_bias[1], 1405,  "fc_bias[1] mismatch");
    assert_eq!(weights.fc_bias[8], -1505, "fc_bias[8] mismatch");
    println!("  FC bias:     {:?} ✓", weights.fc_bias);

    // ---- Test 5: fc_weight_row accessor ----
    println!("\n--- Test 5: FC Row Accessor ---");
    for class_idx in 0..10 {
        let row = weights.fc_weight_row(class_idx);
        assert_eq!(row.len(), 16, "FC row {} wrong length", class_idx);
        // Verify it matches the flat slice
        assert_eq!(row, &weights.fc_weights[class_idx * 16..(class_idx + 1) * 16]);
    }
    // FC row 0 first value (from CSV): -34
    assert_eq!(weights.fc_weight_row(0)[0], -34, "fc_row[0][0] mismatch");
    println!("  All 10 FC rows verified (16 values each) ✓");

    // ---- Summary ----
    println!("\n================================================================");
    println!("Step 6a COMPLETE: Weight loader verified ✓");
    println!("================================================================");
    println!("  Files loaded:      6 CSVs + model_config.json");
    println!("  Parameters:        {}", weights.total_params());
    println!("  Scale factor:      {}", weights.scale_factor());
    println!("  Accuracy (float):  {:.2}%", weights.config.float_accuracy);
    println!("  All values in BFV range: ✓");
    println!("  Spot-checks passed: ✓");
    println!("  Ready for Step 6b (encode weights as OpenFHE plaintexts)");

    Ok(())
}
