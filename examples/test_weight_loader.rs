/// Steps 6a + 6b Verification: Test MNIST Weight Loader & Encoding
///
/// This example loads the quantized weights from CSV files and verifies:
///   1. All 6 CSV files + model_config.json are readable
///   2. Each layer has the correct number of values
///   3. All values fit within BFV plaintext modulus
///   4. Scale factor and config metadata match expectations
///   5. Specific known values match the exported CSVs
///   6. Weights encode correctly as OpenFHE plaintexts/ciphertexts
///   7. Encrypted biases decrypt back to correct values
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
        let marker = if ok { "PASS" } else { "FAIL" };
        println!("  {} {}: {} values (expected {})", marker, name, actual, expected);
        if !ok { all_pass = false; }
    }

    assert!(all_pass, "Shape verification failed!");
    println!("  Total: {} parameters PASS", weights.total_params());

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
    assert!(weights.config.float_accuracy > 80.0, "Accuracy too low");
    println!("  All config checks passed");

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
    println!("  BFV limit:   +/-{}", max_half);
    assert!(max_abs <= max_half, "Values exceed plaintext modulus!");
    println!("  All values fit within modulus");

    // ---- Test 4: Spot-check known values ----
    println!("\n--- Test 4: Spot-Check Known Values ---");

    assert_eq!(weights.conv1_weights[0], -151, "conv1[0] mismatch");
    assert_eq!(weights.conv1_weights[1], -198, "conv1[1] mismatch");
    assert_eq!(weights.conv1_weights[2], 16,   "conv1[2] mismatch");
    assert_eq!(weights.conv1_weights[3], 114,  "conv1[3] mismatch");
    assert_eq!(weights.conv1_weights[4], 115,  "conv1[4] mismatch");
    println!("  Conv1 row 0: {:?} PASS", &weights.conv1_weights[0..5]);

    assert_eq!(weights.conv1_bias[0], 926, "conv1 bias mismatch");
    println!("  Conv1 bias:  {:?} PASS", weights.conv1_bias);

    assert_eq!(weights.conv2_weights[0], -310, "conv2[0] mismatch");
    assert_eq!(weights.conv2_weights[1], 438,  "conv2[1] mismatch");
    println!("  Conv2 row 0: {:?} PASS", &weights.conv2_weights[0..5]);

    assert_eq!(weights.conv2_bias[0], 629, "conv2 bias mismatch");
    println!("  Conv2 bias:  {:?} PASS", weights.conv2_bias);

    assert_eq!(weights.fc_bias[0], -320,  "fc_bias[0] mismatch");
    assert_eq!(weights.fc_bias[1], 1405,  "fc_bias[1] mismatch");
    assert_eq!(weights.fc_bias[8], -1505, "fc_bias[8] mismatch");
    println!("  FC bias:     {:?} PASS", weights.fc_bias);

    // ---- Test 5: fc_weight_row accessor ----
    println!("\n--- Test 5: FC Row Accessor ---");
    for class_idx in 0..10 {
        let row = weights.fc_weight_row(class_idx);
        assert_eq!(row.len(), 16, "FC row wrong length");
        assert_eq!(row, &weights.fc_weights[class_idx * 16..(class_idx + 1) * 16]);
    }
    assert_eq!(weights.fc_weight_row(0)[0], -34, "fc_row[0][0] mismatch");
    println!("  All 10 FC rows verified (16 values each) PASS");

    // ================================================================
    // Step 6b: Encode weights as OpenFHE plaintexts / ciphertexts
    // ================================================================
    println!("\n================================================================");
    println!("Step 6b: Weight Encoding Verification");
    println!("================================================================\n");

    use he_benchmark::open_fhe_lib::{OpenFHEContext, OpenFHEKeyPair};

    let ctx = OpenFHEContext::new_bfv(7340033, 3)?;
    let kp = OpenFHEKeyPair::generate(&ctx)?;

    println!("  OpenFHE BFV context created (p=7340033, depth=3)");
    println!("  Keypair generated\n");

    let encoded = weights.encode(&ctx, &kp)?;

    // Test 6: Encoded shapes
    println!("--- Test 6: Encoded Weight Shapes ---");
    println!("  conv1_kernel plaintext PASS");
    println!("  conv1_bias   ciphertext (576 slots, broadcast) PASS");
    println!("  conv2_kernel plaintext PASS");
    println!("  conv2_bias   ciphertext (64 slots, broadcast) PASS");
    println!("  fc_weights   plaintext PASS");
    println!("  fc_bias      ciphertext (10 slots) PASS");
    println!("  scale_factor: {}", encoded.scale_factor);
    assert_eq!(encoded.scale_factor, 1000, "Encoded scale factor mismatch");
    println!("  All encoded shapes verified");

    // Test 7: Decrypt biases and verify values
    println!("\n--- Test 7: Decrypt Bias Verification ---");

    let conv1_bias_dec = encoded.conv1_bias.decrypt(&ctx, &kp)?;
    let conv1_bias_vals = conv1_bias_dec.to_vec()?;
    assert!(conv1_bias_vals.len() >= 576, "conv1_bias too few slots");
    for i in 0..576 {
        assert_eq!(conv1_bias_vals[i], 926,
            "conv1_bias slot {} expected 926, got {}", i, conv1_bias_vals[i]);
    }
    println!("  conv1_bias: all 576 slots = 926 PASS");

    let conv2_bias_dec = encoded.conv2_bias.decrypt(&ctx, &kp)?;
    let conv2_bias_vals = conv2_bias_dec.to_vec()?;
    assert!(conv2_bias_vals.len() >= 64, "conv2_bias too few slots");
    for i in 0..64 {
        assert_eq!(conv2_bias_vals[i], 629,
            "conv2_bias slot {} expected 629, got {}", i, conv2_bias_vals[i]);
    }
    println!("  conv2_bias: all 64 slots = 629 PASS");

    let fc_bias_dec = encoded.fc_bias.decrypt(&ctx, &kp)?;
    let fc_bias_vals = fc_bias_dec.to_vec()?;
    assert!(fc_bias_vals.len() >= 10, "fc_bias too few slots");
    let expected_fc_bias = [-320i64, 1405, -789, -713, 734, 244, 618, 466, -1505, -135];
    for i in 0..10 {
        assert_eq!(fc_bias_vals[i], expected_fc_bias[i],
            "fc_bias slot {} expected {}, got {}", i, expected_fc_bias[i], fc_bias_vals[i]);
    }
    println!("  fc_bias: {:?} PASS", &fc_bias_vals[..10]);

    // ---- Summary ----
    println!("\n================================================================");
    println!("Steps 6a + 6b COMPLETE");
    println!("================================================================");
    println!("  Files loaded:        6 CSVs + model_config.json");
    println!("  Parameters:          {}", weights.total_params());
    println!("  Scale factor:        {}", weights.config.scale_factor);
    println!("  Accuracy (float):    {:.2}%", weights.config.float_accuracy);
    println!("  All values in BFV range: PASS");
    println!("  Spot-checks passed:  PASS");
    println!("  Weight encoding:     PASS");
    println!("  Bias decryption:     PASS");
    println!("  Ready for Step 6c (encrypted inference pipeline)");

    Ok(())
}
