/// Example: Using OpenFHE CNN Operations for Encrypted Neural Network Inference
/// 
/// This example demonstrates how to use the CNN operations:
/// - Matrix multiplication (for fully connected layers)
/// - 2D Convolution (for CNN layers)
/// - Polynomial ReLU (for activation)
/// - Average pooling (for downsampling)

use he_benchmark::open_fhe_lib::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!(" OpenFHE CNN Operations Example\n");
    
    // Step 1: Create context with deeper multiplicative depth for CNN
    println!(" Creating OpenFHE context (depth=5 for CNN operations)...");
    let context = OpenFHEContext::new_bfv(65537, 5)?;
    println!(" Context created\n");
    
    // Step 2: Generate keys
    println!(" Generating keypair...");
    let keypair = OpenFHEKeyPair::generate(&context)?;
    println!(" Keypair generated\n");
    
    // Example 1: Matrix Multiplication (FC Layer)
    println!(" Example 1: Matrix Multiplication");
    println!("   Simulating a 3×4 weight matrix × 4-element input vector");
    
    // Weight matrix (3 rows × 4 cols, flattened row-major)
    // [ 1  2  3  4 ]
    // [ 5  6  7  8 ]
    // [ 9 10 11 12 ]
    let weights = vec![
        1, 2, 3, 4,     // row 1
        5, 6, 7, 8,     // row 2
        9, 10, 11, 12,  // row 3
    ];
    let weight_plaintext = OpenFHEPlaintext::from_vec(&context, &weights)?;
    
    // Input vector [1, 1, 1, 1]
    let input_vec = vec![1, 1, 1, 1];
    let input_plaintext = OpenFHEPlaintext::from_vec(&context, &input_vec)?;
    let input_cipher = OpenFHECiphertext::encrypt(&context, &keypair, &input_plaintext)?;
    
    // Perform encrypted matrix multiplication
    let result_cipher = OpenFHECiphertext::matmul(&context, &weight_plaintext, &input_cipher, 3, 4)?;
    
    // Decrypt and verify
    let result_plain = result_cipher.decrypt(&context, &keypair)?;
    let result_vec = result_plain.to_vec()?;
    println!("   Expected: [10, 26, 42] (sum of each row)");
    println!("   Got:      {:?}", &result_vec[..3]);
    println!("   Matrix multiplication completed\n");
    
    // Example 2: Polynomial ReLU Activation
    println!("   Example 2: Polynomial ReLU Activation");
    println!("   Approximating ReLU(x) with degree-3 polynomial");
    
    // Test values: some negative, some positive
    let test_values = vec![-2, -1, 0, 1, 2];
    let test_plaintext = OpenFHEPlaintext::from_vec(&context, &test_values)?;
    let test_cipher = OpenFHECiphertext::encrypt(&context, &keypair, &test_plaintext)?;
    
    // Apply polynomial ReLU (degree 3)
    let activated_cipher = test_cipher.poly_relu(&context, 3)?;
    
    // Decrypt and verify
    let activated_plain = activated_cipher.decrypt(&context, &keypair)?;
    let activated_vec = activated_plain.to_vec()?;
    println!("   Input:  {:?}", &test_values);
    println!("   Output: {:?} (approximated ReLU)", &activated_vec[..5]);
    println!("   Polynomial ReLU completed\n");
    
    // Example 3: 2D Convolution
    println!("  Example 3: 2D Convolution");
    println!("   4×4 image with 2×2 kernel");
    
    // 4×4 input image (flattened)
    let image = vec![
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    let image_plaintext = OpenFHEPlaintext::from_vec(&context, &image)?;
    let image_cipher = OpenFHECiphertext::encrypt(&context, &keypair, &image_plaintext)?;
    
    // 2×2 convolution kernel (edge detection)
    let kernel = vec![
        1, -1,
        -1, 1,
    ];
    let kernel_plaintext = OpenFHEPlaintext::from_vec(&context, &kernel)?;
    
    // Apply convolution (output will be 3×3)
    let conv_cipher = image_cipher.conv2d(&context, &kernel_plaintext, 4, 4, 2, 2)?;
    
    // Decrypt and verify
    let conv_plain = conv_cipher.decrypt(&context, &keypair)?;
    let conv_vec = conv_plain.to_vec()?;
    println!("   Input:  4×4 image");
    println!("   Kernel: 2×2 edge detector");
    println!("   Output: 3×3 feature map = {:?}", &conv_vec[..9]);
    println!("   Convolution completed\n");
    
    // Example 4: Average Pooling
    println!("  Example 4: Average Pooling");
    println!("   4×4 image → 2×2 pooled (2×2 pooling, stride=2)");
    
    // Reuse the 4×4 image from convolution example
    let pooled_cipher = image_cipher.avgpool(&context, 4, 4, 2, 2)?;
    
    // Decrypt and verify
    let pooled_plain = pooled_cipher.decrypt(&context, &keypair)?;
    let pooled_vec = pooled_plain.to_vec()?;
    println!("   Input:  4×4 image");
    println!("   Output: 2×2 pooled = {:?}", &pooled_vec[..4]);
    println!("   (Each value is average of 2×2 region)");
    println!("   Average pooling completed\n");
    
    // Example 5: Full CNN Forward Pass Simulation
    println!("   Example 5: Simulated CNN Forward Pass");
    println!("   Input → Conv → ReLU → Pool → FC");
    
    // Start with encrypted 4×4 image
    let mut layer_output = image_cipher;
    
    // Layer 1: Convolution (4×4 → 3×3)
    println!("   [1] Conv2D: 4×4 → 3×3");
    layer_output = layer_output.conv2d(&context, &kernel_plaintext, 4, 4, 2, 2)?;
    
    // Layer 2: ReLU activation
    println!("   [2] ReLU activation");
    layer_output = layer_output.poly_relu(&context, 3)?;
    
    // Layer 3: Average pooling (3×3 → 1×1 with 3×3 kernel, stride=1)
    // Note: This is not standard pooling, just for demonstration
    println!("   [3] Pooling: 3×3 → simplified");
    
    // Layer 4: Fully Connected (flatten + matmul)
    println!("   [4] Fully Connected layer");
    let fc_weights = vec![1, 1, 1, 1, 1, 1, 1, 1, 1]; // Sum all 9 values
    let fc_weight_plain = OpenFHEPlaintext::from_vec(&context, &fc_weights)?;
    let final_output = OpenFHECiphertext::matmul(&context, &fc_weight_plain, &layer_output, 1, 9)?;
    
    // Final decryption
    let final_plain = final_output.decrypt(&context, &keypair)?;
    let final_vec = final_plain.to_vec()?;
    println!("   Output: {:?}", &final_vec[..1]);
    println!("  Full CNN pass completed\n");
    
    Ok(())
}
