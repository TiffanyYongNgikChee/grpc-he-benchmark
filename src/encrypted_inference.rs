//! Encrypted MNIST Inference Engine
//!
//! Reusable library module that runs the full HE-CNN pipeline on encrypted data.
//! Extracted from `examples/mnist_inference.rs` and `examples/mnist_verify.rs`
//! so that the gRPC server (and any other caller) can invoke encrypted inference
//! without duplicating the pipeline logic.
//!
//! # Architecture (matches Python HE_CNN)
//!
//! ```text
//! Input 28×28 → Conv1(5×5,÷S) → +bias → x²/S → AvgPool(2×2)
//!             → Conv2(5×5,÷S) → +bias → x²/S → AvgPool(2×2)
//!             → FC(16→10,÷S)  → +bias → decrypt → argmax
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use he_benchmark::encrypted_inference::EncryptedInferenceEngine;
//!
//! let engine = EncryptedInferenceEngine::new("mnist_training/weights")
//!     .expect("Failed to initialize engine");
//!
//! // 784 pixel values in 0-255 range
//! let pixels: Vec<i64> = vec![0; 784];
//! let result = engine.predict(&pixels).expect("Inference failed");
//!
//! println!("Predicted digit: {}", result.predicted_digit);
//! println!("Total time: {:.2?}ms", result.timing.total_ms);
//! ```

use std::time::Instant;

use crate::open_fhe_lib::{
    OpenFHECiphertext, OpenFHEContext, OpenFHEError, OpenFHEKeyPair, OpenFHEPlaintext,
};
use crate::weight_loader::{EncodedWeights, MnistWeights, WeightLoadError};

// ============================================================================
// Timing breakdown
// ============================================================================

/// Per-layer timing breakdown for a single encrypted inference.
///
/// All times are in milliseconds. The sum of individual layer times
/// may be slightly less than `total_ms` due to overhead between layers.
#[derive(Debug, Clone)]
pub struct InferenceTiming {
    /// Time to encrypt the input image (pixels → ciphertext)
    pub encryption_ms: f64,
    /// Conv1: 28×28 → 24×24 (5×5 kernel, ÷scale)
    pub conv1_ms: f64,
    /// Add Conv1 bias (broadcast to 576 slots)
    pub bias1_ms: f64,
    /// Square activation x²/scale after Conv1
    pub act1_ms: f64,
    /// AvgPool 2×2 (24×24 → 12×12)
    pub pool1_ms: f64,
    /// Conv2: 12×12 → 8×8 (5×5 kernel, ÷scale)
    pub conv2_ms: f64,
    /// Add Conv2 bias (broadcast to 64 slots)
    pub bias2_ms: f64,
    /// Square activation x²/scale after Conv2
    pub act2_ms: f64,
    /// AvgPool 2×2 (8×8 → 4×4)
    pub pool2_ms: f64,
    /// FC matmul: 16 → 10 (÷scale)
    pub fc_ms: f64,
    /// Add FC bias (10 values)
    pub bias_fc_ms: f64,
    /// Decrypt output ciphertext → plaintext logits
    pub decryption_ms: f64,
    /// Total wall-clock time (encrypt → decrypt)
    pub total_ms: f64,
}

// ============================================================================
// Inference result
// ============================================================================

/// Result of a single encrypted MNIST inference.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Predicted digit (0-9), the argmax of `logits`
    pub predicted_digit: usize,
    /// Raw output logits (10 values, one per digit class).
    /// These are integer values in the BFV plaintext space.
    /// The digit with the largest logit is the prediction.
    pub logits: Vec<i64>,
    /// Per-layer timing breakdown
    pub timing: InferenceTiming,
}

// ============================================================================
// Engine error type
// ============================================================================

/// Errors from the encrypted inference engine.
#[derive(Debug)]
pub enum InferenceError {
    /// Weight loading failed
    WeightLoad(WeightLoadError),
    /// OpenFHE operation failed
    OpenFHE(OpenFHEError),
    /// Invalid input (wrong pixel count, etc.)
    InvalidInput(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightLoad(e) => write!(f, "Weight loading error: {}", e),
            Self::OpenFHE(e) => write!(f, "OpenFHE error: {}", e),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<WeightLoadError> for InferenceError {
    fn from(e: WeightLoadError) -> Self {
        Self::WeightLoad(e)
    }
}

impl From<OpenFHEError> for InferenceError {
    fn from(e: OpenFHEError) -> Self {
        Self::OpenFHE(e)
    }
}

// ============================================================================
// Activation routing helper
// ============================================================================

/// Polynomial coefficients for each supported activation degree.
///
/// The coefficients are in highest-degree-first order (Horner's format)
/// and are integer-scaled by `coeff_scale`.
///
/// | Degree | Polynomial              | Coefficients (×scale) | coeff_scale |
/// |--------|-------------------------|-----------------------|-------------|
/// | 2      | x²                      | [1, 0, 0]             | 1           |
/// | 3      | 0.125x³ + 0.5x          | [125, 0, 500, 0]      | 1000        |
/// | 4      | 0.0625x⁴ + 0.25x² + 0.1 | [625, 0, 2500, 0, 1000] | 10000   |
///
/// Note: Since our CNN uses decrypt→compute→re-encrypt for activation,
/// the polynomial is evaluated in plaintext 64-bit integer space.
/// This means mult_depth stays at 6 regardless of polynomial degree.
fn apply_activation(
    x: OpenFHECiphertext,
    ctx: &OpenFHEContext,
    kp: &OpenFHEKeyPair,
    scale_factor: i64,
    activation_degree: u32,
) -> Result<OpenFHECiphertext, InferenceError> {
    match activation_degree {
        2 => {
            // x²/S — the original square activation (fastest path)
            Ok(x.square_activate(ctx, kp, scale_factor)?)
        }
        3 => {
            // 0.125x³ + 0.5x  →  coeffs=[125, 0, 500, 0], coeff_scale=1000
            let coeffs: &[i64] = &[125, 0, 500, 0];
            Ok(x.poly_activate(ctx, kp, 3, coeffs, 1000, scale_factor)?)
        }
        4 => {
            // 0.0625x⁴ + 0.25x² + 0.1  →  coeffs=[625, 0, 2500, 0, 1000], coeff_scale=10000
            let coeffs: &[i64] = &[625, 0, 2500, 0, 1000];
            Ok(x.poly_activate(ctx, kp, 4, coeffs, 10000, scale_factor)?)
        }
        _ => Err(InferenceError::InvalidInput(format!(
            "Unsupported activation_degree {}. Supported: 2, 3, 4",
            activation_degree
        ))),
    }
}

// ============================================================================
// Core inference function (stateless)
// ============================================================================

/// Run the full encrypted CNN inference pipeline on a single image.
///
/// This is the core function — it takes pre-initialized crypto objects
/// and pre-encoded weights, so there's no repeated setup cost.
///
/// # Arguments
/// * `ctx` - OpenFHE BFV context (plaintext_modulus from config, depth=6)
/// * `kp` - Key pair for encrypt/decrypt/re-encrypt
/// * `w` - Pre-encoded weights (from `MnistWeights::encode()`)
/// * `scaled_pixels` - 784 pre-scaled pixel values
///     (computed as `round(pixel_0_255 / 255.0 * scale_factor)`)
/// * `scale_factor` - The quantization scale factor (e.g., 1000)
/// * `activation_degree` - Polynomial degree for activation function (2, 3, or 4)
///
/// # Returns
/// `InferenceResult` with predicted digit, raw logits, and timing
///
/// # Pipeline
/// ```text
/// encrypt → Conv1(÷S) → +bias → poly_act → Pool(24→12)
///         → Conv2(÷S) → +bias → poly_act → Pool(8→4)
///         → FC(÷S)    → +bias → decrypt → argmax
/// ```
pub fn run_encrypted_inference(
    ctx: &OpenFHEContext,
    kp: &OpenFHEKeyPair,
    w: &EncodedWeights,
    scaled_pixels: &[i64],
    scale_factor: i64,
    activation_degree: u32,
) -> Result<InferenceResult, InferenceError> {
    if scaled_pixels.len() != 784 {
        return Err(InferenceError::InvalidInput(format!(
            "Expected 784 scaled pixels, got {}",
            scaled_pixels.len()
        )));
    }

    let t_total = Instant::now();

    // ---- Encrypt input ----
    let t = Instant::now();
    let input_pt = OpenFHEPlaintext::from_vec(ctx, scaled_pixels)?;
    let encrypted_input = OpenFHECiphertext::encrypt(ctx, kp, &input_pt)?;
    let encryption_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ---- Conv1 block ----
    // Conv1: 28×28 → 24×24, integrated rescale ÷S
    let t = Instant::now();
    let x = encrypted_input.conv2d(ctx, kp, &w.conv1_kernel, 28, 28, 5, 5, scale_factor)?;
    let conv1_ms = t.elapsed().as_secs_f64() * 1000.0;

    // + Conv1 bias
    let t = Instant::now();
    let x = x.add(ctx, &w.conv1_bias)?;
    let bias1_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Polynomial activation (degree-aware)
    let t = Instant::now();
    let x = apply_activation(x, ctx, kp, scale_factor, activation_degree)?;
    let act1_ms = t.elapsed().as_secs_f64() * 1000.0;

    // AvgPool 2×2 (24×24 → 12×12)
    let t = Instant::now();
    let x = x.avgpool(ctx, kp, 24, 24, 2, 2)?;
    let pool1_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ---- Conv2 block ----
    // Conv2: 12×12 → 8×8, integrated rescale ÷S
    let t = Instant::now();
    let x = x.conv2d(ctx, kp, &w.conv2_kernel, 12, 12, 5, 5, scale_factor)?;
    let conv2_ms = t.elapsed().as_secs_f64() * 1000.0;

    // + Conv2 bias
    let t = Instant::now();
    let x = x.add(ctx, &w.conv2_bias)?;
    let bias2_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Polynomial activation (degree-aware)
    let t = Instant::now();
    let x = apply_activation(x, ctx, kp, scale_factor, activation_degree)?;
    let act2_ms = t.elapsed().as_secs_f64() * 1000.0;

    // AvgPool 2×2 (8×8 → 4×4 = 16 features)
    let t = Instant::now();
    let x = x.avgpool(ctx, kp, 8, 8, 2, 2)?;
    let pool2_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ---- FC layer ----
    // Matmul: 16 → 10, integrated rescale ÷S
    let t = Instant::now();
    let x = OpenFHECiphertext::matmul(ctx, kp, &w.fc_weights, &x, 10, 16, scale_factor)?;
    let fc_ms = t.elapsed().as_secs_f64() * 1000.0;

    // + FC bias
    let t = Instant::now();
    let x = x.add(ctx, &w.fc_bias)?;
    let bias_fc_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ---- Decrypt & classify ----
    let t = Instant::now();
    let output_pt = x.decrypt(ctx, kp)?;
    let all_values = output_pt.to_vec()?;
    let decryption_ms = t.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    // Extract the 10 class logits
    let logits: Vec<i64> = all_values[..10].to_vec();

    // Argmax
    let predicted_digit = logits
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(InferenceResult {
        predicted_digit,
        logits,
        timing: InferenceTiming {
            encryption_ms,
            conv1_ms,
            bias1_ms,
            act1_ms,
            pool1_ms,
            conv2_ms,
            bias2_ms,
            act2_ms,
            pool2_ms,
            fc_ms,
            bias_fc_ms,
            decryption_ms,
            total_ms,
        },
    })
}

// ============================================================================
// Convenience: scale raw pixels
// ============================================================================

/// Scale raw 0-255 pixel values for BFV encryption.
///
/// Applies the same transform as PyTorch's `ToTensor()` + quantization:
///   `scaled = round(pixel / 255.0 * scale_factor)`
///
/// # Arguments
/// * `pixels_0_255` - 784 raw pixel values in 0-255 range
/// * `scale_factor` - Quantization scale (e.g., 1000)
///
/// # Returns
/// 784 scaled integer values ready for encryption
pub fn scale_pixels(pixels_0_255: &[i64], scale_factor: i64) -> Vec<i64> {
    pixels_0_255
        .iter()
        .map(|&p| ((p as f64 / 255.0) * scale_factor as f64).round() as i64)
        .collect()
}

// ============================================================================
// High-level engine (owns crypto context + weights)
// ============================================================================

/// Self-contained encrypted inference engine.
///
/// Holds the OpenFHE context, key pair, and pre-encoded weights.
/// Create once on server startup, then call `predict()` per request.
///
/// # Example
/// ```no_run
/// use he_benchmark::encrypted_inference::EncryptedInferenceEngine;
///
/// let engine = EncryptedInferenceEngine::new("mnist_training/weights").unwrap();
/// let pixels = vec![0i64; 784]; // 28×28 image, 0-255 range
/// let result = engine.predict(&pixels).unwrap();
/// println!("Predicted: {}", result.predicted_digit);
/// ```
pub struct EncryptedInferenceEngine {
    ctx: OpenFHEContext,
    kp: OpenFHEKeyPair,
    weights: EncodedWeights,
    scale_factor: i64,
    /// Security level used (0=128-bit, 1=192-bit, 2=256-bit)
    pub security_level: u32,
    /// Float model accuracy from training (for reference in responses)
    pub float_accuracy: f64,
    /// Polynomial activation degree (2=x², 3=cubic, 4=quartic)
    pub activation_degree: u32,
}

impl EncryptedInferenceEngine {
    /// Initialize the engine: load weights, create BFV context, encode weights.
    ///
    /// Uses 128-bit security (default). Do it once at startup (~2-5 seconds).
    ///
    /// # Arguments
    /// * `weights_dir` - Path to directory with CSV weight files + model_config.json
    ///                   (e.g., "mnist_training/weights")
    pub fn new(weights_dir: &str) -> Result<Self, InferenceError> {
        Self::new_with_security(weights_dir, 0)
    }

    /// Initialize the engine with a configurable security level.
    ///
    /// # Arguments
    /// * `weights_dir` - Path to directory with CSV weight files + model_config.json
    /// * `security_level` - 0=128-bit, 1=192-bit, 2=256-bit
    pub fn new_with_security(weights_dir: &str, security_level: u32) -> Result<Self, InferenceError> {
        let sec_label = match security_level {
            1 => "192-bit",
            2 => "256-bit",
            _ => "128-bit",
        };
        println!("EncryptedInferenceEngine: Initializing ({} security)...", sec_label);

        // Load quantized weights from CSV
        let mnist_weights = MnistWeights::load(weights_dir)?;
        let scale_factor = mnist_weights.config.scale_factor;
        let float_accuracy = mnist_weights.config.float_accuracy;
        let plaintext_modulus = mnist_weights.config.plaintext_modulus;
        let activation_degree = mnist_weights.config.activation_degree;

        let act_label = match activation_degree {
            3 => "degree-3 (0.125x³+0.5x)",
            4 => "degree-4 (0.0625x⁴+0.25x²+0.1)",
            _ => "degree-2 (x²)",
        };
        println!(
            "  Weights loaded: {} params, scale_factor={}, accuracy={:.2}%, activation={}",
            mnist_weights.total_params(),
            scale_factor,
            float_accuracy,
            act_label
        );

        // Create BFV context
        // Multiplicative depth needed for true FHE CNN:
        //   Conv1 (ct×pt) + x² (ct×ct) + Conv2 (ct×pt) + x² (ct×ct) + FC (ct×pt)
        //   = 5 multiplicative depths, plus margin for accumulation
        let mult_depth = 6;
        println!(
            "  Creating BFV context (p={}, depth={})...",
            plaintext_modulus, mult_depth
        );
        let ctx = OpenFHEContext::new_bfv_with_security(plaintext_modulus, mult_depth, security_level)?;
        let kp = OpenFHEKeyPair::generate(&ctx)?;
        println!("  Keypair generated ({} security, with rotation keys)", sec_label);

        // Encode weights as OpenFHE plaintexts/ciphertexts
        let weights = mnist_weights.encode(&ctx, &kp)?;

        println!("EncryptedInferenceEngine: Ready\n");

        Ok(Self {
            ctx,
            kp,
            weights,
            scale_factor,
            security_level,
            float_accuracy,
            activation_degree,
        })
    }

    /// Run encrypted inference on a single MNIST image.
    ///
    /// # Arguments
    /// * `pixels_0_255` - 784 pixel values in 0-255 range (28×28, row-major)
    ///
    /// # Returns
    /// `InferenceResult` with predicted digit, logits, and per-layer timing
    pub fn predict(&self, pixels_0_255: &[i64]) -> Result<InferenceResult, InferenceError> {
        if pixels_0_255.len() != 784 {
            return Err(InferenceError::InvalidInput(format!(
                "Expected 784 pixels (28×28), got {}",
                pixels_0_255.len()
            )));
        }

        // Scale pixels: round(pixel / 255 * scale_factor)
        let scaled = scale_pixels(pixels_0_255, self.scale_factor);

        // Run the pipeline
        run_encrypted_inference(&self.ctx, &self.kp, &self.weights, &scaled, self.scale_factor, self.activation_degree)
    }

    /// Run encrypted inference on pre-scaled pixel values.
    ///
    /// Use this if you've already called `scale_pixels()` yourself.
    ///
    /// # Arguments
    /// * `scaled_pixels` - 784 pre-scaled pixel values
    pub fn predict_scaled(&self, scaled_pixels: &[i64]) -> Result<InferenceResult, InferenceError> {
        run_encrypted_inference(
            &self.ctx,
            &self.kp,
            &self.weights,
            scaled_pixels,
            self.scale_factor,
            self.activation_degree,
        )
    }

    /// Get the scale factor used for quantization
    pub fn scale_factor(&self) -> i64 {
        self.scale_factor
    }

    /// Run encrypted inference with a progress callback after each layer.
    ///
    /// The callback receives `(layer_name, layer_ms, elapsed_ms)` after
    /// each layer completes. This enables real-time streaming to clients.
    pub fn predict_with_progress<F>(
        &self,
        pixels_0_255: &[i64],
        mut on_layer_done: F,
    ) -> Result<InferenceResult, InferenceError>
    where
        F: FnMut(&str, f64, f64),
    {
        if pixels_0_255.len() != 784 {
            return Err(InferenceError::InvalidInput(format!(
                "Expected 784 pixels (28×28), got {}",
                pixels_0_255.len()
            )));
        }

        let scaled = scale_pixels(pixels_0_255, self.scale_factor);
        run_encrypted_inference_streaming(
            &self.ctx,
            &self.kp,
            &self.weights,
            &scaled,
            self.scale_factor,
            self.activation_degree,
            &mut on_layer_done,
        )
    }
}

/// Run the full encrypted CNN inference pipeline with progress callbacks.
///
/// Calls `on_layer_done(layer_name, layer_ms, elapsed_ms)` after each layer.
pub fn run_encrypted_inference_streaming<F>(
    ctx: &OpenFHEContext,
    kp: &OpenFHEKeyPair,
    w: &EncodedWeights,
    scaled_pixels: &[i64],
    scale_factor: i64,
    activation_degree: u32,
    on_layer_done: &mut F,
) -> Result<InferenceResult, InferenceError>
where
    F: FnMut(&str, f64, f64),
{
    if scaled_pixels.len() != 784 {
        return Err(InferenceError::InvalidInput(format!(
            "Expected 784 scaled pixels, got {}",
            scaled_pixels.len()
        )));
    }

    let t_total = Instant::now();

    // ---- Encrypt input ----
    let t = Instant::now();
    let input_pt = OpenFHEPlaintext::from_vec(ctx, scaled_pixels)?;
    let encrypted_input = OpenFHECiphertext::encrypt(ctx, kp, &input_pt)?;
    let encryption_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("encrypt", encryption_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // ---- Conv1 ----
    let t = Instant::now();
    let x = encrypted_input.conv2d(ctx, kp, &w.conv1_kernel, 28, 28, 5, 5, scale_factor)?;
    let conv1_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("conv1", conv1_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // + Conv1 bias
    let t = Instant::now();
    let x = x.add(ctx, &w.conv1_bias)?;
    let bias1_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("bias1", bias1_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // Polynomial activation (degree-aware)
    let t = Instant::now();
    let x = apply_activation(x, ctx, kp, scale_factor, activation_degree)?;
    let act1_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("relu1", act1_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // AvgPool 2×2 (24×24 → 12×12)
    let t = Instant::now();
    let x = x.avgpool(ctx, kp, 24, 24, 2, 2)?;
    let pool1_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("pool1", pool1_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // ---- Conv2 ----
    let t = Instant::now();
    let x = x.conv2d(ctx, kp, &w.conv2_kernel, 12, 12, 5, 5, scale_factor)?;
    let conv2_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("conv2", conv2_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // + Conv2 bias
    let t = Instant::now();
    let x = x.add(ctx, &w.conv2_bias)?;
    let bias2_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("bias2", bias2_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // Polynomial activation (degree-aware)
    let t = Instant::now();
    let x = apply_activation(x, ctx, kp, scale_factor, activation_degree)?;
    let act2_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("relu2", act2_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // AvgPool 2×2 (8×8 → 4×4)
    let t = Instant::now();
    let x = x.avgpool(ctx, kp, 8, 8, 2, 2)?;
    let pool2_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("pool2", pool2_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // ---- FC layer ----
    let t = Instant::now();
    let x = OpenFHECiphertext::matmul(ctx, kp, &w.fc_weights, &x, 10, 16, scale_factor)?;
    let fc_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("fc", fc_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // + FC bias
    let t = Instant::now();
    let x = x.add(ctx, &w.fc_bias)?;
    let bias_fc_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("biasfc", bias_fc_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    // ---- Decrypt & classify ----
    let t = Instant::now();
    let output_pt = x.decrypt(ctx, kp)?;
    let all_values = output_pt.to_vec()?;
    let decryption_ms = t.elapsed().as_secs_f64() * 1000.0;
    on_layer_done("decrypt", decryption_ms, t_total.elapsed().as_secs_f64() * 1000.0);

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    let logits: Vec<i64> = all_values[..10].to_vec();
    let predicted_digit = logits
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(InferenceResult {
        predicted_digit,
        logits,
        timing: InferenceTiming {
            encryption_ms,
            conv1_ms,
            bias1_ms,
            act1_ms,
            pool1_ms,
            conv2_ms,
            bias2_ms,
            act2_ms,
            pool2_ms,
            fc_ms,
            bias_fc_ms,
            decryption_ms,
            total_ms,
        },
    })
}
