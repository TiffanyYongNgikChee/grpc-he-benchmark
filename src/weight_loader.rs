//! MNIST Weight Loader for HE Inference
//!
//! Reads quantized integer weights from CSV files exported by the Python
//! training pipeline (`mnist_training/train_mnist.py`, Step 5b).
//!
//! # File Layout (in `mnist_training/weights/`)
//!
//! | File               | Shape   | Description                       |
//! |--------------------|---------|-----------------------------------|
//! | conv1_weights.csv  | 5×5     | Conv1 kernel (25 integers)        |
//! | conv1_bias.csv     | 1       | Conv1 bias                        |
//! | conv2_weights.csv  | 5×5     | Conv2 kernel (25 integers)        |
//! | conv2_bias.csv     | 1       | Conv2 bias                        |
//! | fc_weights.csv     | 10×16   | FC weight matrix (160 integers)   |
//! | fc_bias.csv        | 10      | FC bias vector                    |
//! | model_config.json  | —       | Architecture + quantization info  |
//!
//! # CSV Format
//! - Values are comma-separated integers (one row per line)
//! - Conv kernels: 5 rows × 5 columns
//! - FC weights: 10 rows × 16 columns (out_features × in_features)
//! - Biases: single row of comma-separated integers
//!
//! # Usage
//! ```no_run
//! use he_benchmark::weight_loader::MnistWeights;
//!
//! let weights = MnistWeights::load("mnist_training/weights")
//!     .expect("Failed to load weights");
//!
//! println!("Scale factor: {}", weights.config.scale_factor);
//! println!("Conv1 kernel: {:?}", weights.conv1_weights);
//! println!("FC bias: {:?}", weights.fc_bias);
//! ```

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur when loading MNIST weights
#[derive(Debug)]
pub enum WeightLoadError {
    /// File not found or unreadable
    Io(io::Error),
    /// CSV value is not a valid integer
    ParseInt(std::num::ParseIntError),
    /// JSON config is malformed
    ParseJson(String),
    /// Loaded data has wrong number of values
    ShapeMismatch {
        file: String,
        expected: usize,
        actual: usize,
    },
    /// Weights directory doesn't exist
    DirectoryNotFound(String),
}

impl std::fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::ParseInt(e) => write!(f, "Integer parse error: {}", e),
            Self::ParseJson(msg) => write!(f, "JSON parse error: {}", msg),
            Self::ShapeMismatch { file, expected, actual } => {
                write!(f, "Shape mismatch in {}: expected {} values, got {}", file, expected, actual)
            }
            Self::DirectoryNotFound(dir) => write!(f, "Weights directory not found: {}", dir),
        }
    }
}

impl std::error::Error for WeightLoadError {}

impl From<io::Error> for WeightLoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<std::num::ParseIntError> for WeightLoadError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::ParseInt(e)
    }
}

// ============================================================================
// Model configuration (parsed from model_config.json)
// ============================================================================

/// Configuration metadata from model_config.json
///
/// Contains quantization parameters and accuracy info from training.
/// Parsed manually (no serde dependency needed).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name (e.g., "HE_CNN")
    pub model_name: String,
    /// Quantization scale factor (e.g., 1000)
    /// Weights were multiplied by this before rounding to integers
    pub scale_factor: i64,
    /// BFV plaintext modulus (e.g., 7340033)
    pub plaintext_modulus: u64,
    /// Float model accuracy from training (e.g., 87.68)
    pub float_accuracy: f64,
    /// Total parameter count (e.g., 222)
    pub total_parameters: usize,
    /// Input image dimensions [batch, channels, height, width]
    pub input_shape: [usize; 4],
    /// Output dimensions [batch, classes]
    pub output_shape: [usize; 2],
}

// ============================================================================
// MNIST Weights
// ============================================================================

/// All quantized integer weights for the HE-compatible MNIST CNN.
///
/// Architecture:
///   Conv1 (5×5, 1→1) → x² → AvgPool(2×2)
///   Conv2 (5×5, 1→1) → x² → AvgPool(2×2)
///   Flatten(16) → FC(16→10)
///
/// All weights are stored as `Vec<i64>` in row-major order.
/// To use with OpenFHE, encode these into plaintexts and apply
/// via the corresponding HE operations (conv2d, matmul, etc.).
#[derive(Debug, Clone)]
pub struct MnistWeights {
    // ---- Layer weights ----

    /// Conv1 kernel: 5×5 = 25 integers (row-major)
    pub conv1_weights: Vec<i64>,
    /// Conv1 bias: 1 integer
    pub conv1_bias: Vec<i64>,

    /// Conv2 kernel: 5×5 = 25 integers (row-major)
    pub conv2_weights: Vec<i64>,
    /// Conv2 bias: 1 integer
    pub conv2_bias: Vec<i64>,

    /// FC weights: 10×16 = 160 integers (row-major, out_features × in_features)
    pub fc_weights: Vec<i64>,
    /// FC bias: 10 integers
    pub fc_bias: Vec<i64>,

    // ---- Metadata ----

    /// Model configuration from model_config.json
    pub config: ModelConfig,
}

impl MnistWeights {
    /// Load all weights from a directory of CSV files + model_config.json.
    ///
    /// # Arguments
    /// * `weights_dir` - Path to directory containing the CSV files
    ///                   (e.g., "mnist_training/weights")
    ///
    /// # Returns
    /// `MnistWeights` with all layer weights loaded and shape-verified
    ///
    /// # Errors
    /// - `DirectoryNotFound` if weights_dir doesn't exist
    /// - `Io` if any CSV file is missing or unreadable
    /// - `ParseInt` if CSV contains non-integer values
    /// - `ShapeMismatch` if a file has wrong number of values
    /// - `ParseJson` if model_config.json is malformed
    pub fn load(weights_dir: &str) -> Result<Self, WeightLoadError> {
        let dir = Path::new(weights_dir);
        if !dir.is_dir() {
            return Err(WeightLoadError::DirectoryNotFound(weights_dir.to_string()));
        }

        println!("Loading MNIST weights from: {}", weights_dir);

        // Load model config
        let config = Self::load_config(&dir.join("model_config.json"))?;
        println!("  Model: {} (scale_factor={}, accuracy={:.2}%)",
                 config.model_name, config.scale_factor, config.float_accuracy);

        // Load each layer's weights with expected shapes
        let conv1_weights = Self::load_csv(&dir.join("conv1_weights.csv"), "conv1_weights.csv", 25)?;
        let conv1_bias    = Self::load_csv(&dir.join("conv1_bias.csv"),    "conv1_bias.csv",    1)?;
        let conv2_weights = Self::load_csv(&dir.join("conv2_weights.csv"), "conv2_weights.csv", 25)?;
        let conv2_bias    = Self::load_csv(&dir.join("conv2_bias.csv"),    "conv2_bias.csv",    1)?;
        let fc_weights    = Self::load_csv(&dir.join("fc_weights.csv"),    "fc_weights.csv",    160)?;
        let fc_bias       = Self::load_csv(&dir.join("fc_bias.csv"),       "fc_bias.csv",       10)?;

        let total = conv1_weights.len() + conv1_bias.len()
                  + conv2_weights.len() + conv2_bias.len()
                  + fc_weights.len()    + fc_bias.len();

        println!("  Loaded {} parameters across 6 files", total);

        // Print per-layer summary
        println!("\n  {:<20} {:>8} {:>12} {:>12}", "Layer", "Count", "Min", "Max");
        println!("  {:<20} {:>8} {:>12} {:>12}", "─".repeat(20), "─".repeat(8), "─".repeat(12), "─".repeat(12));
        Self::print_layer_stats("Conv1 weights", &conv1_weights);
        Self::print_layer_stats("Conv1 bias",    &conv1_bias);
        Self::print_layer_stats("Conv2 weights", &conv2_weights);
        Self::print_layer_stats("Conv2 bias",    &conv2_bias);
        Self::print_layer_stats("FC weights",    &fc_weights);
        Self::print_layer_stats("FC bias",       &fc_bias);

        // Validate all values fit within plaintext modulus
        let max_half = (config.plaintext_modulus / 2) as i64;
        let all_values: Vec<&i64> = conv1_weights.iter()
            .chain(conv1_bias.iter())
            .chain(conv2_weights.iter())
            .chain(conv2_bias.iter())
            .chain(fc_weights.iter())
            .chain(fc_bias.iter())
            .collect();

        let max_abs = all_values.iter().map(|v| v.abs()).max().unwrap_or(0);
        if max_abs > max_half {
            println!("\n  ⚠ WARNING: max |weight| = {} exceeds ±{}", max_abs, max_half);
        } else {
            println!("\n  ✓ All weights fit within plaintext modulus ±{}", max_half);
        }

        Ok(MnistWeights {
            conv1_weights,
            conv1_bias,
            conv2_weights,
            conv2_bias,
            fc_weights,
            fc_bias,
            config,
        })
    }

    /// Load a CSV file and parse all values as i64.
    ///
    /// Handles multi-row CSVs (like 5×5 kernel) by reading line-by-line
    /// and splitting on commas. Verifies the total count matches `expected`.
    fn load_csv(path: &Path, filename: &str, expected: usize) -> Result<Vec<i64>, WeightLoadError> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);

        let mut values = Vec::with_capacity(expected);

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            for token in trimmed.split(',') {
                let token = token.trim();
                if !token.is_empty() {
                    let val: i64 = token.parse()?;
                    values.push(val);
                }
            }
        }

        if values.len() != expected {
            return Err(WeightLoadError::ShapeMismatch {
                file: filename.to_string(),
                expected,
                actual: values.len(),
            });
        }

        Ok(values)
    }

    /// Parse model_config.json without serde (manual JSON parsing).
    ///
    /// Extracts only the fields we need for inference:
    /// scale_factor, plaintext_modulus, model_name, accuracy, shapes.
    fn load_config(path: &Path) -> Result<ModelConfig, WeightLoadError> {
        let content = fs::read_to_string(path)?;

        // Simple JSON value extraction helpers
        let get_str = |key: &str| -> Result<String, WeightLoadError> {
            // Match "key": "value"
            let pattern = format!("\"{}\"", key);
            let pos = content.find(&pattern)
                .ok_or_else(|| WeightLoadError::ParseJson(format!("missing key: {}", key)))?;
            let after_key = &content[pos + pattern.len()..];
            // Skip `: "`
            let quote_start = after_key.find('"')
                .ok_or_else(|| WeightLoadError::ParseJson(format!("malformed value for: {}", key)))?;
            let rest = &after_key[quote_start + 1..];
            let quote_end = rest.find('"')
                .ok_or_else(|| WeightLoadError::ParseJson(format!("unterminated string for: {}", key)))?;
            Ok(rest[..quote_end].to_string())
        };

        let get_num = |key: &str| -> Result<f64, WeightLoadError> {
            let pattern = format!("\"{}\"", key);
            let pos = content.find(&pattern)
                .ok_or_else(|| WeightLoadError::ParseJson(format!("missing key: {}", key)))?;
            let after_key = &content[pos + pattern.len()..];
            // Skip `: `
            let colon = after_key.find(':')
                .ok_or_else(|| WeightLoadError::ParseJson(format!("missing colon for: {}", key)))?;
            let rest = after_key[colon + 1..].trim_start();
            // Read until comma, newline, or }
            let end = rest.find(|c: char| c == ',' || c == '\n' || c == '}' || c == ']')
                .unwrap_or(rest.len());
            let num_str = rest[..end].trim();
            num_str.parse::<f64>()
                .map_err(|_| WeightLoadError::ParseJson(format!("invalid number for {}: '{}'", key, num_str)))
        };

        let model_name = get_str("model_name")?;
        let scale_factor = get_num("scale_factor")? as i64;
        let plaintext_modulus = get_num("plaintext_modulus")? as u64;
        let float_accuracy = get_num("float_model")?;
        let total_parameters = get_num("total_parameters")? as usize;

        // Parse input_shape array [1, 1, 28, 28]
        let input_shape = Self::parse_shape_array(&content, "input_shape", 4)?;
        let output_shape = Self::parse_shape_array(&content, "output_shape", 2)?;

        Ok(ModelConfig {
            model_name,
            scale_factor,
            plaintext_modulus,
            float_accuracy,
            total_parameters,
            input_shape: [input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
            output_shape: [output_shape[0], output_shape[1]],
        })
    }

    /// Parse a JSON array of integers like `"key": [1, 1, 28, 28]`
    fn parse_shape_array(content: &str, key: &str, expected_len: usize) -> Result<Vec<usize>, WeightLoadError> {
        let pattern = format!("\"{}\"", key);
        let pos = content.find(&pattern)
            .ok_or_else(|| WeightLoadError::ParseJson(format!("missing key: {}", key)))?;
        let after_key = &content[pos + pattern.len()..];

        let bracket_start = after_key.find('[')
            .ok_or_else(|| WeightLoadError::ParseJson(format!("missing [ for: {}", key)))?;
        let bracket_end = after_key.find(']')
            .ok_or_else(|| WeightLoadError::ParseJson(format!("missing ] for: {}", key)))?;

        let inner = &after_key[bracket_start + 1..bracket_end];
        let values: Result<Vec<usize>, _> = inner
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect();

        let values = values
            .map_err(|_| WeightLoadError::ParseJson(format!("invalid array values for: {}", key)))?;

        if values.len() != expected_len {
            return Err(WeightLoadError::ParseJson(
                format!("{}: expected {} elements, got {}", key, expected_len, values.len())
            ));
        }

        Ok(values)
    }

    /// Print min/max/count stats for a layer's weights
    fn print_layer_stats(name: &str, values: &[i64]) {
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        println!("  {:<20} {:>8} {:>12} {:>12}", name, values.len(), min, max);
    }

    // ========================================================================
    // Accessor helpers (for Step 6b — encoding weights into OpenFHE plaintexts)
    // ========================================================================

    /// Get Conv1 kernel as a flat Vec (row-major 5×5 = 25 values).
    /// Ready to pass to `OpenFHEPlaintext::from_i64_vec()`.
    pub fn conv1_kernel(&self) -> &[i64] {
        &self.conv1_weights
    }

    /// Get Conv2 kernel as a flat Vec (row-major 5×5 = 25 values).
    pub fn conv2_kernel(&self) -> &[i64] {
        &self.conv2_weights
    }

    /// Get FC weight matrix as a flat Vec (row-major 10×16 = 160 values).
    /// Row i contains the 16 weights for output class i.
    pub fn fc_weight_matrix(&self) -> &[i64] {
        &self.fc_weights
    }

    /// Get one row of the FC weight matrix (weights for a single output class).
    ///
    /// # Arguments
    /// * `class_idx` - Output class index (0-9)
    ///
    /// # Returns
    /// Slice of 16 weights for the given class
    ///
    /// # Panics
    /// If `class_idx >= 10`
    pub fn fc_weight_row(&self, class_idx: usize) -> &[i64] {
        assert!(class_idx < 10, "class_idx must be 0-9, got {}", class_idx);
        let start = class_idx * 16;
        &self.fc_weights[start..start + 16]
    }

    /// Get the quantization scale factor
    pub fn scale_factor(&self) -> i64 {
        self.config.scale_factor
    }

    /// Total number of quantized parameters (should be 222)
    pub fn total_params(&self) -> usize {
        self.conv1_weights.len() + self.conv1_bias.len()
        + self.conv2_weights.len() + self.conv2_bias.len()
        + self.fc_weights.len() + self.fc_bias.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test loading from the actual exported weights directory
    #[test]
    fn test_load_weights() {
        let weights_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist_training/weights");

        // Skip test if weights haven't been exported yet
        if !Path::new(weights_dir).exists() {
            println!("Skipping test: weights directory not found (run train_mnist.py first)");
            return;
        }

        let weights = MnistWeights::load(weights_dir)
            .expect("Failed to load weights");

        // Verify shapes
        assert_eq!(weights.conv1_weights.len(), 25,  "Conv1 kernel should have 25 values");
        assert_eq!(weights.conv1_bias.len(),     1,  "Conv1 bias should have 1 value");
        assert_eq!(weights.conv2_weights.len(), 25,  "Conv2 kernel should have 25 values");
        assert_eq!(weights.conv2_bias.len(),     1,  "Conv2 bias should have 1 value");
        assert_eq!(weights.fc_weights.len(),   160,  "FC weights should have 160 values");
        assert_eq!(weights.fc_bias.len(),       10,  "FC bias should have 10 values");

        // Total parameters
        assert_eq!(weights.total_params(), 222, "Total should be 222 parameters");

        // Config sanity checks
        assert_eq!(weights.config.scale_factor, 1000);
        assert_eq!(weights.config.plaintext_modulus, 7340033);
        assert_eq!(weights.config.total_parameters, 222);
        assert_eq!(weights.config.input_shape, [1, 1, 28, 28]);
        assert_eq!(weights.config.output_shape, [1, 10]);
        assert!(weights.config.float_accuracy > 80.0, "Accuracy should be > 80%");

        // All values should fit within plaintext modulus
        let max_half = (weights.config.plaintext_modulus / 2) as i64;
        for val in weights.conv1_weights.iter()
            .chain(weights.conv1_bias.iter())
            .chain(weights.conv2_weights.iter())
            .chain(weights.conv2_bias.iter())
            .chain(weights.fc_weights.iter())
            .chain(weights.fc_bias.iter())
        {
            assert!(val.abs() <= max_half,
                    "Weight {} exceeds plaintext modulus ±{}", val, max_half);
        }

        println!("All weight loading tests passed ✓");
    }

    /// Test that fc_weight_row returns correct slices
    #[test]
    fn test_fc_weight_row() {
        let weights_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist_training/weights");
        if !Path::new(weights_dir).exists() {
            return;
        }

        let weights = MnistWeights::load(weights_dir).unwrap();

        for class_idx in 0..10 {
            let row = weights.fc_weight_row(class_idx);
            assert_eq!(row.len(), 16, "FC row {} should have 16 values", class_idx);

            // Verify it matches the corresponding slice of fc_weights
            let start = class_idx * 16;
            assert_eq!(row, &weights.fc_weights[start..start + 16]);
        }
    }

    /// Test error on missing directory
    #[test]
    fn test_missing_directory() {
        let result = MnistWeights::load("/nonexistent/path");
        assert!(result.is_err());
        match result.unwrap_err() {
            WeightLoadError::DirectoryNotFound(_) => {} // expected
            other => panic!("Expected DirectoryNotFound, got: {}", other),
        }
    }

    /// Test CSV parsing with known data
    #[test]
    fn test_parse_csv() {
        use std::io::Write;

        // Create a temp CSV
        let dir = std::env::temp_dir().join("he_test_weights");
        fs::create_dir_all(&dir).unwrap();

        let csv_path = dir.join("test.csv");
        let mut f = fs::File::create(&csv_path).unwrap();
        writeln!(f, "1,2,3").unwrap();
        writeln!(f, "4,5,6").unwrap();

        let values = MnistWeights::load_csv(&csv_path, "test.csv", 6).unwrap();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6]);

        // Test shape mismatch
        let result = MnistWeights::load_csv(&csv_path, "test.csv", 5);
        assert!(result.is_err());

        // Cleanup
        fs::remove_dir_all(&dir).ok();
    }
}
