#ifndef OPENFHE_CNN_OPS_H
#define OPENFHE_CNN_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types (must match openfhe_wrapper.h)
typedef struct OpenFHEContext OpenFHEContext;
typedef struct OpenFHEKeyPair OpenFHEKeyPair;
typedef struct OpenFHECiphertext OpenFHECiphertext;
typedef struct OpenFHEPlaintext OpenFHEPlaintext;

// CNN Operations for Encrypted Neural Networks
/// Matrix multiplication for fully connected layers
/// Multiplies plaintext weight matrix with encrypted input vector
/// Uses decrypt→compute→re-encrypt approach (same as conv2d/avgpool)
/// @param ctx: OpenFHE context
/// @param keypair: Key pair for decrypt/re-encrypt of intermediate values
/// @param weights: Plaintext weight matrix (flattened row-major, size: rows × cols)
/// @param input: Encrypted input vector (size: cols)
/// @param rows: Number of output rows
/// @param cols: Number of input columns
/// @param divisor: Value to divide each output element by (e.g., scale_factor).
///                 Use 1 for no division.
/// @return Encrypted result vector (size: rows) or NULL on failure
OpenFHECiphertext* openfhe_matmul(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHEPlaintext* weights,
    OpenFHECiphertext* input,
    size_t rows,
    size_t cols,
    int64_t divisor
);

/// 2D convolution for CNN layers
/// Applies convolution filter to encrypted image using sliding window
/// @param ctx: OpenFHE context
/// @param keypair: Key pair for encrypt/decrypt of intermediate values
/// @param input: Encrypted input image (flattened, size: input_height × input_width)
/// @param kernel: Plaintext convolution kernel (flattened, size: kernel_height × kernel_width)
/// @param input_height: Height of input image
/// @param input_width: Width of input image
/// @param kernel_height: Height of convolution kernel
/// @param kernel_width: Width of convolution kernel
/// @param divisor: Value to divide each output element by (e.g., scale_factor).
///                 Use 1 for no division.
/// @return Encrypted feature map (size: out_height × out_width) or NULL on failure
///         where out_height = input_height - kernel_height + 1
///               out_width = input_width - kernel_width + 1
OpenFHECiphertext* openfhe_conv2d(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    OpenFHEPlaintext* kernel,
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width,
    int64_t divisor
);

/// Polynomial approximation of ReLU activation function
/// Computes ReLU(x) ≈ a*x^n + ... + b*x + c using polynomial approximation
/// @param ctx: OpenFHE context
/// @param input: Encrypted input values
/// @param degree: Polynomial degree (3, 5, or 7)
///                degree=3: Fast approximation (0.125*x³ + 0.5*x + 0.5)
///                degree=5: Better accuracy (not yet implemented)
///                degree=7: Best accuracy (not yet implemented)
/// @return Encrypted activated values or NULL on failure
/// WARNING: Uses homomorphic EvalMult, values must fit in plaintext modulus
OpenFHECiphertext* openfhe_poly_relu(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    int degree
);

/// Square activation with integrated rescale using decrypt→compute→re-encrypt
/// Computes f(x) = x² / divisor for each value. Avoids modular overflow by
/// computing in plaintext space (64-bit integers) before re-encrypting.
/// The divisor prevents the squared values from exceeding the plaintext modulus.
/// Preferred over openfhe_poly_relu for large intermediate values.
/// @param ctx: OpenFHE context
/// @param keypair: Key pair for decrypt/re-encrypt
/// @param input: Encrypted input values
/// @param divisor: Value to divide by after squaring (e.g., scale_factor)
/// @return Encrypted squared+rescaled values or NULL on failure
OpenFHECiphertext* openfhe_square_activate(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    int64_t divisor
);

/// Average pooling for downsampling feature maps
/// Performs pooling by averaging values in each pooling window
/// @param ctx: OpenFHE context
/// @param keypair: Key pair for encrypt/decrypt of intermediate values
/// @param input: Encrypted feature map (flattened, size: input_height × input_width)
/// @param input_height: Height of input feature map
/// @param input_width: Width of input feature map
/// @param pool_size: Size of pooling window (e.g., 2 for 2×2 pooling)
/// @param stride: Stride for pooling (typically same as pool_size)
/// @return Encrypted downsampled feature map or NULL on failure
OpenFHECiphertext* openfhe_avgpool(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    size_t input_height,
    size_t input_width,
    size_t pool_size,
    size_t stride
);

/// Rescale encrypted values by dividing by a divisor
/// Used after polynomial activation (x²) to prevent scale accumulation.
/// After x² with scale_factor S, values grow as S². Dividing by S
/// brings them back to the expected range.
/// Uses decrypt→divide→re-encrypt approach.
/// @param ctx: OpenFHE context
/// @param keypair: Key pair for decrypt/re-encrypt
/// @param input: Encrypted values to rescale
/// @param divisor: Value to divide by (e.g., scale_factor after x²)
/// @return Rescaled encrypted values or NULL on failure
OpenFHECiphertext* openfhe_rescale(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    int64_t divisor
);

// Error Handling
/// Get last error message from CNN operations
/// @return Error message string (valid until next CNN operation call)
const char* openfhe_cnn_get_last_error();

#ifdef __cplusplus
}
#endif

#endif // OPENFHE_CNN_OPS_H
