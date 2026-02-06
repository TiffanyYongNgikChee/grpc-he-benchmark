#ifndef OPENFHE_CNN_OPS_H
#define OPENFHE_CNN_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types (must match openfhe_wrapper.h)
typedef struct OpenFHEContext OpenFHEContext;
typedef struct OpenFHECiphertext OpenFHECiphertext;
typedef struct OpenFHEPlaintext OpenFHEPlaintext;

// ============================================
// CNN Operations for Encrypted Neural Networks
// ============================================

/// Matrix multiplication for fully connected layers
/// Multiplies plaintext weight matrix with encrypted input vector
/// @param ctx: OpenFHE context
/// @param weights: Plaintext weight matrix (flattened row-major, size: rows × cols)
/// @param input: Encrypted input vector (size: cols)
/// @param rows: Number of output rows
/// @param cols: Number of input columns
/// @return Encrypted result vector (size: rows) or NULL on failure
OpenFHECiphertext* openfhe_matmul(
    OpenFHEContext* ctx,
    OpenFHEPlaintext* weights,
    OpenFHECiphertext* input,
    size_t rows,
    size_t cols
);

/// 2D convolution for CNN layers
/// Applies convolution filter to encrypted image using sliding window
/// @param ctx: OpenFHE context
/// @param input: Encrypted input image (flattened, size: input_height × input_width)
/// @param kernel: Plaintext convolution kernel (flattened, size: kernel_height × kernel_width)
/// @param input_height: Height of input image
/// @param input_width: Width of input image
/// @param kernel_height: Height of convolution kernel
/// @param kernel_width: Width of convolution kernel
/// @return Encrypted feature map (size: out_height × out_width) or NULL on failure
///         where out_height = input_height - kernel_height + 1
///               out_width = input_width - kernel_width + 1
OpenFHECiphertext* openfhe_conv2d(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    OpenFHEPlaintext* kernel,
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width
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
OpenFHECiphertext* openfhe_poly_relu(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    int degree
);

/// Average pooling for downsampling feature maps
/// Performs pooling by averaging values in each pooling window
/// @param ctx: OpenFHE context
/// @param input: Encrypted feature map (flattened, size: input_height × input_width)
/// @param input_height: Height of input feature map
/// @param input_width: Width of input feature map
/// @param pool_size: Size of pooling window (e.g., 2 for 2×2 pooling)
/// @param stride: Stride for pooling (typically same as pool_size)
/// @return Encrypted downsampled feature map or NULL on failure
///         Output size: out_height × out_width
///         where out_height = (input_height - pool_size) / stride + 1
///               out_width = (input_width - pool_size) / stride + 1
OpenFHECiphertext* openfhe_avgpool(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    size_t input_height,
    size_t input_width,
    size_t pool_size,
    size_t stride
);

// Error Handling
/// Get last error message from CNN operations
/// @return Error message string (valid until next CNN operation call)
const char* openfhe_cnn_get_last_error();

#ifdef __cplusplus
}
#endif

#endif // OPENFHE_CNN_OPS_H
