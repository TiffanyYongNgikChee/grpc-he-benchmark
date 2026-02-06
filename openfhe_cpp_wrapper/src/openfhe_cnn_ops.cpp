// openfhe_cnn_ops.cpp
// CNN operations for OpenFHE: matmul, conv2d, poly_relu, avgpool

#include "../../openfhe_cpp_wrapper/include/openfhe_wrapper.h"
#include "openfhe/core/lattice/hal/lat-backend.h"
#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/encoding/plaintext.h"
#include "openfhe/pke/ciphertext.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

using namespace lbcrypto;

// Internal struct definitions (must match openfhe_wrapper.cpp)
struct OpenFHEContext {
    CryptoContext<DCRTPoly> cryptoContext;
};

struct OpenFHEKeyPair {
    KeyPair<DCRTPoly> keyPair;
    OpenFHEContext* ctx;
};

struct OpenFHEPlaintext {
    Plaintext plaintext;
};

struct OpenFHECiphertext {
    Ciphertext<DCRTPoly> ciphertext;
    OpenFHEContext* ctx;
};

// Error handling
static thread_local std::string cnn_last_error;

static void set_cnn_error(const std::string& error) {
    cnn_last_error = error;
}

extern "C" const char* openfhe_cnn_get_last_error() {
    return cnn_last_error.c_str();
}

// Matrix Multiplication (for FC layers)
// Multiplies plaintext weight matrix with encrypted input vector
// weights: plaintext matrix (rows × cols)
// input: encrypted vector (cols elements)
// Returns: encrypted vector (rows elements)
extern "C" OpenFHECiphertext* openfhe_matmul(
    OpenFHEContext* ctx,
    OpenFHEPlaintext* weights,
    OpenFHECiphertext* input,
    size_t rows,
    size_t cols
) {
    if (!ctx || !weights || !input) {
        set_cnn_error("Invalid parameters for matmul");
        return nullptr;
    }
    
    try {
        // Get the weights as a vector
        std::vector<int64_t> weight_vec = weights->plaintext->GetPackedValue();
        
        // For simplicity: weights stored as flattened matrix (row-major)
        // result[i] = sum(weights[i*cols + j] * input[j]) for j=0..cols-1
        
        // This is a simplified implementation
        // In production, you'd use SIMD packing and rotations for efficiency
        
        // For now, return encrypted result by multiplying input with first row
        // TODO: Implement full matrix-vector multiplication with rotations
        
        auto result_ct = ctx->cryptoContext->EvalMult(
            input->ciphertext,
            weights->plaintext
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = result_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Matrix multiplication failed: ") + e.what());
        return nullptr;
    }
}

// 2D Convolution (for CNN layers)
// Applies 2D convolution filter to encrypted image
// input: encrypted image (height × width)
// kernel: plaintext filter (kernel_h × kernel_w)
// Returns: encrypted feature map
extern "C" OpenFHECiphertext* openfhe_conv2d(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    OpenFHEPlaintext* kernel,
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width
) {
    if (!ctx || !input || !kernel) {
        set_cnn_error("Invalid parameters for conv2d");
        return nullptr;
    }
    
    try {
        // Simplified 2D convolution implementation
        // For production: use rotation-based convolution with SIMD packing
        
        // Output dimensions
        size_t out_height = input_height - kernel_height + 1;
        size_t out_width = input_width - kernel_width + 1;
        
        // For now, perform element-wise multiplication with kernel
        // This is a placeholder - full convolution requires rotations
        auto result_ct = ctx->cryptoContext->EvalMult(
            input->ciphertext,
            kernel->plaintext
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = result_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Conv2D failed: ") + e.what());
        return nullptr;
    }
}

// Polynomial ReLU Approximation
// Approximates ReLU using polynomial: a*x^3 + b*x + c
// degree: polynomial degree (3, 5, or 7)
extern "C" OpenFHECiphertext* openfhe_poly_relu(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    int degree
) {
    if (!ctx || !input) {
        set_cnn_error("Invalid parameters for poly_relu");
        return nullptr;
    }
    
    try {
        // Polynomial coefficients for degree-3 ReLU approximation
        // Approximates ReLU(x) ≈ 0.125*x^3 + 0.5*x + 0.5
        
        if (degree == 3) {
            // Compute x^2
            auto x_squared = ctx->cryptoContext->EvalMult(
                input->ciphertext,
                input->ciphertext
            );
            
            // Compute x^3 = x^2 * x
            auto x_cubed = ctx->cryptoContext->EvalMult(
                x_squared,
                input->ciphertext
            );
            
            // Create plaintext coefficients
            std::vector<int64_t> coeff_cubic(1, 125);  // 0.125 scaled by 1000
            std::vector<int64_t> coeff_linear(1, 500); // 0.5 scaled by 1000
            std::vector<int64_t> coeff_const(1, 500);  // 0.5 scaled by 1000
            
            auto pt_cubic = ctx->cryptoContext->MakePackedPlaintext(coeff_cubic);
            auto pt_linear = ctx->cryptoContext->MakePackedPlaintext(coeff_linear);
            auto pt_const = ctx->cryptoContext->MakePackedPlaintext(coeff_const);
            
            // Compute: 0.125*x^3
            auto term1 = ctx->cryptoContext->EvalMult(x_cubed, pt_cubic);
            
            // Compute: 0.5*x
            auto term2 = ctx->cryptoContext->EvalMult(input->ciphertext, pt_linear);
            
            // Add constant: 0.5
            auto result_ct = ctx->cryptoContext->EvalAdd(term1, term2);
            result_ct = ctx->cryptoContext->EvalAdd(result_ct, pt_const);
            
            OpenFHECiphertext* result = new OpenFHECiphertext();
            result->ciphertext = result_ct;
            result->ctx = ctx;
            
            set_cnn_error("");
            return result;
            
        } else {
            // For degree 5 and 7, add more polynomial terms
            // For now, fall back to degree-3
            set_cnn_error("Only degree-3 polynomial currently supported");
            return nullptr;
        }
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Polynomial ReLU failed: ") + e.what());
        return nullptr;
    }
}

// Average Pooling
// Performs average pooling (e.g., 2×2 pooling with stride 2)
// input: encrypted feature map (height × width)
// pool_size: size of pooling window (e.g., 2 for 2×2)
// stride: stride for pooling (typically same as pool_size)
extern "C" OpenFHECiphertext* openfhe_avgpool(
    OpenFHEContext* ctx,
    OpenFHECiphertext* input,
    size_t input_height,
    size_t input_width,
    size_t pool_size,
    size_t stride
) {
    if (!ctx || !input) {
        set_cnn_error("Invalid parameters for avgpool");
        return nullptr;
    }
    
    try {
        // Simplified average pooling implementation
        // For 2x2 pooling: sum 4 values and multiply by 0.25
        
        // Output dimensions
        size_t out_height = (input_height - pool_size) / stride + 1;
        size_t out_width = (input_width - pool_size) / stride + 1;
        
        // For 2x2 average pooling, multiply by 1/4 = 0.25
        if (pool_size == 2) {
            // Create plaintext for 0.25 (scaled appropriately)
            std::vector<int64_t> scale_factor(1, 250); // 0.25 scaled by 1000
            auto pt_scale = ctx->cryptoContext->MakePackedPlaintext(scale_factor);
            
            // Multiply input by scaling factor
            // Note: This is simplified - real pooling needs rotation and summation
            auto result_ct = ctx->cryptoContext->EvalMult(
                input->ciphertext,
                pt_scale
            );
            
            OpenFHECiphertext* result = new OpenFHECiphertext();
            result->ciphertext = result_ct;
            result->ctx = ctx;
            
            set_cnn_error("");
            return result;
        } else {
            set_cnn_error("Only 2x2 pooling currently supported");
            return nullptr;
        }
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Average pooling failed: ") + e.what());
        return nullptr;
    }
}

