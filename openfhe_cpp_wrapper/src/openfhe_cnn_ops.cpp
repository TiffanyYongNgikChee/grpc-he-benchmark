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
        // Get the weights as a vector (flattened row-major matrix)
        std::vector<int64_t> weight_vec = weights->plaintext->GetPackedValue();
        
        // Verify dimensions
        if (weight_vec.size() < rows * cols) {
            set_cnn_error("Weight vector size mismatch");
            return nullptr;
        }
        
        // Matrix-vector multiplication: result[i] = sum(W[i,j] * input[j])
        // Strategy: For each output row, create a plaintext with that row's weights
        // and multiply with rotated input vector
        
        // Get slot count for proper packing
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        
        // Result accumulator - start with zeros
        std::vector<int64_t> zero_vec(slot_count, 0);
        auto zero_pt = ctx->cryptoContext->MakePackedPlaintext(zero_vec);
        auto result_ct = ctx->cryptoContext->Encrypt(
            ctx->cryptoContext->KeyGen().publicKey, // Temporary - need proper key
            zero_pt
        );
        
        // For each output row
        for (size_t i = 0; i < rows && i < slot_count; i++) {
            // Extract weights for this row
            std::vector<int64_t> row_weights(slot_count, 0);
            for (size_t j = 0; j < cols && j < slot_count; j++) {
                row_weights[j] = weight_vec[i * cols + j];
            }
            
            // Create plaintext from row weights
            auto row_pt = ctx->cryptoContext->MakePackedPlaintext(row_weights);
            
            // Multiply encrypted input with this row's weights
            auto mult_result = ctx->cryptoContext->EvalMult(input->ciphertext, row_pt);
            
            // Sum all elements in the slot (this gives dot product)
            // Note: Without EvalSum, we approximate by keeping element-wise products
            // TODO: Enable rotation keys and use EvalSum for proper reduction
            
            // Add to result
            result_ct = ctx->cryptoContext->EvalAdd(result_ct, mult_result);
        }
        
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
// input: encrypted image (height × width) packed in slots
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
        // Output dimensions (valid convolution)
        size_t out_height = input_height - kernel_height + 1;
        size_t out_width = input_width - kernel_width + 1;
        
        // Get kernel weights
        std::vector<int64_t> kernel_vec = kernel->plaintext->GetPackedValue();
        
        if (kernel_vec.size() < kernel_height * kernel_width) {
            set_cnn_error("Kernel size mismatch");
            return nullptr;
        }
        
        // Sliding window convolution using rotation
        // For each position in the output feature map
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        
        // Initialize result with zeros
        std::vector<int64_t> zero_vec(slot_count, 0);
        auto zero_pt = ctx->cryptoContext->MakePackedPlaintext(zero_vec);
        auto result_ct = ctx->cryptoContext->Encrypt(
            ctx->cryptoContext->KeyGen().publicKey,
            zero_pt
        );
        
        // Iterate over kernel window
        for (size_t kh = 0; kh < kernel_height; kh++) {
            for (size_t kw = 0; kw < kernel_width; kw++) {
                // Get kernel weight at this position
                int64_t kernel_weight = kernel_vec[kh * kernel_width + kw];
                
                if (kernel_weight == 0) continue; // Skip zero weights
                
                // Create plaintext mask for this kernel position
                // Mask aligns kernel weight with corresponding input positions
                std::vector<int64_t> mask(slot_count, 0);
                
                // For each output position
                for (size_t oh = 0; oh < out_height; oh++) {
                    for (size_t ow = 0; ow < out_width; ow++) {
                        // Calculate input position
                        size_t ih = oh + kh;
                        size_t iw = ow + kw;
                        size_t input_idx = ih * input_width + iw;
                        size_t output_idx = oh * out_width + ow;
                        
                        if (input_idx < slot_count && output_idx < slot_count) {
                            mask[input_idx] = kernel_weight;
                        }
                    }
                }
                
                // Create plaintext from mask
                auto mask_pt = ctx->cryptoContext->MakePackedPlaintext(mask);
                
                // Multiply input with masked kernel weight
                auto weighted = ctx->cryptoContext->EvalMult(input->ciphertext, mask_pt);
                
                // Accumulate into result
                // Note: Proper implementation would rotate and sum here
                result_ct = ctx->cryptoContext->EvalAdd(result_ct, weighted);
            }
        }
        
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
        // Output dimensions
        size_t out_height = (input_height - pool_size) / stride + 1;
        size_t out_width = (input_width - pool_size) / stride + 1;
        
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        
        // Initialize result with zeros
        std::vector<int64_t> zero_vec(slot_count, 0);
        auto zero_pt = ctx->cryptoContext->MakePackedPlaintext(zero_vec);
        auto result_ct = ctx->cryptoContext->Encrypt(
            ctx->cryptoContext->KeyGen().publicKey,
            zero_pt
        );
        
        // For each pooling window
        for (size_t ph = 0; ph < pool_size; ph++) {
            for (size_t pw = 0; pw < pool_size; pw++) {
                // Create mask for this position in pooling window
                std::vector<int64_t> mask(slot_count, 0);
                
                // For each output position
                for (size_t oh = 0; oh < out_height; oh++) {
                    for (size_t ow = 0; ow < out_width; ow++) {
                        // Calculate input position
                        size_t ih = oh * stride + ph;
                        size_t iw = ow * stride + pw;
                        
                        size_t input_idx = ih * input_width + iw;
                        size_t output_idx = oh * out_width + ow;
                        
                        // Mark this position for summation
                        if (input_idx < slot_count && output_idx < slot_count) {
                            mask[input_idx] = 1;
                        }
                    }
                }
                
                // Create plaintext mask
                auto mask_pt = ctx->cryptoContext->MakePackedPlaintext(mask);
                
                // Multiply input by mask to select pooling region
                auto masked = ctx->cryptoContext->EvalMult(input->ciphertext, mask_pt);
                
                // Accumulate (sum all values in pooling window)
                result_ct = ctx->cryptoContext->EvalAdd(result_ct, masked);
            }
        }
        
        // Divide by pool_size² to get average
        // Scale factor: 1 / (pool_size * pool_size)
        double scale = 1.0 / (pool_size * pool_size);
        
        // Convert to integer representation (scale by 1000 for precision)
        int64_t scale_int = static_cast<int64_t>(scale * 1000);
        
        std::vector<int64_t> scale_vec(slot_count, scale_int);
        auto scale_pt = ctx->cryptoContext->MakePackedPlaintext(scale_vec);
        
        // Multiply result by scale factor
        result_ct = ctx->cryptoContext->EvalMult(result_ct, scale_pt);
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = result_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Average pooling failed: ") + e.what());
        return nullptr;
    }
}

