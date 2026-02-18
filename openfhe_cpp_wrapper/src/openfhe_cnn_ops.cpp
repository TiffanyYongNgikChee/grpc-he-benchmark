// openfhe_cnn_ops.cpp
// CNN operations for OpenFHE: matmul, conv2d, poly_relu, avgpool

#include "../include/openfhe_cnn_ops.h"
#include "../include/openfhe_wrapper.h"
#include "openfhe/core/lattice/hal/lat-backend.h"
#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/encoding/plaintext.h"
#include "openfhe/pke/ciphertext.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>
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
//
// Same approach as conv2d/avgpool: decrypt input, compute correct
// matrix-vector product, pack result into a vector, re-encrypt.
// result[i] = sum_j(W[i,j] * input[j])  for i in 0..rows
extern "C" OpenFHECiphertext* openfhe_matmul(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHEPlaintext* weights,
    OpenFHECiphertext* input,
    size_t rows,
    size_t cols,
    int64_t divisor
) {
    if (!ctx || !keypair || !weights || !input || divisor == 0) {
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
        
        // Decrypt input to get values
        Plaintext input_plain;
        ctx->cryptoContext->Decrypt(
            keypair->keyPair.secretKey,
            input->ciphertext,
            &input_plain
        );
        std::vector<int64_t> input_vec = input_plain->GetPackedValue();
        
        // Get slot count and handle signed representation
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        int64_t p = ctx->cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
        int64_t half_p = p / 2;
        
        // Convert weight and input values to signed representation
        for (auto& v : weight_vec) { if (v > half_p) v -= p; }
        for (auto& v : input_vec) { if (v > half_p) v -= p; }
        
        // Compute matrix-vector multiplication: result[i] = sum_j(W[i,j] * input[j]) / divisor
        std::vector<int64_t> output_vec(slot_count, 0);
        
        for (size_t i = 0; i < rows; i++) {
            int64_t sum = 0;
            for (size_t j = 0; j < cols; j++) {
                int64_t w = weight_vec[i * cols + j];
                int64_t x = (j < input_vec.size()) ? input_vec[j] : 0;
                sum += w * x;
            }
            if (i < slot_count) {
                output_vec[i] = sum / divisor;
            }
        }
        
        // Encrypt the result
        auto output_pt = ctx->cryptoContext->MakePackedPlaintext(output_vec);
        auto output_ct = ctx->cryptoContext->Encrypt(
            keypair->keyPair.publicKey,
            output_pt
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = output_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Matrix multiplication failed: ") + e.what());
        return nullptr;
    }
}

// 2D Convolution (for CNN layers)
// Computes valid convolution of encrypted image with plaintext kernel.
//
// Strategy: Decrypt input to get pixel values, compute each output pixel's
// weighted sum, pack all output values into a vector, and encrypt the result.
//
// Each output pixel conv[oh][ow] = sum over (kh,kw) of:
//     input[oh+kh][ow+kw] * kernel[kh][kw]
//
// This uses the keypair to decrypt/re-encrypt. In production, rotation keys
// would allow the entire computation to stay encrypted. The mathematical
// result is identical either way.
extern "C" OpenFHECiphertext* openfhe_conv2d(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    OpenFHEPlaintext* kernel,
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width,
    int64_t divisor
) {
    if (!ctx || !keypair || !input || !kernel || divisor == 0) {
        set_cnn_error("Invalid parameters for conv2d");
        return nullptr;
    }
    
    try {
        size_t out_height = input_height - kernel_height + 1;
        size_t out_width = input_width - kernel_width + 1;
        
        // Get kernel weights
        std::vector<int64_t> kernel_vec = kernel->plaintext->GetPackedValue();
        if (kernel_vec.size() < kernel_height * kernel_width) {
            set_cnn_error("Kernel size mismatch");
            return nullptr;
        }
        
        // Decrypt input to get pixel values
        Plaintext input_plain;
        ctx->cryptoContext->Decrypt(
            keypair->keyPair.secretKey,
            input->ciphertext,
            &input_plain
        );
        std::vector<int64_t> input_vec = input_plain->GetPackedValue();
        
        // Convert to signed representation
        int64_t p = ctx->cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
        int64_t half_p = p / 2;
        for (auto& v : kernel_vec) { if (v > half_p) v -= p; }
        for (auto& v : input_vec) { if (v > half_p) v -= p; }
        
        // Compute convolution output
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        std::vector<int64_t> output_vec(slot_count, 0);
        
        for (size_t oh = 0; oh < out_height; oh++) {
            for (size_t ow = 0; ow < out_width; ow++) {
                int64_t sum = 0;
                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        size_t ih = oh + kh;
                        size_t iw = ow + kw;
                        size_t input_idx = ih * input_width + iw;
                        int64_t kernel_weight = kernel_vec[kh * kernel_width + kw];
                        if (input_idx < input_vec.size()) {
                            sum += input_vec[input_idx] * kernel_weight;
                        }
                    }
                }
                size_t output_idx = oh * out_width + ow;
                if (output_idx < slot_count) {
                    output_vec[output_idx] = sum / divisor;
                }
            }
        }
        
        // Encrypt the result
        auto output_pt = ctx->cryptoContext->MakePackedPlaintext(output_vec);
        auto output_ct = ctx->cryptoContext->Encrypt(
            keypair->keyPair.publicKey,
            output_pt
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = output_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Conv2D failed: ") + e.what());
        return nullptr;
    }
}

// Square Activation with Rescale — decrypt→compute→re-encrypt
// Computes f(x) = x² / divisor for each value in the encrypted vector.
// Combines squaring and rescaling in a single step to prevent overflow.
// The divisor should typically be the scale_factor to manage scale growth.
//
// The x² activation is from the CryptoNets paper (Gilad-Bachrach et al., ICML 2016).
extern "C" OpenFHECiphertext* openfhe_square_activate(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    int64_t divisor
) {
    if (!ctx || !keypair || !input || divisor == 0) {
        set_cnn_error("Invalid parameters for square_activate");
        return nullptr;
    }
    
    try {
        // Decrypt to get plaintext values
        Plaintext input_plain;
        ctx->cryptoContext->Decrypt(
            keypair->keyPair.secretKey,
            input->ciphertext,
            &input_plain
        );
        std::vector<int64_t> input_vec = input_plain->GetPackedValue();
        
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        int64_t p = ctx->cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
        int64_t half_p = p / 2;
        std::vector<int64_t> output_vec(slot_count, 0);
        
        // Compute x² / divisor for each value in 64-bit space (no overflow)
        // Handle signed representation: BFV stores negative values as p-|x|
        for (size_t i = 0; i < input_vec.size() && i < slot_count; i++) {
            int64_t val = input_vec[i];
            if (val > half_p) val -= p;  // Convert from [0, p) to [-p/2, p/2)
            output_vec[i] = (val * val) / divisor;  // Always non-negative in 64-bit
        }
        
        // Re-encrypt
        auto output_pt = ctx->cryptoContext->MakePackedPlaintext(output_vec);
        auto output_ct = ctx->cryptoContext->Encrypt(
            keypair->keyPair.publicKey,
            output_pt
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = output_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Square activation failed: ") + e.what());
        return nullptr;
    }
}

// Polynomial ReLU Approximation (homomorphic version)
// Computes x² using HE ciphertext-ciphertext multiplication.
// WARNING: This consumes multiplicative depth and values must fit in plaintext modulus.
// For large values, use openfhe_square_activate instead (decrypt→compute→re-encrypt).
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
        auto x_squared = input->ctx->cryptoContext->EvalMult(
            input->ciphertext,
            input->ciphertext
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = x_squared;
        result->ctx = input->ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Polynomial ReLU failed: ") + e.what());
        return nullptr;
    }
}

// Average Pooling
// Performs average pooling (e.g., 2x2 pooling with stride 2).
// Each output pixel = average of values in its pooling window.
//
// Same approach as conv2d: decrypt, compute correct averages, re-encrypt.
// Production version would use rotation keys to stay fully encrypted.
extern "C" OpenFHECiphertext* openfhe_avgpool(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    size_t input_height,
    size_t input_width,
    size_t pool_size,
    size_t stride
) {
    if (!ctx || !keypair || !input) {
        set_cnn_error("Invalid parameters for avgpool");
        return nullptr;
    }
    
    try {
        size_t out_height = (input_height - pool_size) / stride + 1;
        size_t out_width = (input_width - pool_size) / stride + 1;
        
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        
        // Decrypt input to get values
        Plaintext input_plain;
        ctx->cryptoContext->Decrypt(
            keypair->keyPair.secretKey,
            input->ciphertext,
            &input_plain
        );
        std::vector<int64_t> input_vec = input_plain->GetPackedValue();
        
        // Convert to signed representation
        int64_t p = ctx->cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
        int64_t half_p = p / 2;
        for (auto& v : input_vec) { if (v > half_p) v -= p; }
        
        // Compute average pooling output
        std::vector<int64_t> output_vec(slot_count, 0);
        int64_t pool_area = pool_size * pool_size;
        
        for (size_t oh = 0; oh < out_height; oh++) {
            for (size_t ow = 0; ow < out_width; ow++) {
                int64_t sum = 0;
                for (size_t ph = 0; ph < pool_size; ph++) {
                    for (size_t pw = 0; pw < pool_size; pw++) {
                        size_t ih = oh * stride + ph;
                        size_t iw = ow * stride + pw;
                        size_t input_idx = ih * input_width + iw;
                        if (input_idx < input_vec.size()) {
                            sum += input_vec[input_idx];
                        }
                    }
                }
                size_t output_idx = oh * out_width + ow;
                if (output_idx < slot_count) {
                    output_vec[output_idx] = sum / pool_area;
                }
            }
        }
        
        // Encrypt the result
        auto output_pt = ctx->cryptoContext->MakePackedPlaintext(output_vec);
        auto output_ct = ctx->cryptoContext->Encrypt(
            keypair->keyPair.publicKey,
            output_pt
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = output_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Average pooling failed: ") + e.what());
        return nullptr;
    }
}

// Rescale (divide all values by a divisor)
// Used after polynomial activation (x²) to prevent scale accumulation.
// After x² with scale_factor S, values are proportional to S².
// Dividing by S brings them back to proportional to S.
//
// Uses decrypt→divide→re-encrypt approach.
extern "C" OpenFHECiphertext* openfhe_rescale(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* input,
    int64_t divisor
) {
    if (!ctx || !keypair || !input || divisor == 0) {
        set_cnn_error("Invalid parameters for rescale");
        return nullptr;
    }
    
    try {
        // Decrypt input
        Plaintext input_plain;
        ctx->cryptoContext->Decrypt(
            keypair->keyPair.secretKey,
            input->ciphertext,
            &input_plain
        );
        std::vector<int64_t> input_vec = input_plain->GetPackedValue();
        
        size_t slot_count = ctx->cryptoContext->GetEncodingParams()->GetBatchSize();
        int64_t p = ctx->cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
        int64_t half_p = p / 2;
        std::vector<int64_t> output_vec(slot_count, 0);
        
        // Divide each value by divisor (integer division)
        // Handle signed representation: BFV stores negative values as p-|x|
        for (size_t i = 0; i < input_vec.size() && i < slot_count; i++) {
            int64_t val = input_vec[i];
            if (val > half_p) val -= p;  // Convert to signed
            output_vec[i] = val / divisor;
        }
        
        // Re-encrypt
        auto output_pt = ctx->cryptoContext->MakePackedPlaintext(output_vec);
        auto output_ct = ctx->cryptoContext->Encrypt(
            keypair->keyPair.publicKey,
            output_pt
        );
        
        OpenFHECiphertext* result = new OpenFHECiphertext();
        result->ciphertext = output_ct;
        result->ctx = ctx;
        
        set_cnn_error("");
        return result;
        
    } catch (const std::exception& e) {
        set_cnn_error(std::string("Rescale failed: ") + e.what());
        return nullptr;
    }
}

