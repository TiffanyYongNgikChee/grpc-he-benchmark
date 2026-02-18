//! Raw FFI bindings to OpenFHE C wrapper
//! 
//! SAFETY: All functions are unsafe and require careful handling

use std::os::raw::{c_char, c_uint, c_ulonglong};

// Opaque Types (match C header)
#[repr(C)]
pub struct OpenFHEContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OpenFHEKeyPair {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OpenFHEPlaintext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OpenFHECiphertext {
    _private: [u8; 0],
}

// FFI Function Declarations

unsafe extern "C" {
    // Context management
    pub fn openfhe_create_bfv_context(
        plaintext_modulus: c_ulonglong,
        multiplicative_depth: c_uint,
    ) -> *mut OpenFHEContext;
    
    pub fn openfhe_destroy_context(ctx: *mut OpenFHEContext);
    
    // Key management
    pub fn openfhe_generate_keypair(
        ctx: *mut OpenFHEContext,
    ) -> *mut OpenFHEKeyPair;
    
    pub fn openfhe_destroy_keypair(keypair: *mut OpenFHEKeyPair);
    
    // Plaintext operations
    pub fn openfhe_create_plaintext(
        ctx: *mut OpenFHEContext,
        values: *const i64,
        length: usize,
    ) -> *mut OpenFHEPlaintext;
    
    pub fn openfhe_destroy_plaintext(plain: *mut OpenFHEPlaintext);
    
    pub fn openfhe_get_plaintext_values(
        plain: *mut OpenFHEPlaintext,
        out_values: *mut i64,
        out_length: *mut usize,
    ) -> bool;
    
    // Encryption/Decryption
    pub fn openfhe_encrypt(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        plain: *mut OpenFHEPlaintext,
    ) -> *mut OpenFHECiphertext;
    
    pub fn openfhe_decrypt(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        cipher: *mut OpenFHECiphertext,
    ) -> *mut OpenFHEPlaintext;
    
    pub fn openfhe_destroy_ciphertext(cipher: *mut OpenFHECiphertext);
    
    // Homomorphic operations
    pub fn openfhe_eval_add(
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    pub fn openfhe_eval_mult(
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    pub fn openfhe_eval_subtract(
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    // Error handling
    pub fn openfhe_get_last_error() -> *const c_char;
    
    // CNN Operations (from openfhe_cnn_ops.h)
    /// Matrix multiplication for fully connected layers
    /// weights: flattened row-major matrix (rows × cols)
    /// input: encrypted vector (cols elements)
    /// returns: encrypted vector (rows elements)
    pub fn openfhe_matmul(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        weights: *mut OpenFHEPlaintext,
        input: *mut OpenFHECiphertext,
        rows: usize,
        cols: usize,
        divisor: i64,
    ) -> *mut OpenFHECiphertext;
    
    /// 2D convolution for CNN layers
    /// input: encrypted image (flattened height × width)
    /// kernel: plaintext filter (flattened kernel_height × kernel_width)
    /// returns: encrypted feature map (out_height × out_width)
    pub fn openfhe_conv2d(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        input: *mut OpenFHECiphertext,
        kernel: *mut OpenFHEPlaintext,
        input_height: usize,
        input_width: usize,
        kernel_height: usize,
        kernel_width: usize,
        divisor: i64,
    ) -> *mut OpenFHECiphertext;
    
    /// Polynomial ReLU approximation
    /// degree: 3, 5, or 7 (currently only 3 is implemented)
    /// returns: encrypted activated values
    pub fn openfhe_poly_relu(
        ctx: *mut OpenFHEContext,
        input: *mut OpenFHECiphertext,
        degree: i32,
    ) -> *mut OpenFHECiphertext;
    
    /// Square activation with integrated rescale using decrypt→compute→re-encrypt
    /// Computes f(x) = x² / divisor without modular overflow
    pub fn openfhe_square_activate(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        input: *mut OpenFHECiphertext,
        divisor: i64,
    ) -> *mut OpenFHECiphertext;
    
    /// Average pooling for downsampling
    /// input: encrypted feature map (flattened input_height × input_width)
    /// pool_size: pooling window size (e.g., 2 for 2×2)
    /// stride: stride for pooling
    /// returns: encrypted downsampled feature map
    pub fn openfhe_avgpool(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        input: *mut OpenFHECiphertext,
        input_height: usize,
        input_width: usize,
        pool_size: usize,
        stride: usize,
    ) -> *mut OpenFHECiphertext;
    
    /// Rescale encrypted values by dividing by a divisor
    /// Used after x² activation to prevent scale accumulation
    pub fn openfhe_rescale(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        input: *mut OpenFHECiphertext,
        divisor: i64,
    ) -> *mut OpenFHECiphertext;
    
    /// Get last error from CNN operations
    pub fn openfhe_cnn_get_last_error() -> *const c_char;
}