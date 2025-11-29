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
// ============================================

extern "C" {
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
    pub fn openfhe_add(
        ctx: *mut OpenFHEContext,
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    pub fn openfhe_multiply(
        ctx: *mut OpenFHEContext,
        keypair: *mut OpenFHEKeyPair,
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    pub fn openfhe_subtract(
        ctx: *mut OpenFHEContext,
        a: *mut OpenFHECiphertext,
        b: *mut OpenFHECiphertext,
    ) -> *mut OpenFHECiphertext;
    
    // Error handling
    pub fn openfhe_get_last_error() -> *const c_char;
}