//! Raw FFI bindings to SEAL C wrapper

use std::os::raw::{c_char, c_ulonglong};

// ============================================
// Opaque Types (match C header)
// ============================================
#[repr(C)]
pub struct SEALContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALEncryptor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALDecryptor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALCiphertext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALPlaintext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALBatchEncoder {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SEALGaloisKeys {
    _private: [u8; 0],
}

// ============================================
// FFI Function Declarations
// ============================================
unsafe extern "C" {
    // Context management
    pub fn seal_create_context(
        poly_modulus_degree: c_ulonglong,
        coeff_modulus: *const c_ulonglong,
        coeff_modulus_size: usize,
        plain_modulus: c_ulonglong,
    ) -> *mut SEALContext;
    
    pub fn seal_destroy_context(ctx: *mut SEALContext);
    
    // Encryptor
    pub fn seal_create_encryptor(
        ctx: *mut SEALContext,
        public_key: *const u8,
        public_key_size: usize,
    ) -> *mut SEALEncryptor;
    
    pub fn seal_destroy_encryptor(enc: *mut SEALEncryptor);
    
    // Decryptor
    pub fn seal_create_decryptor(
        ctx: *mut SEALContext,
        secret_key: *const u8,
        secret_key_size: usize,
    ) -> *mut SEALDecryptor;
    
    pub fn seal_destroy_decryptor(dec: *mut SEALDecryptor);
    
    // Plaintext
    pub fn seal_create_plaintext(hex_string: *const c_char) -> *mut SEALPlaintext;
    pub fn seal_destroy_plaintext(plain: *mut SEALPlaintext);
    pub fn seal_plaintext_to_string(plain: *mut SEALPlaintext) -> *const c_char;
    
    // Encryption/Decryption
    pub fn seal_encrypt(
        encryptor: *mut SEALEncryptor,
        plaintext: *mut SEALPlaintext,
    ) -> *mut SEALCiphertext;
    
    pub fn seal_decrypt(
        decryptor: *mut SEALDecryptor,
        ciphertext: *mut SEALCiphertext,
    ) -> *mut SEALPlaintext;
    
    pub fn seal_destroy_ciphertext(cipher: *mut SEALCiphertext);

    // Ciphertext inspection (NEW!)
    pub fn seal_ciphertext_size(cipher: *mut SEALCiphertext) -> usize;
    pub fn seal_ciphertext_coeff_count(cipher: *mut SEALCiphertext) -> u64;
    pub fn seal_ciphertext_byte_count(cipher: *mut SEALCiphertext) -> usize;
    pub fn seal_ciphertext_info(cipher: *mut SEALCiphertext) -> *const c_char;
    
    // Homomorphic operations
    pub fn seal_add(
        ctx: *mut SEALContext,
        a: *mut SEALCiphertext,
        b: *mut SEALCiphertext,
    ) -> *mut SEALCiphertext;
    
    pub fn seal_multiply(
        ctx: *mut SEALContext,
        a: *mut SEALCiphertext,
        b: *mut SEALCiphertext,
    ) -> *mut SEALCiphertext;

    // Batch encoder
    pub fn seal_create_batch_encoder(ctx: *mut SEALContext) -> *mut SEALBatchEncoder;
    pub fn seal_destroy_batch_encoder(encoder: *mut SEALBatchEncoder);
    pub fn seal_batch_encode(
        encoder: *mut SEALBatchEncoder,
        values: *const i64,
        values_size: usize,
    ) -> *mut SEALPlaintext;
    pub fn seal_batch_decode(
        encoder: *mut SEALBatchEncoder,
        plain: *mut SEALPlaintext,
        output: *mut i64,
        output_size: *mut usize,
    );
    pub fn seal_get_slot_count(encoder: *mut SEALBatchEncoder) -> usize;
    
    // Galois keys
    pub fn seal_generate_galois_keys(ctx: *mut SEALContext) -> *mut SEALGaloisKeys;
    pub fn seal_destroy_galois_keys(keys: *mut SEALGaloisKeys);
    pub fn seal_rotate_rows(
        ctx: *mut SEALContext,
        cipher: *mut SEALCiphertext,
        steps: i32,
        galois_keys: *mut SEALGaloisKeys,
    ) -> *mut SEALCiphertext;
}