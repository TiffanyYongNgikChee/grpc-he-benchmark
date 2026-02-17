//! Safe Rust wrapper for SEAL homomorphic encryption library
//! 
//! This module provides a safe, idiomatic Rust interface to Microsoft SEAL.

mod bindings; // imports the low-level FFI bindings (the C function definitions) that connect to C++ wrapper
mod helib_bindings;     // HElib FFI bindings
pub mod helib;          // HElib safe wrapper 
mod open_fhe_binding;
pub mod open_fhe_lib;
pub mod weight_loader;  // MNIST weight loader (CSV parser for exported weights)

use std::ffi::{CStr, CString}; // CStr and CString convert between Rust strings and C strings.
use std::ptr::NonNull; // NonNull safely wraps raw pointers that should never be null.

// Error Types
#[derive(Debug)]
pub enum SealError {
    // Defines all possible errors might encounter
    NullPointer,
    InvalidParameter,
    EncryptionFailed,
    DecryptionFailed,
    OperationFailed,
    // Rust’s Result<T, SealError> then makes it safe to handle errors using ?.
}


// Implement Display for SealError
// Makes SealError printable and compatible with Rust’s standard Result and ? error-handling syntax
impl std::fmt::Display for SealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SealError::NullPointer => write!(f, "Null pointer returned from SEAL"),
            SealError::InvalidParameter => write!(f, "Invalid parameter provided"),
            SealError::EncryptionFailed => write!(f, "Encryption operation failed"),
            SealError::DecryptionFailed => write!(f, "Decryption operation failed"),
            SealError::OperationFailed => write!(f, "SEAL operation failed"),
        }
    }
}

// Implement Error trait for SealError
impl std::error::Error for SealError {}

pub type Result<T> = std::result::Result<T, SealError>;

// Context (owns SEAL context and keys)
pub struct Context {
    // store only a pointer to the C++ object, but wrapped in NonNull to ensure it’s valid
    ptr: NonNull<bindings::SEALContext>,
}

impl Context {
    /// Create a new SEAL context with BFV scheme
    /// 
    /// # Parameters
    /// - poly_modulus_degree: Polynomial modulus degree (e.g., 4096, 8192)
    /// - plain_modulus: Plaintext modulus for BFV
    pub fn new(poly_modulus_degree: u64, plain_modulus: u64) -> Result<Self> {
        // Standard coefficient modulus for given poly degree
        let coeff_modulus = vec![36, 36, 37]; // bits per prime (109 bits total)
        
        // Calls C++ seal_create_context function via FFI (marked unsafe because it’s a raw pointer)
        let ptr = unsafe {
            bindings::seal_create_context(
                poly_modulus_degree,
                coeff_modulus.as_ptr(),
                coeff_modulus.len(),
                plain_modulus,
            )
        };
        // If the pointer returned from C++ is valid, store it inside a Context.
        // If it’s null, return a NullPointer error.
        NonNull::new(ptr)
            .map(|ptr| Context { ptr })
            .ok_or(SealError::NullPointer)
    }
}

// When the Rust Context goes out of scope, 
// it automatically calls the C++ function to free memory — so the user can’t forget
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_context(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Encryptor
// ============================================
// Represents the C++ Encryptor object (handles encryption).
pub struct Encryptor {
    ptr: NonNull<bindings::SEALEncryptor>,
}

// Creates an encryptor using the existing SEAL context.
impl Encryptor {
    pub fn new(context: &Context) -> Result<Self> {
        let ptr = unsafe {
            bindings::seal_create_encryptor(
                context.ptr.as_ptr(),
                std::ptr::null(),
                0,
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Encryptor { ptr })
            .ok_or(SealError::NullPointer)
    }
    
    pub fn encrypt(&self, plaintext: &Plaintext) -> Result<Ciphertext> {
        let ptr = unsafe {
            bindings::seal_encrypt(
                self.ptr.as_ptr(),
                plaintext.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(SealError::EncryptionFailed)
    }
}

impl Drop for Encryptor {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_encryptor(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Decryptor
// ============================================
pub struct Decryptor {
    ptr: NonNull<bindings::SEALDecryptor>,
}

impl Decryptor {
    pub fn new(context: &Context) -> Result<Self> {
        let ptr = unsafe {
            bindings::seal_create_decryptor(
                context.ptr.as_ptr(),
                std::ptr::null(),
                0,
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Decryptor { ptr })
            .ok_or(SealError::NullPointer)
    }
    
    pub fn decrypt(&self, ciphertext: &Ciphertext) -> Result<Plaintext> {
        let ptr = unsafe {
            bindings::seal_decrypt(
                self.ptr.as_ptr(),
                ciphertext.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Plaintext { ptr })
            .ok_or(SealError::DecryptionFailed)
    }
}

impl Drop for Decryptor {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_decryptor(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Batch Encoder
// ============================================
pub struct BatchEncoder {
    ptr: NonNull<bindings::SEALBatchEncoder>,
}

impl BatchEncoder {
    pub fn new(context: &Context) -> Result<Self> {
        let ptr = unsafe {
            bindings::seal_create_batch_encoder(context.ptr.as_ptr())
        };
        
        NonNull::new(ptr)
            .map(|ptr| BatchEncoder { ptr })
            .ok_or(SealError::NullPointer)
    }
    
    /// Encode a vector of integers into a plaintext
    pub fn encode(&self, values: &[i64]) -> Result<Plaintext> {
        let ptr = unsafe {
            bindings::seal_batch_encode(
                self.ptr.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Plaintext { ptr })
            .ok_or(SealError::NullPointer)
    }
    
    /// Decode a plaintext back to vector of integers
    pub fn decode(&self, plain: &Plaintext) -> Result<Vec<i64>> {
        let mut output = vec![0i64; self.slot_count()];
        let mut output_size = output.len();
        
        unsafe {
            bindings::seal_batch_decode(
                self.ptr.as_ptr(),
                plain.ptr.as_ptr(),
                output.as_mut_ptr(),
                &mut output_size,
            );
        }
        
        output.truncate(output_size);
        Ok(output)
    }
    
    pub fn slot_count(&self) -> usize {
        unsafe { bindings::seal_get_slot_count(self.ptr.as_ptr()) }
    }
}

impl Drop for BatchEncoder {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_batch_encoder(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Galois Keys
// ============================================
pub struct GaloisKeys {
    ptr: NonNull<bindings::SEALGaloisKeys>,
}

impl GaloisKeys {
    pub fn generate(context: &Context) -> Result<Self> {
        let ptr = unsafe {
            bindings::seal_generate_galois_keys(context.ptr.as_ptr())
        };
        
        NonNull::new(ptr)
            .map(|ptr| GaloisKeys { ptr })
            .ok_or(SealError::NullPointer)
    }
}

impl Drop for GaloisKeys {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_galois_keys(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Rotation
// ============================================
pub fn rotate_rows(
    context: &Context,
    cipher: &Ciphertext,
    steps: i32,
    galois_keys: &GaloisKeys,
) -> Result<Ciphertext> {
    let ptr = unsafe {
        bindings::seal_rotate_rows(
            context.ptr.as_ptr(),
            cipher.ptr.as_ptr(),
            steps,
            galois_keys.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| Ciphertext { ptr })
        .ok_or(SealError::OperationFailed)
}

// ============================================
// Plaintext
// ============================================
pub struct Plaintext {
    ptr: NonNull<bindings::SEALPlaintext>,
}

impl Plaintext {
    pub fn from_hex(hex: &str) -> Result<Self> {
        let c_hex = CString::new(hex).map_err(|_| SealError::InvalidParameter)?;
        
        let ptr = unsafe {
            bindings::seal_create_plaintext(c_hex.as_ptr())
        };
        
        NonNull::new(ptr)
            .map(|ptr| Plaintext { ptr })
            .ok_or(SealError::NullPointer)
    }
    
    pub fn to_string(&self) -> Result<String> {
        let ptr = unsafe {
            bindings::seal_plaintext_to_string(self.ptr.as_ptr())
        };
        
        if ptr.is_null() {
            return Err(SealError::NullPointer);
        }
        
        let c_str = unsafe { CStr::from_ptr(ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }
}

impl Drop for Plaintext {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_plaintext(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Ciphertext
// ============================================
pub struct Ciphertext {
    ptr: NonNull<bindings::SEALCiphertext>,
}

impl Ciphertext {
    /// Get the number of polynomials in the ciphertext (usually 2 for fresh encryptions)
    pub fn size(&self) -> usize {
        unsafe {
            bindings::seal_ciphertext_size(self.ptr.as_ptr())
        }
    }
    
    /// Get the polynomial modulus degree (number of coefficients per polynomial)
    pub fn coeff_count(&self) -> u64 {
        unsafe {
            bindings::seal_ciphertext_coeff_count(self.ptr.as_ptr())
        }
    }
    
    /// Get the total size in bytes when serialized
    pub fn byte_count(&self) -> usize {
        unsafe {
            bindings::seal_ciphertext_byte_count(self.ptr.as_ptr())
        }
    }
    
    /// Get a human-readable summary of the ciphertext
    pub fn info(&self) -> Result<String> {
        let c_str = unsafe {
            let ptr = bindings::seal_ciphertext_info(self.ptr.as_ptr());
            if ptr.is_null() {
                return Err(SealError::NullPointer);
            }
            CStr::from_ptr(ptr)
        };
        
        c_str.to_str()
            .map(|s| s.to_owned())
            .map_err(|_| SealError::OperationFailed)
    }
}

impl Drop for Ciphertext {
    fn drop(&mut self) {
        unsafe {
            bindings::seal_destroy_ciphertext(self.ptr.as_ptr());
        }
    }
}

// ============================================
// Homomorphic Operations
// ============================================
pub fn add(context: &Context, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
    let ptr = unsafe {
        bindings::seal_add(
            context.ptr.as_ptr(),
            a.ptr.as_ptr(),
            b.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| Ciphertext { ptr })
        .ok_or(SealError::OperationFailed)
}

pub fn multiply(context: &Context, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
    let ptr = unsafe {
        bindings::seal_multiply(
            context.ptr.as_ptr(),
            a.ptr.as_ptr(),
            b.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| Ciphertext { ptr })
        .ok_or(SealError::OperationFailed)
}
// Re-export HElib types with prefix
pub use helib::{
    HEContext, HESecretKey, HEPublicKey, 
    HEPlaintext, HECiphertext
};

pub use open_fhe_lib::{
    OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext
};