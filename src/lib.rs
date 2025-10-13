//! Safe Rust wrapper for SEAL homomorphic encryption library
//! 
//! This module provides a safe, idiomatic Rust interface to Microsoft SEAL.

mod bindings;

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

// ============================================
// Error Types
// ============================================
#[derive(Debug)]
pub enum SealError {
    NullPointer,
    InvalidParameter,
    EncryptionFailed,
    DecryptionFailed,
    OperationFailed,
}


// Implement Display for SealError
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

// ============================================
// Context (owns SEAL context and keys)
// ============================================
pub struct Context {
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
        let coeff_modulus = vec![60, 40, 40, 60]; // bits per prime
        
        let ptr = unsafe {
            bindings::seal_create_context(
                poly_modulus_degree,
                coeff_modulus.as_ptr(),
                coeff_modulus.len(),
                plain_modulus,
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Context { ptr })
            .ok_or(SealError::NullPointer)
    }
}

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
pub struct Encryptor {
    ptr: NonNull<bindings::SEALEncryptor>,
}

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