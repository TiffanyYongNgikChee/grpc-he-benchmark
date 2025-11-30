//! Safe Rust wrapper for OpenFHE homomorphic encryption library
//! 
//! This module provides a safe, idiomatic Rust interface to OpenFHE.

mod ffi;

use std::ffi::CStr;
use std::ptr::NonNull;

// Error Types
#[derive(Debug)]
pub enum OpenFHEError {
    NullPointer,
    InvalidParameter,
    EncryptionFailed,
    DecryptionFailed,
    OperationFailed,
    Unknown(String),
}

impl std::fmt::Display for OpenFHEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullPointer => write!(f, "Null pointer returned from OpenFHE"),
            Self::InvalidParameter => write!(f, "Invalid parameter"),
            Self::EncryptionFailed => write!(f, "Encryption failed"),
            Self::DecryptionFailed => write!(f, "Decryption failed"),
            Self::OperationFailed => write!(f, "Operation failed"),
            Self::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for OpenFHEError {}

pub type Result<T> = std::result::Result<T, OpenFHEError>;

/// Get last error from OpenFHE
fn get_last_error() -> String {
    unsafe {
        let err_ptr = ffi::openfhe_get_last_error();
        if err_ptr.is_null() {
            return String::from("Unknown error");
        }
        CStr::from_ptr(err_ptr)
            .to_string_lossy()
            .into_owned()
    }
}

// Context (owns OpenFHE crypto context)
pub struct Context {
    ptr: NonNull<ffi::OpenFHEContext>,
}

impl Context {
    /// Create a new OpenFHE BFV context
    /// 
    /// # Parameters
    /// - plaintext_modulus: Plaintext modulus (e.g., 65537)
    /// - multiplicative_depth: Multiplicative depth (e.g., 2)
    pub fn new_bfv(plaintext_modulus: u64, multiplicative_depth: u32) -> Result<Self> {
        let ptr = unsafe {
            ffi::openfhe_create_bfv_context(plaintext_modulus, multiplicative_depth)
        };
        
        NonNull::new(ptr)
            .map(|ptr| Context { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Get raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut ffi::OpenFHEContext {
        self.ptr.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            ffi::openfhe_destroy_context(self.ptr.as_ptr());
        }
    }
}

// Context is thread-safe
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

// KeyPair (owns public and secret keys)
pub struct KeyPair {
    ptr: NonNull<ffi::OpenFHEKeyPair>,
}

impl KeyPair {
    /// Generate a new key pair from context
    pub fn generate(context: &Context) -> Result<Self> {
        let ptr = unsafe {
            ffi::openfhe_generate_keypair(context.as_ptr())
        };
        
        NonNull::new(ptr)
            .map(|ptr| KeyPair { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Get raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut ffi::OpenFHEKeyPair {
        self.ptr.as_ptr()
    }
}

impl Drop for KeyPair {
    fn drop(&mut self) {
        unsafe {
            ffi::openfhe_destroy_keypair(self.ptr.as_ptr());
        }
    }
}

// Plaintext (unencrypted data)
pub struct Plaintext {
    ptr: NonNull<ffi::OpenFHEPlaintext>,
}

impl Plaintext {
    /// Create plaintext from integer vector
    pub fn from_vec(context: &Context, values: &[i64]) -> Result<Self> {
        if values.is_empty() {
            return Err(OpenFHEError::InvalidParameter);
        }
        
        let ptr = unsafe {
            ffi::openfhe_create_plaintext(
                context.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Plaintext { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Extract values from plaintext
    pub fn to_vec(&self) -> Result<Vec<i64>> {
        const MAX_SIZE: usize = 8192; // Reasonable maximum
        let mut buffer = vec![0i64; MAX_SIZE];
        let mut length = MAX_SIZE;
        
        let success = unsafe {
            ffi::openfhe_get_plaintext_values(
                self.ptr.as_ptr(),
                buffer.as_mut_ptr(),
                &mut length as *mut usize,
            )
        };
        
        if !success {
            return Err(OpenFHEError::Unknown(get_last_error()));
        }
        
        buffer.truncate(length);
        Ok(buffer)
    }
    
    /// Get raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut ffi::OpenFHEPlaintext {
        self.ptr.as_ptr()
    }
}

impl Drop for Plaintext {
    fn drop(&mut self) {
        unsafe {
            ffi::openfhe_destroy_plaintext(self.ptr.as_ptr());
        }
    }
}

// Ciphertext (encrypted data)
pub struct Ciphertext {
    ptr: NonNull<ffi::OpenFHECiphertext>,
}

impl Ciphertext {
    /// Encrypt a plaintext
    pub fn encrypt(
        context: &Context,
        keypair: &KeyPair,
        plaintext: &Plaintext,
    ) -> Result<Self> {
        let ptr = unsafe {
            ffi::openfhe_encrypt(
                context.as_ptr(),
                keypair.as_ptr(),
                plaintext.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(OpenFHEError::EncryptionFailed)
    }
    
    /// Decrypt to plaintext
    pub fn decrypt(
        &self,
        context: &Context,
        keypair: &KeyPair,
    ) -> Result<Plaintext> {
        let ptr = unsafe {
            ffi::openfhe_decrypt(
                context.as_ptr(),
                keypair.as_ptr(),
                self.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Plaintext { ptr })
            .ok_or(OpenFHEError::DecryptionFailed)
    }
    
    /// Add two ciphertexts homomorphically
    pub fn add(&self, context: &Context, other: &Ciphertext) -> Result<Ciphertext> {
        let ptr = unsafe {
            ffi::openfhe_add(
                context.as_ptr(),
                self.ptr.as_ptr(),
                other.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(OpenFHEError::OperationFailed)
    }
    
    /// Multiply two ciphertexts homomorphically
    pub fn multiply(
        &self,
        context: &Context,
        keypair: &KeyPair,
        other: &Ciphertext,
    ) -> Result<Ciphertext> {
        let ptr = unsafe {
            ffi::openfhe_multiply(
                context.as_ptr(),
                keypair.as_ptr(),
                self.ptr.as_ptr(),
                other.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(OpenFHEError::OperationFailed)
    }
    
    /// Subtract two ciphertexts homomorphically
    pub fn subtract(&self, context: &Context, other: &Ciphertext) -> Result<Ciphertext> {
        let ptr = unsafe {
            ffi::openfhe_subtract(
                context.as_ptr(),
                self.ptr.as_ptr(),
                other.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(OpenFHEError::OperationFailed)
    }
}

impl Drop for Ciphertext {
    fn drop(&mut self) {
        unsafe {
            ffi::openfhe_destroy_ciphertext(self.ptr.as_ptr());
        }
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        let ctx = Context::new_bfv(65537, 2);
        assert!(ctx.is_ok());
    }
    
    #[test]
    fn test_keypair_generation() {
        let ctx = Context::new_bfv(65537, 2).unwrap();
        let keypair = KeyPair::generate(&ctx);
        assert!(keypair.is_ok());
    }
    
    #[test]
    fn test_encryption_decryption() {
        let ctx = Context::new_bfv(65537, 2).unwrap();
        let keypair = KeyPair::generate(&ctx).unwrap();
        
        let values = vec![1, 2, 3, 4, 5];
        let plaintext = Plaintext::from_vec(&ctx, &values).unwrap();
        
        let ciphertext = Ciphertext::encrypt(&ctx, &keypair, &plaintext).unwrap();
        let decrypted = ciphertext.decrypt(&ctx, &keypair).unwrap();
        
        let result = decrypted.to_vec().unwrap();
        assert_eq!(&result[..5], &values[..]);
    }
}