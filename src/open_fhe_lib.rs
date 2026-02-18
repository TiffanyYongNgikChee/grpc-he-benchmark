//! Safe Rust wrapper for OpenFHE homomorphic encryption library
//! 
//! This module provides a safe, idiomatic Rust interface to OpenFHE.
use crate::open_fhe_binding;
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
        let err_ptr = open_fhe_binding::openfhe_get_last_error();
        if err_ptr.is_null() {
            return String::from("Unknown error");
        }
        CStr::from_ptr(err_ptr)
            .to_string_lossy()
            .into_owned()
    }
}

// Context (owns OpenFHE crypto context)
pub struct OpenFHEContext {
    ptr: NonNull<open_fhe_binding::OpenFHEContext>,
}

impl OpenFHEContext {
    /// Create a new OpenFHE BFV context
    /// 
    /// # Parameters
    /// - plaintext_modulus: Plaintext modulus (e.g., 65537)
    /// - multiplicative_depth: Multiplicative depth (e.g., 2)
    pub fn new_bfv(plaintext_modulus: u64, multiplicative_depth: u32) -> Result<Self> {
        let ptr = unsafe {
            open_fhe_binding::openfhe_create_bfv_context(plaintext_modulus, multiplicative_depth)
        };
        
        NonNull::new(ptr)
            .map(|ptr| OpenFHEContext { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Get raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut open_fhe_binding::OpenFHEContext {
        self.ptr.as_ptr()
    }
}

impl Drop for OpenFHEContext {
    fn drop(&mut self) {
        unsafe {
            open_fhe_binding::openfhe_destroy_context(self.ptr.as_ptr());
        }
    }
}

// Context is thread-safe
unsafe impl Send for OpenFHEContext {}
unsafe impl Sync for OpenFHEContext {}

// KeyPair (owns public and secret keys)
pub struct OpenFHEKeyPair {
    ptr: NonNull<open_fhe_binding::OpenFHEKeyPair>,
}

impl OpenFHEKeyPair {
    /// Generate a new key pair from context
    pub fn generate(context: &OpenFHEContext) -> Result<Self> {
        let ptr = unsafe {
            open_fhe_binding::openfhe_generate_keypair(context.as_ptr())
        };
        
        NonNull::new(ptr)
            .map(|ptr| OpenFHEKeyPair { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Get raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut open_fhe_binding::OpenFHEKeyPair {
        self.ptr.as_ptr()
    }
}

impl Drop for OpenFHEKeyPair {
    fn drop(&mut self) {
        unsafe {
            open_fhe_binding::openfhe_destroy_keypair(self.ptr.as_ptr());
        }
    }
}

// Plaintext (unencrypted data)
pub struct OpenFHEPlaintext {
    ptr: NonNull<open_fhe_binding::OpenFHEPlaintext>,
}

impl OpenFHEPlaintext {
    /// Create plaintext from integer vector
    pub fn from_vec(context: &OpenFHEContext, values: &[i64]) -> Result<Self> {
        if values.is_empty() {
            return Err(OpenFHEError::InvalidParameter);
        }
        
        let ptr = unsafe {
            open_fhe_binding::openfhe_create_plaintext(
                context.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| OpenFHEPlaintext { ptr })
            .ok_or_else(|| OpenFHEError::Unknown(get_last_error()))
    }
    
    /// Extract values from plaintext
    pub fn to_vec(&self) -> Result<Vec<i64>> {
        const MAX_SIZE: usize = 8192; // Reasonable maximum
        let mut buffer = vec![0i64; MAX_SIZE];
        let mut length = MAX_SIZE;
        
        let success = unsafe {
            open_fhe_binding::openfhe_get_plaintext_values(
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
    pub(crate) fn as_ptr(&self) -> *mut open_fhe_binding::OpenFHEPlaintext {
        self.ptr.as_ptr()
    }
}

impl Drop for OpenFHEPlaintext {
    fn drop(&mut self) {
        unsafe {
            open_fhe_binding::openfhe_destroy_plaintext(self.ptr.as_ptr());
        }
    }
}

// Ciphertext (encrypted data)
pub struct OpenFHECiphertext {
    ptr: NonNull<open_fhe_binding::OpenFHECiphertext>,
}

impl OpenFHECiphertext {
    /// Encrypt a plaintext
    pub fn encrypt(
        context: &OpenFHEContext,
        keypair: &OpenFHEKeyPair,
        plaintext: &OpenFHEPlaintext,
    ) -> Result<Self> {
        let ptr = unsafe {
            open_fhe_binding::openfhe_encrypt(
                context.as_ptr(),
                keypair.as_ptr(),
                plaintext.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| OpenFHECiphertext { ptr })
            .ok_or(OpenFHEError::EncryptionFailed)
    }
    
    /// Decrypt to plaintext
    pub fn decrypt(
        &self,
        context: &OpenFHEContext,
        keypair: &OpenFHEKeyPair,
    ) -> Result<OpenFHEPlaintext> {
        let ptr = unsafe {
            open_fhe_binding::openfhe_decrypt(
                context.as_ptr(),
                keypair.as_ptr(),
                self.ptr.as_ptr(),
            )
        };
        
        NonNull::new(ptr)
            .map(|ptr| OpenFHEPlaintext { ptr })
            .ok_or(OpenFHEError::DecryptionFailed)
    }
    
    /// Add two ciphertexts homomorphically
    pub fn add(&self, _context: &OpenFHEContext, other: &OpenFHECiphertext) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_eval_add(
            self.ptr.as_ptr(),
            other.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or(OpenFHEError::OperationFailed)
}

/// Multiply two ciphertexts homomorphically
pub fn multiply(
    &self,
    _context: &OpenFHEContext,
    _keypair: &OpenFHEKeyPair,
    other: &OpenFHECiphertext,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_eval_mult(
            self.ptr.as_ptr(),
            other.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or(OpenFHEError::OperationFailed)
}

/// Subtract two ciphertexts homomorphically
pub fn subtract(&self, _context: &OpenFHEContext, other: &OpenFHECiphertext) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_eval_subtract(
            self.ptr.as_ptr(),
            other.ptr.as_ptr(),
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or(OpenFHEError::OperationFailed)
}

// CNN Operations
/// Matrix multiplication with plaintext weights (for fully connected layers)
/// Uses decrypt→compute→re-encrypt approach (same as conv2d/avgpool)
/// 
/// # Parameters
/// - context: OpenFHE context
/// - keypair: Key pair for decrypt/re-encrypt
/// - weights: Plaintext weight matrix (flattened row-major)
/// - rows: Number of output rows
/// - cols: Number of input columns
/// - divisor: Value to divide each output element by (e.g., scale_factor). Use 1 for no division.
/// 
/// # Returns
/// Encrypted result vector (size: rows)
pub fn matmul(
    context: &OpenFHEContext,
    keypair: &OpenFHEKeyPair,
    weights: &OpenFHEPlaintext,
    input: &OpenFHECiphertext,
    rows: usize,
    cols: usize,
    divisor: i64,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_matmul(
            context.as_ptr(),
            keypair.as_ptr(),
            weights.as_ptr(),
            input.ptr.as_ptr(),
            rows,
            cols,
            divisor,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Matrix multiplication failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}

/// 2D convolution with plaintext kernel (for CNN layers)
/// 
/// # Parameters
/// - context: OpenFHE context
/// - kernel: Plaintext convolution filter (flattened)
/// - input_height: Height of input image
/// - input_width: Width of input image
/// - kernel_height: Height of convolution kernel
/// - kernel_width: Width of convolution kernel
/// - divisor: Value to divide each output element by (e.g., scale_factor). Use 1 for no division.
/// 
/// # Returns
/// Encrypted feature map with dimensions:
/// - out_height = input_height - kernel_height + 1
/// - out_width = input_width - kernel_width + 1
pub fn conv2d(
    &self,
    context: &OpenFHEContext,
    keypair: &OpenFHEKeyPair,
    kernel: &OpenFHEPlaintext,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    divisor: i64,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_conv2d(
            context.as_ptr(),
            keypair.as_ptr(),
            self.ptr.as_ptr(),
            kernel.as_ptr(),
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            divisor,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Convolution failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}

/// Polynomial ReLU activation approximation
/// 
/// # Parameters
/// - context: OpenFHE context
/// - degree: Polynomial degree (3, 5, or 7). Currently only 3 is supported.
/// 
/// # Returns
/// Encrypted activated values (approximates ReLU)
/// 
/// # Notes
/// Degree-3 polynomial: 0.125*x³ + 0.5*x + 0.5
pub fn poly_relu(&self, context: &OpenFHEContext, degree: i32) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_poly_relu(
            context.as_ptr(),
            self.ptr.as_ptr(),
            degree,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Polynomial ReLU failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}

/// Square activation using decrypt→compute→re-encrypt approach
/// 
/// Computes f(x) = x² for each value without modular overflow.
/// Unlike `poly_relu` (which uses homomorphic multiplication), this
/// decrypts, squares in 64-bit integer space, and re-encrypts.
/// 
/// # Parameters
/// - context: OpenFHE context
/// - keypair: Key pair for decrypt/re-encrypt
/// - divisor: Value to divide by after squaring (e.g., scale_factor)
/// 
/// # Returns
/// Encrypted squared+rescaled values (x² / divisor)
pub fn square_activate(
    &self,
    context: &OpenFHEContext,
    keypair: &OpenFHEKeyPair,
    divisor: i64,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_square_activate(
            context.as_ptr(),
            keypair.as_ptr(),
            self.ptr.as_ptr(),
            divisor,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Square activation failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}

/// Average pooling for downsampling feature maps
/// 
/// # Parameters
/// - context: OpenFHE context
/// - input_height: Height of input feature map
/// - input_width: Width of input feature map
/// - pool_size: Pooling window size (e.g., 2 for 2×2 pooling)
/// - stride: Stride for pooling (typically same as pool_size)
/// 
/// # Returns
/// Encrypted downsampled feature map with dimensions:
/// - out_height = (input_height - pool_size) / stride + 1
/// - out_width = (input_width - pool_size) / stride + 1
pub fn avgpool(
    &self,
    context: &OpenFHEContext,
    keypair: &OpenFHEKeyPair,
    input_height: usize,
    input_width: usize,
    pool_size: usize,
    stride: usize,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_avgpool(
            context.as_ptr(),
            keypair.as_ptr(),
            self.ptr.as_ptr(),
            input_height,
            input_width,
            pool_size,
            stride,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Average pooling failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}

/// Rescale encrypted values by dividing by a divisor
/// 
/// Used after polynomial activation (x²) to prevent scale accumulation.
/// After x² with scale_factor S, intermediate values grow as S².
/// Dividing by S brings them back to the expected range (~S).
/// 
/// # Parameters
/// - context: OpenFHE context
/// - keypair: Key pair for decrypt/re-encrypt
/// - divisor: Value to divide by (e.g., scale_factor after x²)
/// 
/// # Returns
/// Rescaled encrypted values
pub fn rescale(
    &self,
    context: &OpenFHEContext,
    keypair: &OpenFHEKeyPair,
    divisor: i64,
) -> Result<OpenFHECiphertext> {
    let ptr = unsafe {
        open_fhe_binding::openfhe_rescale(
            context.as_ptr(),
            keypair.as_ptr(),
            self.ptr.as_ptr(),
            divisor,
        )
    };
    
    NonNull::new(ptr)
        .map(|ptr| OpenFHECiphertext { ptr })
        .ok_or_else(|| {
            let err = unsafe {
                let err_ptr = open_fhe_binding::openfhe_cnn_get_last_error();
                if !err_ptr.is_null() {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                } else {
                    String::from("Rescale failed")
                }
            };
            OpenFHEError::Unknown(err)
        })
}
}

impl Drop for OpenFHECiphertext {
    fn drop(&mut self) {
        unsafe {
            open_fhe_binding::openfhe_destroy_ciphertext(self.ptr.as_ptr());
        }
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        let ctx = OpenFHEContext::new_bfv(65537, 2);
        assert!(ctx.is_ok());
    }
    
    #[test]
    fn test_keypair_generation() {
        let ctx = OpenFHEContext::new_bfv(65537, 2).unwrap();
        let keypair = OpenFHEKeyPair::generate(&ctx);
        assert!(keypair.is_ok());
    }
    
    #[test]
    fn test_encryption_decryption() {
        let ctx = OpenFHEContext::new_bfv(65537, 2).unwrap();
        let keypair = OpenFHEKeyPair::generate(&ctx).unwrap();
        
        let values = vec![1, 2, 3, 4, 5];
        let plaintext = OpenFHEPlaintext::from_vec(&ctx, &values).unwrap();
        
        let ciphertext = OpenFHECiphertext::encrypt(&ctx, &keypair, &plaintext).unwrap();
        let decrypted = ciphertext.decrypt(&ctx, &keypair).unwrap();
        
        let result = decrypted.to_vec().unwrap();
        assert_eq!(&result[..5], &values[..]);
    }
}