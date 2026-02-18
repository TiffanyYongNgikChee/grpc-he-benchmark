//! Homomorphic Encryption Benchmark Library
//!
//! Safe Rust wrappers for HE libraries (SEAL, HElib, OpenFHE).
//! Use feature flags to control which backends are linked:
//!
//!   cargo build                       # default: OpenFHE only
//!   cargo build --features seal       # + SEAL
//!   cargo build --features helib      # + HElib
//!   cargo build --features all_he     # all three (Docker build)

// ============================================================================
// Module declarations (feature-gated)
// ============================================================================

#[cfg(feature = "seal")]
mod bindings;

#[cfg(feature = "helib")]
mod helib_bindings;

#[cfg(feature = "helib")]
pub mod helib;

mod open_fhe_binding;
pub mod open_fhe_lib;
pub mod weight_loader;

// ============================================================================
// SEAL wrapper (only compiled with --features seal)
// ============================================================================

#[cfg(feature = "seal")]
mod seal_wrapper {
    use super::bindings;
    use std::ffi::{CStr, CString};
    use std::ptr::NonNull;

    // Error Types
    #[derive(Debug)]
    pub enum SealError {
        NullPointer,
        InvalidParameter,
        EncryptionFailed,
        DecryptionFailed,
        OperationFailed,
    }

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

    impl std::error::Error for SealError {}

    pub type Result<T> = std::result::Result<T, SealError>;

    // Context
    pub struct Context {
        ptr: NonNull<bindings::SEALContext>,
    }

    impl Context {
        pub fn new(poly_modulus_degree: u64, plain_modulus: u64) -> Result<Self> {
            let coeff_modulus = vec![36, 36, 37];
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
            unsafe { bindings::seal_destroy_context(self.ptr.as_ptr()); }
        }
    }

    // Encryptor
    pub struct Encryptor {
        ptr: NonNull<bindings::SEALEncryptor>,
    }

    impl Encryptor {
        pub fn new(context: &Context) -> Result<Self> {
            let ptr = unsafe {
                bindings::seal_create_encryptor(context.ptr.as_ptr(), std::ptr::null(), 0)
            };
            NonNull::new(ptr)
                .map(|ptr| Encryptor { ptr })
                .ok_or(SealError::NullPointer)
        }

        pub fn encrypt(&self, plaintext: &Plaintext) -> Result<Ciphertext> {
            let ptr = unsafe { bindings::seal_encrypt(self.ptr.as_ptr(), plaintext.ptr.as_ptr()) };
            NonNull::new(ptr)
                .map(|ptr| Ciphertext { ptr })
                .ok_or(SealError::EncryptionFailed)
        }
    }

    impl Drop for Encryptor {
        fn drop(&mut self) {
            unsafe { bindings::seal_destroy_encryptor(self.ptr.as_ptr()); }
        }
    }

    // Decryptor
    pub struct Decryptor {
        ptr: NonNull<bindings::SEALDecryptor>,
    }

    impl Decryptor {
        pub fn new(context: &Context) -> Result<Self> {
            let ptr = unsafe {
                bindings::seal_create_decryptor(context.ptr.as_ptr(), std::ptr::null(), 0)
            };
            NonNull::new(ptr)
                .map(|ptr| Decryptor { ptr })
                .ok_or(SealError::NullPointer)
        }

        pub fn decrypt(&self, ciphertext: &Ciphertext) -> Result<Plaintext> {
            let ptr = unsafe { bindings::seal_decrypt(self.ptr.as_ptr(), ciphertext.ptr.as_ptr()) };
            NonNull::new(ptr)
                .map(|ptr| Plaintext { ptr })
                .ok_or(SealError::DecryptionFailed)
        }
    }

    impl Drop for Decryptor {
        fn drop(&mut self) {
            unsafe { bindings::seal_destroy_decryptor(self.ptr.as_ptr()); }
        }
    }

    // BatchEncoder
    pub struct BatchEncoder {
        ptr: NonNull<bindings::SEALBatchEncoder>,
    }

    impl BatchEncoder {
        pub fn new(context: &Context) -> Result<Self> {
            let ptr = unsafe { bindings::seal_create_batch_encoder(context.ptr.as_ptr()) };
            NonNull::new(ptr)
                .map(|ptr| BatchEncoder { ptr })
                .ok_or(SealError::NullPointer)
        }

        pub fn encode(&self, values: &[i64]) -> Result<Plaintext> {
            let ptr = unsafe {
                bindings::seal_batch_encode(self.ptr.as_ptr(), values.as_ptr(), values.len())
            };
            NonNull::new(ptr)
                .map(|ptr| Plaintext { ptr })
                .ok_or(SealError::NullPointer)
        }

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
            unsafe { bindings::seal_destroy_batch_encoder(self.ptr.as_ptr()); }
        }
    }

    // GaloisKeys
    pub struct GaloisKeys {
        ptr: NonNull<bindings::SEALGaloisKeys>,
    }

    impl GaloisKeys {
        pub fn generate(context: &Context) -> Result<Self> {
            let ptr = unsafe { bindings::seal_generate_galois_keys(context.ptr.as_ptr()) };
            NonNull::new(ptr)
                .map(|ptr| GaloisKeys { ptr })
                .ok_or(SealError::NullPointer)
        }
    }

    impl Drop for GaloisKeys {
        fn drop(&mut self) {
            unsafe { bindings::seal_destroy_galois_keys(self.ptr.as_ptr()); }
        }
    }

    // Rotation
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

    // Plaintext
    pub struct Plaintext {
        ptr: NonNull<bindings::SEALPlaintext>,
    }

    impl Plaintext {
        pub fn from_hex(hex: &str) -> Result<Self> {
            let c_hex = CString::new(hex).map_err(|_| SealError::InvalidParameter)?;
            let ptr = unsafe { bindings::seal_create_plaintext(c_hex.as_ptr()) };
            NonNull::new(ptr)
                .map(|ptr| Plaintext { ptr })
                .ok_or(SealError::NullPointer)
        }

        pub fn to_string(&self) -> Result<String> {
            let ptr = unsafe { bindings::seal_plaintext_to_string(self.ptr.as_ptr()) };
            if ptr.is_null() {
                return Err(SealError::NullPointer);
            }
            let c_str = unsafe { CStr::from_ptr(ptr) };
            Ok(c_str.to_string_lossy().into_owned())
        }
    }

    impl Drop for Plaintext {
        fn drop(&mut self) {
            unsafe { bindings::seal_destroy_plaintext(self.ptr.as_ptr()); }
        }
    }

    // Ciphertext
    pub struct Ciphertext {
        ptr: NonNull<bindings::SEALCiphertext>,
    }

    impl Ciphertext {
        pub fn size(&self) -> usize {
            unsafe { bindings::seal_ciphertext_size(self.ptr.as_ptr()) }
        }

        pub fn coeff_count(&self) -> u64 {
            unsafe { bindings::seal_ciphertext_coeff_count(self.ptr.as_ptr()) }
        }

        pub fn byte_count(&self) -> usize {
            unsafe { bindings::seal_ciphertext_byte_count(self.ptr.as_ptr()) }
        }

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
            unsafe { bindings::seal_destroy_ciphertext(self.ptr.as_ptr()); }
        }
    }

    // Homomorphic Operations
    pub fn add(context: &Context, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        let ptr = unsafe {
            bindings::seal_add(context.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr())
        };
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(SealError::OperationFailed)
    }

    pub fn multiply(context: &Context, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        let ptr = unsafe {
            bindings::seal_multiply(context.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr())
        };
        NonNull::new(ptr)
            .map(|ptr| Ciphertext { ptr })
            .ok_or(SealError::OperationFailed)
    }
}

// ============================================================================
// Re-exports
// ============================================================================

#[cfg(feature = "seal")]
pub use seal_wrapper::*;

#[cfg(feature = "helib")]
pub use helib::{
    HEContext, HESecretKey, HEPublicKey,
    HEPlaintext, HECiphertext
};

pub use open_fhe_lib::{
    OpenFHEContext, OpenFHEKeyPair, OpenFHEPlaintext, OpenFHECiphertext
};
