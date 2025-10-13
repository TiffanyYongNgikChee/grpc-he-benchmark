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

pub type Result<T> = std::result::Result<T, SealError>;