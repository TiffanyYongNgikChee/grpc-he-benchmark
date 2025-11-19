//! Medical Data Encryption Comparison: SEAL vs HElib
//! 
//! This example encrypts the same medical record using both frameworks
//! and provides a detailed performance comparison.

use he_benchmark::{
    Context as SealContext, 
    Encryptor as SealEncryptor, 
    Decryptor as SealDecryptor,
    BatchEncoder as SealBatchEncoder,
    HEContext,
    HESecretKey,
    HEPublicKey,
    HEPlaintext,
};

use std::time::{Instant, Duration};
use std::thread::sleep;
use std::io::{self, Write};

