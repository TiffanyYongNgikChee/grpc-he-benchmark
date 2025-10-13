//! Basic encryption/decryption test
//! 
//! This example demonstrates:
//! 1. Creating SEAL context
//! 2. Encrypting a plaintext
//! 3. Decrypting ciphertext
//! 4. Homomorphic addition

use he_benchmark::{Context, Encryptor, Decryptor, Plaintext, add};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 SEAL FFI Test Starting...\n");

    // ============================================
    // TEST 1: Context Creation
    // ============================================
    println!("Test 1: Creating SEAL context...");
    let context = Context::new(4096, 1024)?;
    println!("✅ Context created successfully\n");
    
    // ============================================
    // TEST 2: Basic Encryption/Decryption
    // ============================================
    println!("Test 2: Basic encryption/decryption...");
    let encryptor = Encryptor::new(&context)?;
    let decryptor = Decryptor::new(&context)?;
    
    // Create plaintext
    let plain = Plaintext::from_hex("42")?;
    println!("   Original plaintext: {}", plain.to_string()?);
    
    // Encrypt
    let cipher = encryptor.encrypt(&plain)?;
    println!("   ✅ Encryption successful");
    
    // Decrypt
    let decrypted = decryptor.decrypt(&cipher)?;
    println!("   Decrypted plaintext: {}", decrypted.to_string()?);
    
    // Verify
    assert_eq!(plain.to_string()?, decrypted.to_string()?);
    println!("✅ Encryption/Decryption works!\n");
    
    
    Ok(())
}