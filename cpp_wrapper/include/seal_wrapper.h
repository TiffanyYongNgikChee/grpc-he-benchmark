#ifndef SEAL_WRAPPER_H 
// Only read this file one time.
#define SEAL_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
// This makes sure Rust can call these functions directly, 
// because Rust understands C-style names, not C++ names
extern "C" {
#endif

// ============================================
// Opaque Pointers (hide C++ from Rust)
// ============================================

// These act like ‘handles’ or ‘boxes’ that Rust can hold.
// They hide the internal structure of SEAL objects for safety.
typedef struct SEALContextWrapper SEALContextWrapper;
typedef struct SEALEncryptor SEALEncryptor;
typedef struct SEALDecryptor SEALDecryptor;
typedef struct SEALCiphertext SEALCiphertext;
typedef struct SEALPlaintext SEALPlaintext;

// ============================================
// Context Management
// ============================================

// This part sets up the encryption environment — 
// kind of like creating a workspace where all encryption happens
SEALContextWrapper* seal_create_context(
    uint64_t poly_modulus_degree,
    const uint64_t* coeff_modulus, 
    size_t coeff_modulus_size,
    uint64_t plain_modulus
);
void seal_destroy_context(SEALContextWrapper* ctx);

// ============================================
// Encryption/Decryption Setup
// ============================================

// These create the encryptor and decryptor objects that actually do encryption and decryption.
// They take in public and secret keys as input.
SEALEncryptor* seal_create_encryptor(
    SEALContextWrapper* ctx,
    const uint8_t* public_key,
    size_t public_key_size
);
void seal_destroy_encryptor(SEALEncryptor* enc);

SEALDecryptor* seal_create_decryptor(
    SEALContextWrapper* ctx,
    const uint8_t* secret_key,
    size_t secret_key_size
);
void seal_destroy_decryptor(SEALDecryptor* dec);

// ============================================
// Plaintext Operations
// ============================================

// These handle plain text data (before encryption).
// They allow you to create plaintext objects and convert them to readable strings (for debugging).
SEALPlaintext* seal_create_plaintext(const char* hex_string);
void seal_destroy_plaintext(SEALPlaintext* plain);
const char* seal_plaintext_to_string(SEALPlaintext* plain);

// ============================================
// Encryption Operations
// ============================================

// These perform the actual encryption.
// The input is plaintext, and the output is ciphertext (encrypted data)
SEALCiphertext* seal_encrypt(
    SEALEncryptor* encryptor,
    SEALPlaintext* plaintext
);
void seal_destroy_ciphertext(SEALCiphertext* cipher);

// ============================================
// Decryption Operations
// ============================================
SEALPlaintext* seal_decrypt(
    SEALDecryptor* decryptor,
    SEALCiphertext* ciphertext
);

// ============================================
// Homomorphic Operations
// ============================================

// These are special operations that work directly on encrypted data.
// They show that you can do math (like addition or multiplication) without decrypting first —
// which is the key idea of homomorphic encryption.
SEALCiphertext* seal_add(
    SEALContextWrapper* ctx,
    SEALCiphertext* a,
    SEALCiphertext* b
);

SEALCiphertext* seal_multiply(
    SEALContextWrapper* ctx,
    SEALCiphertext* a,
    SEALCiphertext* b
);

#ifdef __cplusplus
}
#endif

#endif // SEAL_WRAPPER_H