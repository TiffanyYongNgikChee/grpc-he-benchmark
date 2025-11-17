#ifndef HELIB_WRAPPER_H
#define HELIB_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque Pointers
typedef struct HElibContext HElibContext;
typedef struct HElibSecretKey HElibSecretKey;
typedef struct HElibPublicKey HElibPublicKey;
typedef struct HElibCiphertext HElibCiphertext;
typedef struct HElibPlaintext HElibPlaintext;

// Context Management
/// Create HElib context with BGV scheme
/// @param m: Cyclotomic polynomial parameter (use 4095 for starter)
/// @param p: Plaintext modulus (use 2 for binary, or small prime)
/// @param r: Lifting parameter (use 1 for starter)
/// @return Context pointer or NULL on failure
HElibContext* helib_create_context(
    unsigned long m,
    unsigned long p,
    unsigned long r
);

void helib_destroy_context(HElibContext* ctx);

// Key Management
/// Generate secret key
HElibSecretKey* helib_generate_secret_key(HElibContext* ctx);
void helib_destroy_secret_key(HElibSecretKey* sk);

/// Derive public key from secret key
HElibPublicKey* helib_get_public_key(HElibSecretKey* sk);
void helib_destroy_public_key(HElibPublicKey* pk);

// Plaintext Operations
/// Create plaintext from integer value
HElibPlaintext* helib_create_plaintext(HElibContext* ctx, long value);

/// Get integer value from plaintext
long helib_plaintext_to_long(HElibPlaintext* plain);

void helib_destroy_plaintext(HElibPlaintext* plain);

// Encryption/Decryption
/// Encrypt plaintext
HElibCiphertext* helib_encrypt(
    HElibPublicKey* pk,
    HElibPlaintext* plain
);

/// Decrypt ciphertext
HElibPlaintext* helib_decrypt(
    HElibSecretKey* sk,
    HElibCiphertext* cipher
);

void helib_destroy_ciphertext(HElibCiphertext* cipher);

// Homomorphic Operations

/// Homomorphic addition: result = a + b
HElibCiphertext* helib_add(
    HElibCiphertext* a,
    HElibCiphertext* b
);

/// Homomorphic multiplication: result = a * b
HElibCiphertext* helib_multiply(
    HElibCiphertext* a,
    HElibCiphertext* b
);

/// Homomorphic subtraction: result = a - b
HElibCiphertext* helib_subtract(
    HElibCiphertext* a,
    HElibCiphertext* b
);

// Utility Functions

/// Get noise budget (for debugging)
int helib_noise_budget(HElibSecretKey* sk, HElibCiphertext* cipher);

#ifdef __cplusplus
}
#endif

#endif