#ifndef OPENFHE_WRAPPER_H
#define OPENFHE_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque Pointers (hide C++ from Rust)
typedef struct OpenFHEContext OpenFHEContext;
typedef struct OpenFHEKeyPair OpenFHEKeyPair;
typedef struct OpenFHEPlaintext OpenFHEPlaintext;
typedef struct OpenFHECiphertext OpenFHECiphertext;

// Context Management
/// Create a new OpenFHE BFV context
/// @param plaintext_modulus: Plaintext modulus (e.g., 65537)
/// @param multiplicative_depth: Multiplicative depth (e.g., 2)
/// @return Pointer to context or NULL on failure
OpenFHEContext* openfhe_create_bfv_context(
    uint64_t plaintext_modulus,
    uint32_t multiplicative_depth
);

/// Destroy context and free memory
void openfhe_destroy_context(OpenFHEContext* ctx);

// Key Management
/// Generate public/private k ey pair
/// @param ctx: OpenFHE context
/// @return Pointer to key pair or NULL on failure
OpenFHEKeyPair* openfhe_generate_keypair(OpenFHEContext* ctx);

/// Destroy key pair and free memory
void openfhe_destroy_keypair(OpenFHEKeyPair* keypair);

// Plaintext Operations

/// Create plaintext from integer vector
/// @param ctx: OpenFHE context
/// @param values: Array of integers
/// @param length: Number of integers
/// @return Pointer to plaintext or NULL on failure
OpenFHEPlaintext* openfhe_create_plaintext(
    OpenFHEContext* ctx,
    const int64_t* values,
    size_t length
);

/// Destroy plaintext and free memory
void openfhe_destroy_plaintext(OpenFHEPlaintext* plain);

/// Get plaintext values
/// @param plain: Plaintext to extract from
/// @param out_values: Output buffer (caller allocates)
/// @param out_length: Pointer to store actual length
/// @return true on success, false on failure
bool openfhe_get_plaintext_values(
    OpenFHEPlaintext* plain,
    int64_t* out_values,
    size_t* out_length
);

// Encryption/Decryption

/// Encrypt a plaintext
/// @param ctx: OpenFHE context
/// @param keypair: Key pair (uses public key)
/// @param plain: Plaintext to encrypt
/// @return Pointer to ciphertext or NULL on failure
OpenFHECiphertext* openfhe_encrypt(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHEPlaintext* plain
);

/// Decrypt a ciphertext
/// @param ctx: OpenFHE context
/// @param keypair: Key pair (uses secret key)
/// @param cipher: Ciphertext to decrypt
/// @return Pointer to plaintext or NULL on failure
OpenFHEPlaintext* openfhe_decrypt(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* cipher
);

/// Destroy ciphertext and free memory
void openfhe_destroy_ciphertext(OpenFHECiphertext* cipher);

// Homomorphic Operations
/// Add two ciphertexts
/// @param ctx: OpenFHE context
/// @param a: First ciphertext
/// @param b: Second ciphertext
/// @return Pointer to result ciphertext or NULL on failure
OpenFHECiphertext* openfhe_add(
    OpenFHEContext* ctx,
    OpenFHECiphertext* a,
    OpenFHECiphertext* b
);

/// Multiply two ciphertexts
/// @param ctx: OpenFHE context
/// @param keypair: Key pair (for relinearization)
/// @param a: First ciphertext
/// @param b: Second ciphertext
/// @return Pointer to result ciphertext or NULL on failure
OpenFHECiphertext* openfhe_multiply(
    OpenFHEContext* ctx,
    OpenFHEKeyPair* keypair,
    OpenFHECiphertext* a,
    OpenFHECiphertext* b
);

/// Subtract two ciphertexts
/// @param ctx: OpenFHE context
/// @param a: First ciphertext
/// @param b: Second ciphertext
/// @return Pointer to result ciphertext or NULL on failure
OpenFHECiphertext* openfhe_subtract(
    OpenFHEContext* ctx,
    OpenFHECiphertext* a,
    OpenFHECiphertext* b
);

// Error Handling
/// Get last error message
/// @return Error message string (valid until next call)
const char* openfhe_get_last_error();


#ifdef __cplusplus
}
#endif

#endif