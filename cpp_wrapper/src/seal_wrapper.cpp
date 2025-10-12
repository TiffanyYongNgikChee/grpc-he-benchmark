#include "../include/seal_wrapper.h"
#include "seal/seal.h"
#include <memory>
#include <stdexcept>

using namespace seal;
using namespace std;

// ============================================
// Opaque Struct Definitions
// ============================================
struct SEALContext {
    shared_ptr<seal::SEALContext> context;
    shared_ptr<KeyGenerator> keygen;
    PublicKey public_key;
    SecretKey secret_key;
};

struct SEALEncryptor {
    unique_ptr<Encryptor> encryptor;
};

struct SEALDecryptor {
    unique_ptr<Decryptor> decryptor;
};

struct SEALCiphertext {
    Ciphertext ciphertext;
};

struct SEALPlaintext {
    Plaintext plaintext;
};

// ============================================
// Context Management Implementation
// ============================================
extern "C" SEALContext* seal_create_context(
    uint64_t poly_modulus_degree,
    const uint64_t* coeff_modulus_bits, 
    size_t coeff_modulus_size,
    uint64_t plain_modulus_value
) {
    try {
        // Create encryption parameters
        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        
        // Set coefficient modulus
        vector<Modulus> coeff_modulus;
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            coeff_modulus.push_back(Modulus(coeff_modulus_bits[i]));
        }
        parms.set_coeff_modulus(coeff_modulus);
        
        // Set plaintext modulus
        parms.set_plain_modulus(plain_modulus_value);
        
        // Create context
        auto context = make_shared<seal::SEALContext>(parms);
        
        // Generate keys
        KeyGenerator keygen(*context);
        
        // Allocate and populate result
        SEALContext* result = new SEALContext();
        result->context = context;
        result->keygen = make_shared<KeyGenerator>(keygen);
        result->public_key = keygen.create_public_key();
        result->secret_key = keygen.secret_key();
        
        return result;
    } catch (const exception& e) {
        // Error handling
        return nullptr;
    }
}

extern "C" void seal_destroy_context(SEALContext* ctx) {
    if (ctx) delete ctx;
}

// ============================================
// Encryptor Implementation
// ============================================
extern "C" SEALEncryptor* seal_create_encryptor(
    SEALContext* ctx,
    const uint8_t* public_key,
    size_t public_key_size
) {
    try {
        if (!ctx) return nullptr;
        
        SEALEncryptor* enc = new SEALEncryptor();
        enc->encryptor = make_unique<Encryptor>(
            *ctx->context, 
            ctx->public_key
        );
        
        return enc;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void seal_destroy_encryptor(SEALEncryptor* enc) {
    if (enc) delete enc;
}

// ============================================
// Decryptor Implementation
// ============================================
extern "C" SEALDecryptor* seal_create_decryptor(
    SEALContext* ctx,
    const uint8_t* secret_key,
    size_t secret_key_size
) {
    try {
        if (!ctx) return nullptr;
        
        SEALDecryptor* dec = new SEALDecryptor();
        dec->decryptor = make_unique<Decryptor>(
            *ctx->context,
            ctx->secret_key
        );
        
        return dec;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void seal_destroy_decryptor(SEALDecryptor* dec) {
    if (dec) delete dec;
}

// ============================================
// Plaintext Operations
// ============================================
extern "C" SEALPlaintext* seal_create_plaintext(const char* hex_string) {
    try {
        SEALPlaintext* plain = new SEALPlaintext();
        plain->plaintext = Plaintext(hex_string);
        return plain;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void seal_destroy_plaintext(SEALPlaintext* plain) {
    if (plain) delete plain;
}

extern "C" const char* seal_plaintext_to_string(SEALPlaintext* plain) {
    if (!plain) return nullptr;
    // Note: This leaks memory - for demo only
    // In production, need better string management
    string str = plain->plaintext.to_string();
    char* result = new char[str.length() + 1];
    strcpy(result, str.c_str());
    return result;
}

// ============================================
// Encryption Implementation
// ============================================
extern "C" SEALCiphertext* seal_encrypt(
    SEALEncryptor* encryptor,
    SEALPlaintext* plaintext
) {
    try {
        if (!encryptor || !plaintext) return nullptr;
        
        SEALCiphertext* cipher = new SEALCiphertext();
        encryptor->encryptor->encrypt(
            plaintext->plaintext,
            cipher->ciphertext
        );
        
        return cipher;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void seal_destroy_ciphertext(SEALCiphertext* cipher) {
    if (cipher) delete cipher;
}

// ============================================
// Decryption Implementation
// ============================================
extern "C" SEALPlaintext* seal_decrypt(
    SEALDecryptor* decryptor,
    SEALCiphertext* ciphertext
) {
    try {
        if (!decryptor || !ciphertext) return nullptr;
        
        SEALPlaintext* plain = new SEALPlaintext();
        decryptor->decryptor->decrypt(
            ciphertext->ciphertext,
            plain->plaintext
        );
        
        return plain;
    } catch (...) {
        return nullptr;
    }
}

// ============================================
// Homomorphic Operations
// ============================================
extern "C" SEALCiphertext* seal_add(
    SEALContext* ctx,
    SEALCiphertext* a,
    SEALCiphertext* b
) {
    try {
        if (!ctx || !a || !b) return nullptr;
        
        Evaluator evaluator(*ctx->context);
        SEALCiphertext* result = new SEALCiphertext();
        evaluator.add(
            a->ciphertext,
            b->ciphertext,
            result->ciphertext
        );
        
        return result;
    } catch (...) {
        return nullptr;
    }
}

extern "C" SEALCiphertext* seal_multiply(
    SEALContext* ctx,
    SEALCiphertext* a,
    SEALCiphertext* b
) {
    try {
        if (!ctx || !a || !b) return nullptr;
        
        Evaluator evaluator(*ctx->context);
        SEALCiphertext* result = new SEALCiphertext();
        evaluator.multiply(
            a->ciphertext,
            b->ciphertext,
            result->ciphertext
        );
        
        return result;
    } catch (...) {
        return nullptr;
    }
}