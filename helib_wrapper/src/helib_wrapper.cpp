#include "../include/helib_wrapper.h"
#include <helib/helib.h>
#include <memory>
#include <iostream>

using namespace helib;
using namespace std;


// Opaque Struct Implementations
struct HElibContext {
    unique_ptr<Context> context;
};

struct HElibSecretKey {
    unique_ptr<SecKey> secretKey;
    HElibContext* ctx; // Keep reference to context
};

struct HElibPublicKey {
    const PubKey* publicKey; // Non-owning pointer
};

struct HElibCiphertext {
    unique_ptr<Ctxt> ctxt;
};

struct HElibPlaintext {
    long value;
};

// Context Management Implementation
extern "C" HElibContext* helib_create_context(
    unsigned long m,
    unsigned long p,
    unsigned long r
) {
    try {
        // Create context builder
        auto contextBuilder = ContextBuilder<BGV>()
            .m(m)           // Cyclotomic polynomial
            .p(p)           // Plaintext modulus
            .r(r)           // Lifting
            .bits(300)      // Bit precision
            .c(2);          // Columns in key-switching matrix
        
        // Build context
        auto context = contextBuilder.build();
        
        // Wrap in our struct
        HElibContext* result = new HElibContext();
        result->context = make_unique<Context>(move(context));
        
        return result;
        
    } catch (const exception& e) {
        cerr << "HElib context creation failed: " << e.what() << endl;
        return nullptr;
    }
}

extern "C" void helib_destroy_context(HElibContext* ctx) {
    if (ctx) delete ctx;
}

// Key Management Implementation
extern "C" HElibSecretKey* helib_generate_secret_key(HElibContext* ctx) {
    try {
        if (!ctx || !ctx->context) return nullptr;
        
        HElibSecretKey* sk = new HElibSecretKey();
        sk->ctx = ctx;
        
        // Create secret key
        sk->secretKey = make_unique<SecKey>(*ctx->context);
        sk->secretKey->GenSecKey(); // Generate secret key polynomial
        
        // Add key-switching matrices (required for multiplication)
        addSome1DMatrices(*sk->secretKey);
        
        return sk;
        
    } catch (const exception& e) {
        cerr << "Secret key generation failed: " << e.what() << endl;
        return nullptr;
    }
}

extern "C" void helib_destroy_secret_key(HElibSecretKey* sk) {
    if (sk) delete sk;
}

extern "C" HElibPublicKey* helib_get_public_key(HElibSecretKey* sk) {
    try {
        if (!sk || !sk->secretKey) return nullptr;
        
        HElibPublicKey* pk = new HElibPublicKey();
        pk->publicKey = sk->secretKey.get(); // SecKey inherits from PubKey
        
        return pk;
        
    } catch (...) {
        return nullptr;
    }
}

extern "C" void helib_destroy_public_key(HElibPublicKey* pk) {
    if (pk) delete pk;
}