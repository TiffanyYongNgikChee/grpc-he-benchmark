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

