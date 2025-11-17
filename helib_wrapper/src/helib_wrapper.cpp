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
