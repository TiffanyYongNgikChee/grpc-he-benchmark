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
