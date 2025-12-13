// grpc_server/src/main.rs
use tonic::{transport::Server, Request, Response, Status};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Include the generated proto code
pub mod he_service {
    tonic::include_proto!("he_service");
}

use he_service::{
    he_service_server::{HeService, HeServiceServer},
    *,
};

// Session data structure to store encryption context
#[derive(Debug)]
struct SessionData {
    library: String,
    // TODO: Add your HE context objects here
    // For now, we'll just store the library name
}

// Our gRPC service implementation
pub struct HEServiceImpl {
    sessions: Arc<Mutex<HashMap<String, SessionData>>>,
}

impl HEServiceImpl {
    fn new() -> Self {
        HEServiceImpl {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// Implement the gRPC service methods
#[tonic::async_trait]
impl HeService for HEServiceImpl {
    async fn generate_keys(
        &self,
        request: Request<GenerateKeysRequest>,
    ) -> Result<Response<GenerateKeysResponse>, Status> {
        let req = request.into_inner();
        
        println!("Received GenerateKeys request for library: {}", req.library);
        
        // Validate library name
        if !["SEAL", "HELib", "OpenFHE"].contains(&req.library.as_str()) {
            return Err(Status::invalid_argument(
                "Library must be one of: SEAL, HELib, OpenFHE"
            ));
        }
        
        // Generate a unique session ID
        let session_id = uuid::Uuid::new_v4().to_string();
        
        // Store session data
        let session = SessionData {
            library: req.library.clone(),
        };
        
        self.sessions.lock().unwrap().insert(session_id.clone(), session);
        
        // TODO: Actually generate keys using your HE library
        // For now, return a dummy response
        let response = GenerateKeysResponse {
            session_id: session_id.clone(),
            public_key: vec![1, 2, 3, 4], // Dummy key
            status: format!("Keys generated for {} (session: {})", req.library, session_id),
        };
        
        println!("✓ Session created: {}", session_id);
        
        Ok(Response::new(response))
    }

    async fn encrypt(
        &self,
        request: Request<EncryptRequest>,
    ) -> Result<Response<EncryptResponse>, Status> {
        let req = request.into_inner();
        
        println!("Received Encrypt request for session: {}", req.session_id);
        println!("Values to encrypt: {:?}", req.values);
        
        // Check if session exists
        let sessions = self.sessions.lock().unwrap();
        if !sessions.contains_key(&req.session_id) {
            return Err(Status::not_found("Session not found"));
        }
        
        // TODO: Actually encrypt using your HE library
        let response = EncryptResponse {
            ciphertext: vec![5, 6, 7, 8], // Dummy ciphertext
            status: "Encrypted successfully".to_string(),
        };
        
        println!("✓ Encryption complete");
        
        Ok(Response::new(response))
    }

    async fn decrypt(
        &self,
        request: Request<DecryptRequest>,
    ) -> Result<Response<DecryptResponse>, Status> {
        let req = request.into_inner();
        
        println!("Received Decrypt request for session: {}", req.session_id);
        
        // Check if session exists
        let sessions = self.sessions.lock().unwrap();
        if !sessions.contains_key(&req.session_id) {
            return Err(Status::not_found("Session not found"));
        }
        
        // TODO: Actually decrypt using your HE library
        let response = DecryptResponse {
            values: vec![42], // Dummy decrypted value
            status: "Decrypted successfully".to_string(),
        };
        
        println!("✓ Decryption complete");
        
        Ok(Response::new(response))
    }

    async fn add(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        
        println!("Received Add request for session: {}", req.session_id);
        
        // TODO: Implement homomorphic addition
        let response = BinaryOpResponse {
            result_ciphertext: vec![9, 10, 11, 12],
            status: "Addition complete".to_string(),
        };
        
        Ok(Response::new(response))
    }

    async fn multiply(
        &self,
        request: Request<BinaryOpRequest>,
    ) -> Result<Response<BinaryOpResponse>, Status> {
        let req = request.into_inner();
        
        println!("Received Multiply request for session: {}", req.session_id);
        
        // TODO: Implement homomorphic multiplication
        let response = BinaryOpResponse {
            result_ciphertext: vec![13, 14, 15, 16],
            status: "Multiplication complete".to_string(),
        };
        
        Ok(Response::new(response))
    }

    async fn run_benchmark(
        &self,
        request: Request<BenchmarkRequest>,
    ) -> Result<Response<BenchmarkResponse>, Status> {
        let req = request.into_inner();
        
        println!("Running benchmark for library: {}", req.library);
        
        // TODO: Run actual benchmark using your existing benchmark code
        let response = BenchmarkResponse {
            key_gen_time_ms: 100.5,
            encryption_time_ms: 50.2,
            addition_time_ms: 20.1,
            multiplication_time_ms: 30.3,
            decryption_time_ms: 45.7,
            status: format!("Benchmark complete for {}", req.library),
        };
        
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let service = HEServiceImpl::new();

    println!("╔════════════════════════════════════════════╗");
    println!("║   HE gRPC Server Starting...              ║");
    println!("╚════════════════════════════════════════════╝");
    println!(" Listening on: {}", addr);
    println!(" Ready to accept gRPC connections");
    println!();
    println!("Available services:");
    println!("  - GenerateKeys");
    println!("  - Encrypt / Decrypt");
    println!("  - Add / Multiply (homomorphic)");
    println!("  - RunBenchmark");
    println!();

    Server::builder()
        .add_service(HeServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
