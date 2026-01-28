package com.fyp.hebench.service;

import org.springframework.stereotype.Service;

import com.fyp.hebench.grpc.BenchmarkRequest;      // Generated from .proto message
import com.fyp.hebench.grpc.BenchmarkResponse;     // Generated from .proto message
import com.fyp.hebench.grpc.ComparisonBenchmarkResponse;  // Generated from .proto message
import com.fyp.hebench.grpc.HEServiceGrpc;         // Generated gRPC client stub

import io.grpc.ManagedChannel;           // gRPC network connection handler
import io.grpc.ManagedChannelBuilder;    // Builder to create the channel
import jakarta.annotation.PostConstruct; // Runs method after bean is created
import jakarta.annotation.PreDestroy;    // Runs method before bean is destroyed

/**
 * GrpcClientService - The "Translator" between Spring Boot and Rust gRPC Server
 * 
 * This service acts as a bridge:
 *   1. Receives Java objects from BenchmarkController
 *   2. Converts them to Protobuf format (binary, efficient)
 *   3. Sends to Rust gRPC server on port 50051 (inside Docker)
 *   4. Receives Protobuf response from Rust
 *   5. Converts back to Java objects for JSON response
 * 
 * @Service annotation tells Spring to create ONE instance of this class
 * and inject it wherever needed (like in BenchmarkController)
 */
@Service
public class GrpcClientService {

    // ManagedChannel = the network connection to the Rust gRPC server
    // Think of it like a phone line that stays open
    private ManagedChannel channel;
    
    // Stub = the actual gRPC client that makes calls to the server
    // "BlockingStub" means it waits for response before continuing (synchronous)
    // There's also "FutureStub" (async) and "Stub" (callback-based)
    private HEServiceGrpc.HEServiceBlockingStub stub;
    
    /**
     * @PostConstruct = Spring calls this method automatically AFTER creating the service
     * This is where we establish the connection to the Rust server
     */
    @PostConstruct
    public void init() {
        // Create a channel (connection) to the Rust gRPC server
        // "localhost" = same machine (Docker exposes port to host)
        // 50051 = the port where Rust gRPC server listens
        // .usePlaintext() = no TLS encryption (ok for local development)
        channel = ManagedChannelBuilder.forAddress("localhost", 50051)
                .usePlaintext()
                .build();
        
        // Create the stub (client) using this channel
        // HEServiceGrpc was auto-generated from he_service.proto
        stub = HEServiceGrpc.newBlockingStub(channel);
    }

    /**
     * @PreDestroy = Spring calls this method automatically BEFORE shutting down
     * Clean up: close the network connection gracefully
     */
    @PreDestroy
    public void shutdown() {
        if (channel != null) {
            channel.shutdown();
        }
    }

    /**
     * Run benchmark for a SINGLE HE library (SEAL, HELib, or OpenFHE)
     * 
     * Flow:
     *   Controller calls this with ("SEAL", 10)
     *        |
     *   Build Protobuf request
     *        |
     *   stub.runBenchmark() sends to Rust via gRPC
     *        |
     *   Rust runs actual HE operations in Docker
     *        |
     *   Receive Protobuf response
     *        |
     *   Convert to Java POJO for JSON response
     * 
     * @param library - Which library: "SEAL", "HELib", or "OpenFHE"
     * @param numOperations - How many times to repeat each operation (for averaging)
     * @return BenchmarkResponse with timing metrics
     */
    public com.fyp.hebench.model.BenchmarkResponse runBenchmark(String library, int numOperations) {
        // Step 1: Build Protobuf request using the generated Builder pattern
        // BenchmarkRequest.newBuilder() creates a mutable builder
        // .setLibrary() and .setNumOperations() set the fields
        // .build() creates an immutable Protobuf object
        BenchmarkRequest request = BenchmarkRequest.newBuilder()
                .setLibrary(library)
                .setNumOperations(numOperations)
                .build();
        
        // Step 2: Call the Rust server via gRPC
        // stub.runBenchmark() sends the request over the network
        // This BLOCKS until Rust responds (because we use BlockingStub)
        // The actual HE operations happen inside Docker right now!
        BenchmarkResponse result = stub.runBenchmark(request);
        
        // Step 3: Convert Protobuf response to our Java POJO
        // Why? Because Spring converts POJOs to JSON automatically
        // Protobuf objects have complex serialization that doesn't map nicely to JSON
        return new com.fyp.hebench.model.BenchmarkResponse(
                library,
                result.getKeyGenTimeMs(),
                result.getEncryptionTimeMs(),
                result.getAdditionTimeMs(),
                result.getMultiplicationTimeMs(),
                result.getDecryptionTimeMs(),
                result.getTotalTimeMs(),
                result.getStatus().equals("success"),
                result.getStatus().equals("success") ? "" : result.getStatus()
        );
    }

    /**
     * Run benchmark for ALL THREE HE libraries and compare them
     * 
     * This is called when user wants to compare SEAL vs HELib vs OpenFHE
     * 
     * Flow:
     *   Controller calls this with (10) operations
     *        |
     *   Build Protobuf request with library="ALL"
     *        |
     *   Rust runs benchmarks for all 3 libraries sequentially
     *        |
     *   Receive ComparisonBenchmarkResponse with 3 sets of results
     *        |
     *   Convert each library's results to Java POJOs
     *        |
     *   Return list of LibraryResult objects
     * 
     * @param numOperations - How many times to repeat each operation
     * @return ComparisonResponse containing results for all 3 libraries
     */
    public com.fyp.hebench.model.ComparisonResponse runComparisonBenchmark(int numOperations) {
        // Build request - "ALL" tells Rust to benchmark all 3 libraries
        BenchmarkRequest request = BenchmarkRequest.newBuilder()
                .setLibrary("ALL")
                .setNumOperations(numOperations)
                .build();
        
        // Call Rust server - this takes longer because it runs 3 benchmarks
        ComparisonBenchmarkResponse result = stub.runComparisonBenchmark(request);
        
        // Create list to hold results for each library
        java.util.List<com.fyp.hebench.model.LibraryResult> libraryResults = new java.util.ArrayList<>();
        
        // Extract SEAL results (if present)
        // hasSeal() checks if the Rust server included SEAL results
        if (result.hasSeal()) {
            BenchmarkResponse seal = result.getSeal();
            libraryResults.add(new com.fyp.hebench.model.LibraryResult(
                    "SEAL",
                    seal.getKeyGenTimeMs(),
                    seal.getEncryptionTimeMs(),
                    seal.getAdditionTimeMs(),
                    seal.getMultiplicationTimeMs(),
                    seal.getDecryptionTimeMs(),
                    seal.getTotalTimeMs(),
                    seal.getStatus().equals("success"),
                    seal.getStatus().equals("success") ? "" : seal.getStatus()
            ));
        }
        
        // Extract HELib results (if present)
        if (result.hasHelib()) {
            BenchmarkResponse helib = result.getHelib();
            libraryResults.add(new com.fyp.hebench.model.LibraryResult(
                    "HELib",
                    helib.getKeyGenTimeMs(),
                    helib.getEncryptionTimeMs(),
                    helib.getAdditionTimeMs(),
                    helib.getMultiplicationTimeMs(),
                    helib.getDecryptionTimeMs(),
                    helib.getTotalTimeMs(),
                    helib.getStatus().equals("success"),
                    helib.getStatus().equals("success") ? "" : helib.getStatus()
            ));
        }
        
        // Extract OpenFHE results (if present)
        if (result.hasOpenfhe()) {
            BenchmarkResponse openfhe = result.getOpenfhe();
            libraryResults.add(new com.fyp.hebench.model.LibraryResult(
                    "OpenFHE",
                    openfhe.getKeyGenTimeMs(),
                    openfhe.getEncryptionTimeMs(),
                    openfhe.getAdditionTimeMs(),
                    openfhe.getMultiplicationTimeMs(),
                    openfhe.getDecryptionTimeMs(),
                    openfhe.getTotalTimeMs(),
                    openfhe.getStatus().equals("success"),
                    openfhe.getStatus().equals("success") ? "" : openfhe.getStatus()
            ));
        }
        
        // Return wrapped in ComparisonResponse for JSON serialization
        return new com.fyp.hebench.model.ComparisonResponse(libraryResults);
    }
}
