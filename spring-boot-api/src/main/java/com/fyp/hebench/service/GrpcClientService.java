package com.fyp.hebench.service;

import com.fyp.hebench.grpc.BenchmarkRequest;
import com.fyp.hebench.grpc.BenchmarkResponse;
import com.fyp.hebench.grpc.ComparisonBenchmarkResponse;
import com.fyp.hebench.grpc.HEServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

@Service
public class GrpcClientService {

    private ManagedChannel channel;
    private HEServiceGrpc.HEServiceBlockingStub stub;

    @PostConstruct
    public void init() {
        channel = ManagedChannelBuilder.forAddress("localhost", 50051)
                .usePlaintext()
                .build();
        stub = HEServiceGrpc.newBlockingStub(channel);
    }

    @PreDestroy
    public void shutdown() {
        if (channel != null) {
            channel.shutdown();
        }
    }

    public com.fyp.hebench.model.BenchmarkResponse runBenchmark(String library, int numOperations) {
        BenchmarkRequest request = BenchmarkRequest.newBuilder()
                .setLibrary(library)
                .setNumOperations(numOperations)
                .build();
        
        BenchmarkResponse result = stub.runBenchmark(request);
        
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

    public com.fyp.hebench.model.ComparisonResponse runComparisonBenchmark(int numOperations) {
        BenchmarkRequest request = BenchmarkRequest.newBuilder()
                .setLibrary("ALL")
                .setNumOperations(numOperations)
                .build();
        
        ComparisonBenchmarkResponse result = stub.runComparisonBenchmark(request);
        
        java.util.List<com.fyp.hebench.model.LibraryResult> libraryResults = new java.util.ArrayList<>();
        
        // Add SEAL results
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
        
        // Add HELib results
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
        
        // Add OpenFHE results
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
        
        return new com.fyp.hebench.model.ComparisonResponse(libraryResults);
    }
}
