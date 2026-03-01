package com.fyp.hebench.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.fyp.hebench.model.BenchmarkRequest;
import com.fyp.hebench.model.BenchmarkResponse;
import com.fyp.hebench.model.ComparisonResponse;
import com.fyp.hebench.model.PredictRequest;
import com.fyp.hebench.model.PredictResponse;
import com.fyp.hebench.service.GrpcClientService;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class BenchmarkController {

    @Autowired
    private GrpcClientService grpcClientService;

    @PostMapping("/benchmark/run")
    public ResponseEntity<BenchmarkResponse> runBenchmark(@RequestBody BenchmarkRequest request) {
        try {
            BenchmarkResponse response = grpcClientService.runBenchmark(
                request.getLibrary(), 
                request.getNumOperations()
            );
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @PostMapping("/benchmark/compare")
    public ResponseEntity<ComparisonResponse> compareBenchmarks(@RequestBody BenchmarkRequest request) {
        try {
            ComparisonResponse response = grpcClientService.runComparisonBenchmark(
                request.getNumOperations()
            );
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK");
    }

    /**
     * POST /api/predict - Run encrypted MNIST digit prediction
     * 
     * This is the main endpoint for the React frontend.
     * The user draws a digit → frontend sends 784 pixels here →
     * Spring Boot forwards to Rust gRPC server (inside Docker) →
     * Rust encrypts, runs CNN on encrypted data, decrypts →
     * Returns predicted digit with confidence and timing.
     * 
     * Request body example:
     *   {"pixels": [0, 0, 0, ..., 255, ..., 0], "scaleFactor": 1000}
     * 
     * Response example:
     *   {"predictedDigit": 7, "confidence": 0.43, "totalMs": 48.2, ...}
     */
    @PostMapping("/predict")
    public ResponseEntity<?> predictDigit(@RequestBody PredictRequest request) {
        // Validate: pixels must be exactly 784 values (28×28 image)
        if (request.getPixels() == null || request.getPixels().size() != 784) {
            return ResponseEntity.badRequest().body(
                java.util.Map.of(
                    "error", "pixels must contain exactly 784 values (28x28 image)",
                    "received", request.getPixels() == null ? 0 : request.getPixels().size()
                )
            );
        }

        // Default scale factor to 1000 if not provided
        long scaleFactor = request.getScaleFactor() > 0 ? request.getScaleFactor() : 1000;

        try {
            // Call the gRPC service which forwards to Rust server in Docker
            PredictResponse response = grpcClientService.predictDigit(
                request.getPixels(), 
                scaleFactor
            );
            return ResponseEntity.ok(response);
        } catch (io.grpc.StatusRuntimeException e) {
            // gRPC-specific error (e.g., Docker container not running)
            return ResponseEntity.internalServerError().body(
                java.util.Map.of(
                    "error", "Failed to connect to HE inference engine",
                    "detail", e.getStatus().getDescription() != null 
                        ? e.getStatus().getDescription() 
                        : e.getStatus().getCode().name(),
                    "hint", "Make sure Docker container is running: docker run -p 50051:50051 he-benchmark-server"
                )
            );
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                java.util.Map.of(
                    "error", "Prediction failed",
                    "detail", e.getMessage() != null ? e.getMessage() : "Unknown error"
                )
            );
        }
    }
}
