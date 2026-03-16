package com.fyp.hebench.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

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
                request.getNumOperations(),
                request.getTestValues()
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
                request.getNumOperations(),
                request.getTestValues()
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

    /**
     * POST /api/predict/stream - SSE streaming endpoint for encrypted MNIST inference
     * 
     * Returns a Server-Sent Events stream with real-time layer-by-layer progress.
     * Each event is JSON with: {eventType, layer, layerMs, elapsedMs, result}
     * 
     * The frontend uses fetch() with ReadableStream to consume these events.
     */
    @PostMapping("/predict/stream")
    public SseEmitter predictDigitStream(@RequestBody PredictRequest request) {
        // 5-minute timeout to match the gRPC deadline
        SseEmitter emitter = new SseEmitter(300_000L);

        // Validate pixels
        if (request.getPixels() == null || request.getPixels().size() != 784) {
            SseEmitter errorEmitter = new SseEmitter(5_000L);
            try {
                errorEmitter.send(SseEmitter.event()
                    .name("error")
                    .data("{\"error\":\"pixels must contain exactly 784 values\"}"));
                errorEmitter.complete();
            } catch (Exception e) {
                errorEmitter.completeWithError(e);
            }
            return errorEmitter;
        }

        long scaleFactor = request.getScaleFactor() > 0 ? request.getScaleFactor() : 1000;

        // Run the streaming gRPC call on a background thread so we don't block
        new Thread(() -> {
            try {
                java.util.Iterator<com.fyp.hebench.grpc.PredictProgressEvent> stream = 
                    grpcClientService.predictDigitStream(request.getPixels(), scaleFactor);

                while (stream.hasNext()) {
                    com.fyp.hebench.grpc.PredictProgressEvent event = stream.next();
                    
                    // Build a JSON object for the SSE event
                    String json = buildProgressJson(event);
                    
                    emitter.send(SseEmitter.event()
                        .name(event.getEventType())
                        .data(json));
                }

                emitter.complete();
            } catch (io.grpc.StatusRuntimeException e) {
                try {
                    emitter.send(SseEmitter.event()
                        .name("error")
                        .data("{\"error\":\"gRPC connection failed: " + 
                              e.getStatus().getCode().name() + "\"}"));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            } catch (Exception e) {
                try {
                    emitter.send(SseEmitter.event()
                        .name("error")
                        .data("{\"error\":\"" + 
                              (e.getMessage() != null ? e.getMessage().replace("\"", "'") : "Unknown error") + 
                              "\"}"));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            }
        }).start();

        return emitter;
    }

    /**
     * Convert a gRPC PredictProgressEvent to JSON string for SSE
     */
    private String buildProgressJson(com.fyp.hebench.grpc.PredictProgressEvent event) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"eventType\":\"").append(event.getEventType()).append("\"");
        sb.append(",\"layer\":\"").append(event.getLayer()).append("\"");
        sb.append(",\"layerMs\":").append(event.getLayerMs());
        sb.append(",\"elapsedMs\":").append(event.getElapsedMs());

        if ("complete".equals(event.getEventType()) && event.hasResult()) {
            com.fyp.hebench.grpc.PredictResponse r = event.getResult();
            sb.append(",\"result\":{");
            sb.append("\"predictedDigit\":").append(r.getPredictedDigit());
            sb.append(",\"confidence\":").append(r.getConfidence());
            sb.append(",\"status\":\"").append(r.getStatus()).append("\"");
            sb.append(",\"logits\":[");
            for (int i = 0; i < r.getLogitsList().size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(r.getLogitsList().get(i));
            }
            sb.append("]");
            sb.append(",\"encryptionMs\":").append(r.getEncryptionMs());
            sb.append(",\"conv1Ms\":").append(r.getConv1Ms());
            sb.append(",\"bias1Ms\":").append(r.getBias1Ms());
            sb.append(",\"act1Ms\":").append(r.getAct1Ms());
            sb.append(",\"pool1Ms\":").append(r.getPool1Ms());
            sb.append(",\"conv2Ms\":").append(r.getConv2Ms());
            sb.append(",\"bias2Ms\":").append(r.getBias2Ms());
            sb.append(",\"act2Ms\":").append(r.getAct2Ms());
            sb.append(",\"pool2Ms\":").append(r.getPool2Ms());
            sb.append(",\"fcMs\":").append(r.getFcMs());
            sb.append(",\"biasFcMs\":").append(r.getBiasFcMs());
            sb.append(",\"decryptionMs\":").append(r.getDecryptionMs());
            sb.append(",\"totalMs\":").append(r.getTotalMs());
            sb.append(",\"floatModelAccuracy\":").append(r.getFloatModelAccuracy());
            sb.append("}");
        }

        sb.append("}");
        return sb.toString();
    }
}
