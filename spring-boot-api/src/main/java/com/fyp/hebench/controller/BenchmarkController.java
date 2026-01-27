package com.fyp.hebench.controller;

import com.fyp.hebench.model.BenchmarkRequest;
import com.fyp.hebench.model.BenchmarkResponse;
import com.fyp.hebench.model.ComparisonResponse;
import com.fyp.hebench.service.GrpcClientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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
}
