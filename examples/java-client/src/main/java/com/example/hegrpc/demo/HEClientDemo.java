package com.example.hegrpc.demo;

import java.util.List;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.example.hegrpc.service.HEClientService;
import com.google.protobuf.ByteString;

import he_service.HeService.*;
import he_service.HeService.BinaryOpResponse;
import he_service.HeService.ComparisonBenchmarkResponse;
import he_service.HeService.DecryptResponse;
import he_service.HeService.EncryptResponse;
import he_service.HeService.GenerateKeysResponse;

/**
 * Demo runner that executes when Spring Boot starts.
 * Shows all available HE operations.
 */
@Component
public class HEClientDemo implements CommandLineRunner {

    private final HEClientService heClient;

    public HEClientDemo(HEClientService heClient) {
        this.heClient = heClient;
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("\n🔐 Homomorphic Encryption gRPC Client Demo\n");
        
        demoSEAL();
        demoComparisonBenchmark();
        
        System.out.println("\n✅ Demo completed!\n");
    }

    private void demoSEAL() {
        System.out.println("=== Testing SEAL ===\n");

        // 1. Generate keys
        System.out.println("1. Generating keys...");
        GenerateKeysResponse keys = heClient.generateKeys("SEAL", 8192);
        String sessionId = keys.getSessionId();
        System.out.println("   Session ID: " + sessionId);

        // 2. Encrypt values
        List<Long> values = List.of(10L, 20L, 30L);
        System.out.println("\n2. Encrypting: " + values);
        EncryptResponse encrypted = heClient.encrypt(sessionId, values);
        ByteString ciphertext = encrypted.getCiphertext();
        System.out.println("   Ciphertext size: " + ciphertext.size() + " bytes");

        // 3. Decrypt
        System.out.println("\n3. Decrypting...");
        DecryptResponse decrypted = heClient.decrypt(sessionId, ciphertext);
        System.out.println("   Result: " + decrypted.getValuesList());

        // 4. Homomorphic addition
        List<Long> values2 = List.of(5L, 10L, 15L);
        System.out.println("\n4. Homomorphic Addition:");
        System.out.println("   " + values + " + " + values2);
        
        EncryptResponse encrypted2 = heClient.encrypt(sessionId, values2);
        BinaryOpResponse addResult = heClient.add(sessionId, 
            ciphertext, encrypted2.getCiphertext());
        DecryptResponse addDecrypted = heClient.decrypt(sessionId, 
            addResult.getResultCiphertext());
        System.out.println("   = " + addDecrypted.getValuesList());

        System.out.println("\n✓ SEAL demo complete!\n");
    }

    private void demoComparisonBenchmark() {
        System.out.println("=== Comparison Benchmark ===\n");

        ComparisonBenchmarkResponse result = heClient.runComparisonBenchmark(20);

        System.out.println("SEAL:    " + result.getSeal().getTotalTimeMs() + " ms");
        System.out.println("HELib:   " + result.getHelib().getTotalTimeMs() + " ms");
        System.out.println("OpenFHE: " + result.getOpenfhe().getTotalTimeMs() + " ms");
        System.out.println("\n🏆 Fastest: " + result.getFastestLibrary());
        System.out.println("💡 " + result.getRecommendation());
    }
}
