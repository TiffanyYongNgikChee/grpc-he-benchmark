package com.example.hegrpc.benchmark;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.example.hegrpc.service.HEClientService;

import he_service.HeService.BenchmarkResponse;
import he_service.HeService.ComparisonBenchmarkResponse;

/**
 * Benchmark Demo - Compare HE Library Performance
 * 
 * Compares three homomorphic encryption libraries:
 * - Microsoft SEAL
 * - IBM HELib  
 * - OpenFHE
 * 
 * Measures key generation, encryption, operations, and decryption times.
 */
public class BenchmarkDemo {
    
    private final Scanner scanner;
    private final HEClientService heClient;
    
    // Store results for comparison
    private final List<BenchmarkResult> benchmarkHistory;
    
    public BenchmarkDemo(Scanner scanner, HEClientService heClient) {
        this.scanner = scanner;
        this.heClient = heClient;
        this.benchmarkHistory = new ArrayList<>();
    }
    
    /**
     * Show the benchmark submenu
     */
    public void showMenu() {
        boolean inSubmenu = true;
        
        while (inSubmenu) {
            System.out.println();
            System.out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
            System.out.println("│                           BENCHMARK LIBRARIES                                │");
            System.out.println("│               Compare SEAL, HELib, and OpenFHE Performance                   │");
            System.out.println("├──────────────────────────────────────────────────────────────────────────────┤");
            System.out.println("│   [1] Quick Comparison (All Libraries)                                       │");
            System.out.println("│   [2] Detailed SEAL Benchmark                                                │");
            System.out.println("│   [3] Detailed HELib Benchmark                                               │");
            System.out.println("│   [4] Detailed OpenFHE Benchmark                                             │");
            System.out.println("│   [5] Custom Benchmark (Configure Operations)                                │");
            System.out.println("│   [6] View Benchmark History                                                 │");
            System.out.println("│   [7] Performance Visualization                                              │");
            System.out.println("│   [0] Back to Main Menu                                                      │");
            System.out.println("└──────────────────────────────────────────────────────────────────────────────┘");
            System.out.print("   Enter your choice: ");
            
            int choice = readIntChoice(0, 7);
            
            switch (choice) {
                case 1 -> quickComparison();
                case 2 -> detailedBenchmark("SEAL");
                case 3 -> detailedBenchmark("HELib");
                case 4 -> detailedBenchmark("OpenFHE");
                case 5 -> customBenchmark();
                case 6 -> viewBenchmarkHistory();
                case 7 -> performanceVisualization();
                case 0 -> inSubmenu = false;
            }
        }
    }
    
    /**
     * Option 1: Quick comparison of all libraries
     */
    private void quickComparison() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                     QUICK LIBRARY COMPARISON                                 ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.println("║                                                                              ║");
        System.out.println("║  Running benchmark with 10 operations per library...                        ║");
        System.out.println("║                                                                              ║");
        System.out.println("║     [SEAL]     Key Gen → Encrypt → Add → Multiply → Decrypt                 ║");
        System.out.println("║     [HELib]    Key Gen → Encrypt → Add → Multiply → Decrypt                 ║");
        System.out.println("║     [OpenFHE]  Key Gen → Encrypt → Add → Multiply → Decrypt                 ║");
        System.out.println("║                                                                              ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        try {
            System.out.println("║  Starting benchmark... (this may take 30-60 seconds)                        ║");
            System.out.println("║                                                                              ║");
            
            ComparisonBenchmarkResponse response = heClient.runComparisonBenchmark(10);
            
            // Store result
            benchmarkHistory.add(new BenchmarkResult("Comparison", 10, response));
            
            // Display results table
            displayComparisonTable(response);
            
            // Show winner and recommendation
            System.out.println("║                                                                              ║");
            System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
            System.out.printf("║      FASTEST LIBRARY: %-54s║%n", response.getFastestLibrary());
            System.out.println("║                                                                              ║");
            String rec = response.getRecommendation();
            if (rec.length() > 70) {
                System.out.printf("║  %s ║%n", rec.substring(0, 70));
                if (rec.length() > 140) {
                    System.out.printf("║  %s ║%n", rec.substring(70, 140));
                } else if (rec.length() > 70) {
                    System.out.printf("║  %-74s║%n", rec.substring(70));
                }
            } else {
                System.out.printf("║  %-74s║%n", rec);
            }
            
        } catch (Exception e) {
            System.out.println("║                                                                              ║");
            System.out.println("║      Error running benchmark: " + truncate(e.getMessage(), 43) + " ║");
            System.out.println("║                                                                              ║");
            System.out.println("║   Make sure the gRPC server is running with all three libraries enabled.   ║");
        }
        
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Option 2-4: Detailed benchmark for a specific library
     */
    private void detailedBenchmark(String library) {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.printf("║                      DETAILED %s BENCHMARK                              ║%n", 
            padCenter(library, 8));
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        printLibraryInfo(library);
        
        System.out.println("║                                                                              ║");
        System.out.print("║  Enter number of operations (1-100): ");
        
        int numOps = readIntChoice(1, 100);
        System.out.println("║                                                                              ║");
        System.out.printf("║  Running %d operations...                                                    ║%n", numOps);
        System.out.println("║                                                                              ║");
        
        try {
            long startTime = System.currentTimeMillis();
            BenchmarkResponse response = heClient.runBenchmark(library, numOps);
            long totalClientTime = System.currentTimeMillis() - startTime;
            
            displayDetailedResults(library, response, numOps, totalClientTime);
            
        } catch (Exception e) {
            System.out.println("║                                                                              ║");
            System.out.println("║      Error: " + truncate(e.getMessage(), 60) + " ║");
            System.out.println("║                                                                              ║");
            System.out.println("║   The " + library + " library may not be available on the server.           ║");
        }
        
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Option 5: Custom benchmark with user-defined parameters
     */
    private void customBenchmark() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                          CUSTOM BENCHMARK                                    ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.println("║                                                                              ║");
        System.out.println("║  Configure your benchmark parameters:                                        ║");
        System.out.println("║                                                                              ║");
        System.out.println("║  Select libraries to benchmark:                                              ║");
        System.out.println("║    [1] SEAL only                                                             ║");
        System.out.println("║    [2] HELib only                                                            ║");
        System.out.println("║    [3] OpenFHE only                                                          ║");
        System.out.println("║    [4] All three libraries                                                   ║");
        System.out.print("║  Choice: ");
        
        int libChoice = readIntChoice(1, 4);
        
        System.out.println("║                                                                              ║");
        System.out.print("║  Enter number of operations (1-1000): ");
        int numOps = readIntChoice(1, 1000);
        
        System.out.println("║                                                                              ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Running custom benchmark with %d operations...                             ║%n", numOps);
        System.out.println("║                                                                              ║");
        
        try {
            if (libChoice == 4) {
                // Compare all
                ComparisonBenchmarkResponse response = heClient.runComparisonBenchmark(numOps);
                benchmarkHistory.add(new BenchmarkResult("Custom-All", numOps, response));
                displayComparisonTable(response);
                
                System.out.println("║                                                                              ║");
                System.out.printf("║      FASTEST: %-60s║%n", response.getFastestLibrary());
            } else {
                // Single library
                String library = switch (libChoice) {
                    case 1 -> "SEAL";
                    case 2 -> "HELib";
                    case 3 -> "OpenFHE";
                    default -> "SEAL";
                };
                
                long startTime = System.currentTimeMillis();
                BenchmarkResponse response = heClient.runBenchmark(library, numOps);
                long totalClientTime = System.currentTimeMillis() - startTime;
                
                displayDetailedResults(library, response, numOps, totalClientTime);
            }
            
        } catch (Exception e) {
            System.out.println("║      Error: " + truncate(e.getMessage(), 60) + " ║");
        }
        
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Option 6: View benchmark history
     */
    private void viewBenchmarkHistory() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                          BENCHMARK HISTORY                                   ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        if (benchmarkHistory.isEmpty()) {
            System.out.println("║                                                                              ║");
            System.out.println("║     No benchmarks have been run yet.                                        ║");
            System.out.println("║     Run a benchmark first to see results here.                              ║");
            System.out.println("║                                                                              ║");
        } else {
            System.out.println("║                                                                              ║");
            System.out.println("║  #   TYPE            OPERATIONS    FASTEST       TOTAL TIME                 ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            
            int idx = 1;
            for (BenchmarkResult result : benchmarkHistory) {
                String fastest = result.response != null ? result.response.getFastestLibrary() : "N/A";
                double totalTime = 0;
                if (result.response != null) {
                    totalTime = result.response.getSeal().getTotalTimeMs() 
                              + result.response.getHelib().getTotalTimeMs()
                              + result.response.getOpenfhe().getTotalTimeMs();
                }
                
                System.out.printf("║  %-3d %-15s %-13d %-13s %.0f ms              ║%n",
                    idx++, result.type, result.numOperations, fastest, totalTime);
            }
            System.out.println("║                                                                              ║");
        }
        
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Option 7: Visual performance comparison
     */
    private void performanceVisualization() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                      PERFORMANCE VISUALIZATION                               ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.println("║                                                                              ║");
        System.out.println("║  Running quick benchmark for visualization...                                ║");
        System.out.println("║                                                                              ║");
        
        try {
            ComparisonBenchmarkResponse response = heClient.runComparisonBenchmark(5);
            
            BenchmarkResponse seal = response.getSeal();
            BenchmarkResponse helib = response.getHelib();
            BenchmarkResponse openfhe = response.getOpenfhe();
            
            System.out.println("║  KEY GENERATION TIME (ms)                                                    ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            printBarChart("SEAL", seal.getKeyGenTimeMs(), getMax(seal.getKeyGenTimeMs(), helib.getKeyGenTimeMs(), openfhe.getKeyGenTimeMs()));
            printBarChart("HELib", helib.getKeyGenTimeMs(), getMax(seal.getKeyGenTimeMs(), helib.getKeyGenTimeMs(), openfhe.getKeyGenTimeMs()));
            printBarChart("OpenFHE", openfhe.getKeyGenTimeMs(), getMax(seal.getKeyGenTimeMs(), helib.getKeyGenTimeMs(), openfhe.getKeyGenTimeMs()));
            
            System.out.println("║                                                                              ║");
            System.out.println("║  ENCRYPTION TIME (ms)                                                        ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            printBarChart("SEAL", seal.getEncryptionTimeMs(), getMax(seal.getEncryptionTimeMs(), helib.getEncryptionTimeMs(), openfhe.getEncryptionTimeMs()));
            printBarChart("HELib", helib.getEncryptionTimeMs(), getMax(seal.getEncryptionTimeMs(), helib.getEncryptionTimeMs(), openfhe.getEncryptionTimeMs()));
            printBarChart("OpenFHE", openfhe.getEncryptionTimeMs(), getMax(seal.getEncryptionTimeMs(), helib.getEncryptionTimeMs(), openfhe.getEncryptionTimeMs()));
            
            System.out.println("║                                                                              ║");
            System.out.println("║  ADDITION TIME (ms)                                                          ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            printBarChart("SEAL", seal.getAdditionTimeMs(), getMax(seal.getAdditionTimeMs(), helib.getAdditionTimeMs(), openfhe.getAdditionTimeMs()));
            printBarChart("HELib", helib.getAdditionTimeMs(), getMax(seal.getAdditionTimeMs(), helib.getAdditionTimeMs(), openfhe.getAdditionTimeMs()));
            printBarChart("OpenFHE", openfhe.getAdditionTimeMs(), getMax(seal.getAdditionTimeMs(), helib.getAdditionTimeMs(), openfhe.getAdditionTimeMs()));
            
            System.out.println("║                                                                              ║");
            System.out.println("║  MULTIPLICATION TIME (ms)                                                    ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            printBarChart("SEAL", seal.getMultiplicationTimeMs(), getMax(seal.getMultiplicationTimeMs(), helib.getMultiplicationTimeMs(), openfhe.getMultiplicationTimeMs()));
            printBarChart("HELib", helib.getMultiplicationTimeMs(), getMax(seal.getMultiplicationTimeMs(), helib.getMultiplicationTimeMs(), openfhe.getMultiplicationTimeMs()));
            printBarChart("OpenFHE", openfhe.getMultiplicationTimeMs(), getMax(seal.getMultiplicationTimeMs(), helib.getMultiplicationTimeMs(), openfhe.getMultiplicationTimeMs()));
            
            System.out.println("║                                                                              ║");
            System.out.println("║  TOTAL TIME (ms)                                                             ║");
            System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
            printBarChart("SEAL", seal.getTotalTimeMs(), getMax(seal.getTotalTimeMs(), helib.getTotalTimeMs(), openfhe.getTotalTimeMs()));
            printBarChart("HELib", helib.getTotalTimeMs(), getMax(seal.getTotalTimeMs(), helib.getTotalTimeMs(), openfhe.getTotalTimeMs()));
            printBarChart("OpenFHE", openfhe.getTotalTimeMs(), getMax(seal.getTotalTimeMs(), helib.getTotalTimeMs(), openfhe.getTotalTimeMs()));
            
            System.out.println("║                                                                              ║");
            System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
            System.out.println("║  LEGEND:  Shorter bars = Better performance                                  ║");
            System.out.printf("║           Winner: %-56s║%n", response.getFastestLibrary());
            
        } catch (Exception e) {
            System.out.println("║      Error: " + truncate(e.getMessage(), 60) + " ║");
            System.out.println("║                                                                              ║");
            System.out.println("║   Make sure the gRPC server is running with benchmark support.              ║");
        }
        
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    // ==================== Display Helpers ====================
    
    private void displayComparisonTable(ComparisonBenchmarkResponse response) {
        BenchmarkResponse seal = response.getSeal();
        BenchmarkResponse helib = response.getHelib();
        BenchmarkResponse openfhe = response.getOpenfhe();
        
        System.out.println("║                                                                              ║");
        System.out.println("║  OPERATION           │    SEAL     │   HELib     │  OpenFHE    │  WINNER    ║");
        System.out.println("║  ────────────────────┼─────────────┼─────────────┼─────────────┼────────────║");
        
        // Key Generation
        String keyWinner = findWinner(seal.getKeyGenTimeMs(), helib.getKeyGenTimeMs(), openfhe.getKeyGenTimeMs());
        System.out.printf("║  Key Generation      │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getKeyGenTimeMs(), helib.getKeyGenTimeMs(), openfhe.getKeyGenTimeMs(), keyWinner);
        
        // Encryption
        String encWinner = findWinner(seal.getEncryptionTimeMs(), helib.getEncryptionTimeMs(), openfhe.getEncryptionTimeMs());
        System.out.printf("║  Encryption          │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getEncryptionTimeMs(), helib.getEncryptionTimeMs(), openfhe.getEncryptionTimeMs(), encWinner);
        
        // Addition
        String addWinner = findWinner(seal.getAdditionTimeMs(), helib.getAdditionTimeMs(), openfhe.getAdditionTimeMs());
        System.out.printf("║  Addition            │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getAdditionTimeMs(), helib.getAdditionTimeMs(), openfhe.getAdditionTimeMs(), addWinner);
        
        // Multiplication
        String mulWinner = findWinner(seal.getMultiplicationTimeMs(), helib.getMultiplicationTimeMs(), openfhe.getMultiplicationTimeMs());
        System.out.printf("║  Multiplication      │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getMultiplicationTimeMs(), helib.getMultiplicationTimeMs(), openfhe.getMultiplicationTimeMs(), mulWinner);
        
        // Decryption
        String decWinner = findWinner(seal.getDecryptionTimeMs(), helib.getDecryptionTimeMs(), openfhe.getDecryptionTimeMs());
        System.out.printf("║  Decryption          │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getDecryptionTimeMs(), helib.getDecryptionTimeMs(), openfhe.getDecryptionTimeMs(), decWinner);
        
        System.out.println("║  ────────────────────┼─────────────┼─────────────┼─────────────┼────────────║");
        
        // Total
        String totalWinner = findWinner(seal.getTotalTimeMs(), helib.getTotalTimeMs(), openfhe.getTotalTimeMs());
        System.out.printf("║  TOTAL               │ %9.2f ms│ %9.2f ms│ %9.2f ms│ %-10s ║%n",
            seal.getTotalTimeMs(), helib.getTotalTimeMs(), openfhe.getTotalTimeMs(), totalWinner);
    }
    
    private void displayDetailedResults(String library, BenchmarkResponse response, int numOps, long clientTime) {
        System.out.println("║  ┌─────────────────────────────────────────────────────────────────────────┐ ║");
        System.out.printf("║  │  Library: %-64s│ ║%n", library);
        System.out.printf("║  │  Operations: %-61d│ ║%n", numOps);
        System.out.printf("║  │  Status: %-65s│ ║%n", response.getStatus());
        System.out.println("║  └─────────────────────────────────────────────────────────────────────────┘ ║");
        System.out.println("║                                                                              ║");
        System.out.println("║  OPERATION                        TIME (ms)        PER-OP (ms)              ║");
        System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
        
        double keyGenPerOp = response.getKeyGenTimeMs() / numOps;
        double encPerOp = response.getEncryptionTimeMs() / numOps;
        double addPerOp = response.getAdditionTimeMs() / numOps;
        double mulPerOp = response.getMultiplicationTimeMs() / numOps;
        double decPerOp = response.getDecryptionTimeMs() / numOps;
        
        System.out.printf("║  Key Generation               %10.2f         %10.4f                ║%n", 
            response.getKeyGenTimeMs(), keyGenPerOp);
        System.out.printf("║  Encryption                   %10.2f         %10.4f                ║%n", 
            response.getEncryptionTimeMs(), encPerOp);
        System.out.printf("║  Addition                     %10.2f         %10.4f                ║%n", 
            response.getAdditionTimeMs(), addPerOp);
        System.out.printf("║  Multiplication               %10.2f         %10.4f                ║%n", 
            response.getMultiplicationTimeMs(), mulPerOp);
        System.out.printf("║  Decryption                   %10.2f         %10.4f                ║%n", 
            response.getDecryptionTimeMs(), decPerOp);
        
        if (response.getEncodingTimeMs() > 0) {
            double encodePerOp = response.getEncodingTimeMs() / numOps;
            System.out.printf("║  Encoding                     %10.2f         %10.4f                ║%n", 
                response.getEncodingTimeMs(), encodePerOp);
        }
        
        System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
        System.out.printf("║  TOTAL (Server)               %10.2f ms                                  ║%n", 
            response.getTotalTimeMs());
        System.out.printf("║  TOTAL (Client, incl. gRPC)   %10d ms                                  ║%n", 
            clientTime);
        System.out.println("║                                                                              ║");
        
        // Performance analysis
        System.out.println("║  ANALYSIS:                                                                   ║");
        double overhead = clientTime - response.getTotalTimeMs();
        double overheadPercent = (overhead / clientTime) * 100;
        System.out.printf("║    • Network/gRPC overhead: %.0f ms (%.1f%% of total)                       ║%n", 
            overhead, overheadPercent);
        
        double opsPerSec = (numOps * 1000.0) / response.getTotalTimeMs();
        System.out.printf("║    • Throughput: %.1f operations/second                                     ║%n", opsPerSec);
    }
    
    private void printLibraryInfo(String library) {
        switch (library) {
            case "SEAL" -> {
                System.out.println("║                                                                              ║");
                System.out.println("║     Microsoft SEAL                                                           ║");
                System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
                System.out.println("║  • Developed by Microsoft Research                                           ║");
                System.out.println("║  • Supports BFV and CKKS schemes                                             ║");
                System.out.println("║  • Excellent for integer arithmetic (BFV)                                    ║");
                System.out.println("║  • Good balance of speed and security                                        ║");
            }
            case "HELib" -> {
                System.out.println("║                                                                              ║");
                System.out.println("║     IBM HELib                                                                ║");
                System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
                System.out.println("║  • Developed by IBM Research                                                 ║");
                System.out.println("║  • Implements BGV scheme                                                     ║");
                System.out.println("║  • Strong SIMD support for vectorized operations                             ║");
                System.out.println("║  • Best for complex computations                                             ║");
            }
            case "OpenFHE" -> {
                System.out.println("║                                                                              ║");
                System.out.println("║     OpenFHE                                                                  ║");
                System.out.println("║  ─────────────────────────────────────────────────────────────────────────  ║");
                System.out.println("║  • Open-source, multi-institutional project                                  ║");
                System.out.println("║  • Supports BFV, BGV, CKKS, FHEW, and TFHE                                   ║");
                System.out.println("║  • Most flexible library                                                     ║");
                System.out.println("║  • Active development and community support                                  ║");
            }
        }
    }
    
    private void printBarChart(String label, double value, double maxValue) {
        int barLength = maxValue > 0 ? (int) ((value / maxValue) * 40) : 0;
        barLength = Math.max(1, Math.min(40, barLength));
        String bar = "█".repeat(barLength);
        System.out.printf("║  %-8s %8.2f ms  %-43s║%n", label, value, bar);
    }
    
    // ==================== Helper Methods ====================
    
    private String findWinner(double seal, double helib, double openfhe) {
        if (seal <= helib && seal <= openfhe) return "SEAL ★";
        if (helib <= seal && helib <= openfhe) return "HELib ★";
        return "OpenFHE ★";
    }
    
    private double getMax(double a, double b, double c) {
        return Math.max(a, Math.max(b, c));
    }
    
    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() <= maxLen ? str : str.substring(0, maxLen - 3) + "...";
    }
    
    private String padCenter(String str, int width) {
        if (str.length() >= width) return str;
        int padding = width - str.length();
        int leftPad = padding / 2;
        int rightPad = padding - leftPad;
        return " ".repeat(leftPad) + str + " ".repeat(rightPad);
    }
    
    private int readIntChoice(int min, int max) {
        while (true) {
            try {
                String input = scanner.nextLine().trim();
                int choice = Integer.parseInt(input);
                if (choice >= min && choice <= max) {
                    return choice;
                }
                System.out.printf("      Please enter a number between %d and %d: ", min, max);
            } catch (NumberFormatException e) {
                System.out.print("      Invalid input. Please enter a number: ");
            }
        }
    }
    
    private void pressEnterToContinue() {
        System.out.print("\n   Press Enter to continue...");
        scanner.nextLine();
    }
    
    // ==================== Inner Classes ====================
    
    /**
     * Store benchmark results for history
     */
    private static class BenchmarkResult {
        String type;
        int numOperations;
        ComparisonBenchmarkResponse response;
        
        BenchmarkResult(String type, int numOperations, ComparisonBenchmarkResponse response) {
            this.type = type;
            this.numOperations = numOperations;
            this.response = response;
        }
    }
}
