package com.example.hegrpc.menu;

import java.util.Scanner;

import com.example.hegrpc.data.PatientDataEntry;
import com.example.hegrpc.manager.HospitalManager;
import com.example.hegrpc.model.Hospital;
import com.example.hegrpc.service.HEClientService;

/**
 * Interactive menu system for the Hospital Privacy Demo.
 * Provides a user-friendly interface for demonstrating
 * homomorphic encryption in healthcare scenarios.
 */
public class MenuSystem {
    
    private final Scanner scanner;
    private final HEClientService heClient;
    private final HospitalManager hospitalManager;
    private final PatientDataEntry patientDataEntry;
    private boolean running;
    
    public MenuSystem(HEClientService heClient) {
        this.scanner = new Scanner(System.in);
        this.heClient = heClient;
        this.hospitalManager = new HospitalManager(heClient);
        this.patientDataEntry = new PatientDataEntry(scanner, heClient, hospitalManager);
        this.running = true;
    }
    
    /**
     * Start the interactive menu
     */
    public void start() {
        printWelcomeBanner();
        
        while (running) {
            printMainMenu();
            int choice = readIntChoice(0, 7);
            handleMainMenuChoice(choice);
        }
        
        printGoodbye();
    }
    
    /**
     * Print welcome banner
     */
    private void printWelcomeBanner() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                                                                              ║");
        System.out.println("║        PRIVACY-PRESERVING HEALTHCARE DATA MANAGEMENT SYSTEM                  ║");
        System.out.println("║                                                                              ║");
        System.out.println("║   Powered by Homomorphic Encryption (SEAL/HELib/OpenFHE)                     ║");
        System.out.println("║                                                                              ║");
        System.out.println("║   FYP Project: Medical Data Privacy Using Homomorphic Encryption             ║");
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
    
    /**
     * Print main menu
     */
    private void printMainMenu() {
        System.out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
        System.out.println("│                                 MAIN MENU                                    │");
        System.out.println("├──────────────────────────────────────────────────────────────────────────────┤");
        System.out.println("│                                                                              │");
        System.out.println("│   [1]    Hospital Management     - Create, view, manage hospitals            │");
        System.out.println("│   [2]    Patient Data Entry      - Add encrypted department statistics       │");
        System.out.println("│   [3]    Regional Analytics      - Compute on encrypted data across regions  │");
        System.out.println("│   [4]    Save/Load Data          - Persist encrypted data to files           │");
        System.out.println("│   [5]    Security Demo           - Visualize encryption, simulate attacks    │");
        System.out.println("│   [6]    Benchmark Libraries      - Compare SEAL, HELib, OpenFHE             │");
        System.out.println("│   [7]    Help & About            - Learn about homomorphic encryption        │");
        System.out.println("│                                                                              │");
        System.out.println("│   [0]    Exit                                                                │");
        System.out.println("│                                                                              │");
        System.out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        System.out.print("   Enter your choice: ");
    }
    
    /**
     * Handle main menu choice
     */
    private void handleMainMenuChoice(int choice) {
        switch (choice) {
            case 1 -> hospitalManagementMenu();
            case 2 -> patientDataEntryMenu();
            case 3 -> regionalAnalyticsMenu();
            case 4 -> saveLoadDataMenu();
            case 5 -> securityDemoMenu();
            case 6 -> benchmarkMenu();
            case 7 -> helpMenu();
            case 0 -> running = false;
            default -> System.out.println("      Invalid choice. Please try again.");
        }
    }
    
    // ==================== OPTION 1: Hospital Management ====================
    
    private void hospitalManagementMenu() {
        boolean inSubmenu = true;
        
        while (inSubmenu) {
            System.out.println();
            System.out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
            System.out.println("│                           HOSPITAL MANAGEMENT                                │");
            System.out.println("├──────────────────────────────────────────────────────────────────────────────┤");
            System.out.println("│   [1] Create New Hospital                                                    │");
            System.out.println("│   [2] View All Hospitals                                                     │");
            System.out.println("│   [3] View Hospital Details                                                  │");
            System.out.println("│   [4] View Region Summary                                                    │");
            System.out.println("│   [5] Create Demo Hospitals (Quick Setup)                                    │");
            System.out.println("│   [0] Back to Main Menu                                                      │");
            System.out.println("└──────────────────────────────────────────────────────────────────────────────┘");
            System.out.print("   Enter your choice: ");
            
            int choice = readIntChoice(0, 5);
            
            switch (choice) {
                case 1 -> createNewHospital();
                case 2 -> hospitalManager.displayAllHospitals();
                case 3 -> viewHospitalDetails();
                case 4 -> hospitalManager.displayRegionSummary();
                case 5 -> createDemoHospitals();
                case 0 -> inSubmenu = false;
            }
        }
    }
    
    private void createNewHospital() {
        System.out.println();
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("      CREATE NEW HOSPITAL");
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Get hospital name
        System.out.print("   Enter hospital name: ");
        String name = scanner.nextLine().trim();
        if (name.isEmpty()) {
            System.out.println("      Hospital name cannot be empty.");
            return;
        }
        
        // Select region
        System.out.println("\n   Select region:");
        for (int i = 0; i < HospitalManager.REGIONS.length; i++) {
            System.out.printf("   [%d] %s%n", i + 1, HospitalManager.REGIONS[i]);
        }
        System.out.print("   Enter region number: ");
        
        int regionChoice = readIntChoice(1, HospitalManager.REGIONS.length);
        String region = HospitalManager.REGIONS[regionChoice - 1];
        
        // Create hospital
        System.out.println("\n      Creating hospital and generating encryption keys...");
        Hospital hospital = hospitalManager.createHospital(name, region);
        
        System.out.println();
        System.out.println("      Hospital created successfully!");
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.printf("      Hospital ID: %s%n", hospital.getId());
        System.out.printf("      Name: %s%n", hospital.getName());
        System.out.printf("      Region: %s%n", hospital.getRegion());
        System.out.printf("      Session ID: %s%n", hospital.getSessionId());
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println();
        
        pressEnterToContinue();
    }
    
    private void viewHospitalDetails() {
        if (hospitalManager.getHospitalCount() == 0) {
            System.out.println("\n      No hospitals registered yet. Create one first!");
            return;
        }
        
        hospitalManager.displayAllHospitals();
        System.out.print("   Enter Hospital ID to view details: ");
        String id = scanner.nextLine().trim().toUpperCase();
        
        Hospital hospital = hospitalManager.getHospital(id);
        if (hospital == null) {
            System.out.println("      Hospital not found: " + id);
            return;
        }
        
        System.out.println();
        System.out.println(hospital.getDetailedInfo());
        System.out.println();
        
        pressEnterToContinue();
    }
    
    private void createDemoHospitals() {
        System.out.println("\n      Creating demo hospitals with encryption keys...\n");
        hospitalManager.createDemoHospitals();
        System.out.println("      Created 5 demo hospitals!\n");
        hospitalManager.displayAllHospitals();
    }
    
    // ==================== OPTION 2: Patient Data Entry ====================
    
    private void patientDataEntryMenu() {
        patientDataEntry.showMenu();
    }
    
    // ==================== OPTION 3: Regional Analytics (Placeholder) ====================
    
    private void regionalAnalyticsMenu() {
        System.out.println();
        System.out.println("      COMING IN PHASE 3: Regional Analytics");
        System.out.println("   - Sum encrypted data across hospitals");
        System.out.println("   - Compute regional statistics");
        System.out.println("   - Privacy-preserving aggregation");
        System.out.println();
        pressEnterToContinue();
    }
    
    // ==================== OPTION 4: Save/Load Data (Placeholder) ====================
    
    private void saveLoadDataMenu() {
        System.out.println();
        System.out.println("      COMING IN PHASE 4: Data Persistence");
        System.out.println("   - Save encrypted data to JSON files");
        System.out.println("   - Load previous sessions");
        System.out.println("   - Export/import hospital data");
        System.out.println();
        pressEnterToContinue();
    }
    
    // ==================== OPTION 5: Security Demo (Placeholder) ====================
    
    private void securityDemoMenu() {
        System.out.println();
        System.out.println("      COMING IN PHASE 5: Security Demo");
        System.out.println("   - Visualize encrypted data (hex dump)");
        System.out.println("   - Show encryption randomness");
        System.out.println("   - Simulate interception attacks");
        System.out.println();
        pressEnterToContinue();
    }
    
    // ==================== OPTION 6: Benchmark (Placeholder) ====================
    
    private void benchmarkMenu() {
        System.out.println();
        System.out.println("      COMING IN PHASE 6: Benchmark");
        System.out.println("   - Compare SEAL, HELib, OpenFHE");
        System.out.println("   - Performance metrics");
        System.out.println("   - Visual charts");
        System.out.println();
        pressEnterToContinue();
    }
    
    // ==================== OPTION 7: Help Menu ====================
    
    private void helpMenu() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                             ABOUT THIS SYSTEM                                ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.println("║                                                                              ║");
        System.out.println("║     WHAT IS HOMOMORPHIC ENCRYPTION?                                          ║");
        System.out.println("║     Homomorphic encryption allows computations on encrypted data             ║");
        System.out.println("║     without decrypting it first. This means:                                 ║");
        System.out.println("║                                                                              ║");
        System.out.println("║     • Hospitals can share encrypted patient statistics                       ║");
        System.out.println("║     • Regional health authorities can compute totals                         ║");
        System.out.println("║     • Nobody sees the actual numbers except the data owner                   ║");
        System.out.println("║                                                                              ║");
        System.out.println("║     HEALTHCARE USE CASE                                                      ║");
        System.out.println("║     In this demo, multiple hospitals can:                                    ║");
        System.out.println("║     1. Encrypt their patient counts (ER, ICU, Surgery, etc.)                 ║");
        System.out.println("║     2. Share the encrypted data with health authorities                      ║");
        System.out.println("║     3. Authorities sum the encrypted data to get regional totals             ║");
        System.out.println("║     4. No hospital reveals its actual patient counts!                        ║");
        System.out.println("║                                                                              ║");
        System.out.println("║     LIBRARIES USED                                                           ║");
        System.out.println("║     • Microsoft SEAL - Fast BFV/CKKS schemes                                 ║");
        System.out.println("║     • HELib - IBM's BGV scheme implementation                                ║");
        System.out.println("║     • OpenFHE - Flexible multi-scheme library                                ║");
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        pressEnterToContinue();
    }
    
    // ==================== Utility Methods ====================
    
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
                System.out.printf("      Invalid input. Please enter a number: ");
            }
        }
    }
    
    private void pressEnterToContinue() {
        System.out.print("   Press Enter to continue...");
        scanner.nextLine();
    }
    
    private void printGoodbye() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                                                                              ║");
        System.out.println("║       Thank you for using the Privacy-Preserving Healthcare System!          ║");
        System.out.println("║                                                                              ║");
        System.out.println("║       Remember: With homomorphic encryption, data stays private even         ║");
        System.out.println("║       during computation. The future of healthcare data privacy!             ║");
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
    
    // Getter for hospital manager (for other components)
    public HospitalManager getHospitalManager() {
        return hospitalManager;
    }
}
