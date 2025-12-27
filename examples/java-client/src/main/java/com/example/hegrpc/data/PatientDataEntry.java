package com.example.hegrpc.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.example.hegrpc.manager.HospitalManager;
import com.example.hegrpc.model.Hospital;
import com.example.hegrpc.service.HEClientService;

import he_service.HeService.EncryptResponse;

/**
 * Handles patient data entry with encryption.
 * Allows users to input department statistics and encrypts them
 * using homomorphic encryption before storage.
 */
public class PatientDataEntry {
    
    private final Scanner scanner;
    private final HEClientService heClient;
    private final HospitalManager hospitalManager;
    
    // Predefined departments for healthcare demo
    public static final String[] DEPARTMENTS = {
        "Emergency Room (ER)",
        "Intensive Care Unit (ICU)",
        "Surgery",
        "Pediatrics",
        "Cardiology",
        "Oncology",
        "Neurology",
        "Orthopedics"
    };
    
    // Data fields for each department
    public static final String[] DATA_FIELDS = {
        "Current Patients",
        "Admitted Today",
        "Discharged Today",
        "Available Beds"
    };
    
    public PatientDataEntry(Scanner scanner, HEClientService heClient, HospitalManager hospitalManager) {
        this.scanner = scanner;
        this.heClient = heClient;
        this.hospitalManager = hospitalManager;
    }
    
    /**
     * Show the patient data entry submenu
     */
    public void showMenu() {
        boolean inSubmenu = true;
        
        while (inSubmenu) {
            System.out.println();
            System.out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
            System.out.println("│                           PATIENT DATA ENTRY                                 │");
            System.out.println("├──────────────────────────────────────────────────────────────────────────────┤");
            System.out.println("│   [1] Add Department Data (Manual Entry)                                     │");
            System.out.println("│   [2] Add Department Data (Random Demo Data)                                 │");
            System.out.println("│   [3] View Hospital's Encrypted Data                                         │");
            System.out.println("│   [4] View Encryption Visualization                                          │");
            System.out.println("│   [0] Back to Main Menu                                                      │");
            System.out.println("└──────────────────────────────────────────────────────────────────────────────┘");
            System.out.print("   Enter your choice: ");
            
            int choice = readIntChoice(0, 4);
            
            switch (choice) {
                case 1 -> addDepartmentDataManual();
                case 2 -> addDepartmentDataRandom();
                case 3 -> viewHospitalEncryptedData();
                case 4 -> viewEncryptionVisualization();
                case 0 -> inSubmenu = false;
            }
        }
    }
    
    /**
     * Add department data with manual entry
     */
    private void addDepartmentDataManual() {
        // Check if hospitals exist
        if (hospitalManager.getHospitalCount() == 0) {
            System.out.println("\n   📭 No hospitals registered yet!");
            System.out.println("   Please create a hospital first in Hospital Management.");
            pressEnterToContinue();
            return;
        }
        
        // Select hospital
        Hospital hospital = selectHospital();
        if (hospital == null) return;
        
        // Select department
        String department = selectDepartment();
        if (department == null) return;
        
        // Check if department already has data
        if (hospital.hasDepartment(department)) {
            System.out.println("\n   ⚠️  This department already has data. Overwrite? (y/n): ");
            String confirm = scanner.nextLine().trim().toLowerCase();
            if (!confirm.equals("y") && !confirm.equals("yes")) {
                System.out.println("   Cancelled.");
                return;
            }
        }
        
        // Enter data for each field
        System.out.println("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.printf("      Enter data for %s - %s%n", hospital.getName(), department);
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        long[] values = new long[DATA_FIELDS.length];
        for (int i = 0; i < DATA_FIELDS.length; i++) {
            System.out.printf("   %s: ", DATA_FIELDS[i]);
            values[i] = readLongValue(0, 10000);
        }
        
        // Encrypt and store
        encryptAndStore(hospital, department, values);
    }
    
    /**
     * Add department data with random values for demo
     */
    private void addDepartmentDataRandom() {
        // Check if hospitals exist
        if (hospitalManager.getHospitalCount() == 0) {
            System.out.println("\n      No hospitals registered yet!");
            System.out.println("   Please create a hospital first in Hospital Management.");
            pressEnterToContinue();
            return;
        }
        
        // Select hospital
        Hospital hospital = selectHospital();
        if (hospital == null) return;
        
        // Select department
        String department = selectDepartment();
        if (department == null) return;
        
        // Generate random demo data
        java.util.Random random = new java.util.Random();
        long[] values = new long[DATA_FIELDS.length];
        
        // Realistic ranges for each field
        values[0] = 20 + random.nextInt(80);   // Current Patients: 20-100
        values[1] = 5 + random.nextInt(20);    // Admitted Today: 5-25
        values[2] = 3 + random.nextInt(15);    // Discharged Today: 3-18
        values[3] = 10 + random.nextInt(40);   // Available Beds: 10-50
        
        System.out.println("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.printf("     Generated random data for %s%n", department);
        System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (int i = 0; i < DATA_FIELDS.length; i++) {
            System.out.printf("   %s: %d%n", DATA_FIELDS[i], values[i]);
        }
        
        // Encrypt and store
        encryptAndStore(hospital, department, values);
    }
    
    /**
     * Encrypt values and store in hospital
     */
    private void encryptAndStore(Hospital hospital, String department, long[] values) {
        System.out.println("\n      Encrypting data using SEAL...");
        
        try {
            // Convert to List<Long> for gRPC
            List<Long> valueList = new ArrayList<>();
            for (long v : values) {
                valueList.add(v);
            }
            
            // Encrypt using the hospital's session
            String sessionId = hospital.getSessionId();
            if (sessionId == null) {
                System.out.println("      Hospital session not initialized. Please recreate the hospital.");
                return;
            }
            
            EncryptResponse response = heClient.encrypt(sessionId, valueList);
            byte[] ciphertext = response.getCiphertext().toByteArray();
            
            // Store in hospital
            hospital.addDepartmentData(department, ciphertext, values);
            
            System.out.println();
            System.out.println("     Data encrypted and stored successfully!");
            System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.printf("      Hospital: %s (%s)%n", hospital.getName(), hospital.getId());
            System.out.printf("      Department: %s%n", department);
            System.out.printf("      Original values: [%d, %d, %d, %d]%n", values[0], values[1], values[2], values[3]);
            System.out.printf("      Encrypted size: %,d bytes%n", ciphertext.length);
            System.out.println("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            // Show a preview of encrypted bytes
            System.out.println("\n      Encrypted data preview (first 64 bytes in hex):");
            System.out.print("   ");
            for (int i = 0; i < Math.min(64, ciphertext.length); i++) {
                System.out.printf("%02X ", ciphertext[i] & 0xFF);
                if ((i + 1) % 16 == 0) System.out.print("\n   ");
            }
            System.out.println("...");
            
        } catch (Exception e) {
            System.out.println("      Encryption failed: " + e.getMessage());
        }
        
        pressEnterToContinue();
    }
    
    /**
     * View all encrypted data for a hospital
     */
    private void viewHospitalEncryptedData() {
        if (hospitalManager.getHospitalCount() == 0) {
            System.out.println("\n      No hospitals registered yet!");
            pressEnterToContinue();
            return;
        }
        
        Hospital hospital = selectHospital();
        if (hospital == null) return;
        
        if (hospital.getEncryptedDepartmentData().isEmpty()) {
            System.out.println("\n      This hospital has no encrypted data yet.");
            System.out.println("   Use 'Add Department Data' to add some!");
            pressEnterToContinue();
            return;
        }
        
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.printf("║     %s (%s)%n", hospital.getName(), hospital.getId());
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        for (String dept : hospital.getEncryptedDepartmentData().keySet()) {
            byte[] encrypted = hospital.getEncryptedDepartmentData().get(dept);
            long[] original = hospital.getOriginalData().get(dept);
            
            System.out.printf("║     %s%n", dept);
            System.out.println("║  ────────────────────────────────────────────────────────────────────────────");
            
            // Show original data (for demo verification)
            System.out.print("║       Original Data: ");
            for (int i = 0; i < DATA_FIELDS.length && i < original.length; i++) {
                System.out.printf("%s=%d  ", DATA_FIELDS[i].split(" ")[0], original[i]);
            }
            System.out.println();
            
            // Show encrypted size
            System.out.printf("║       Encrypted: %,d bytes%n", encrypted.length);
            System.out.println("║");
        }
        
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Show detailed encryption visualization
     */
    private void viewEncryptionVisualization() {
        if (hospitalManager.getHospitalCount() == 0) {
            System.out.println("\n      No hospitals registered yet!");
            pressEnterToContinue();
            return;
        }
        
        Hospital hospital = selectHospital();
        if (hospital == null) return;
        
        if (hospital.getEncryptedDepartmentData().isEmpty()) {
            System.out.println("\n      This hospital has no encrypted data yet.");
            pressEnterToContinue();
            return;
        }
        
        // Select department
        System.out.println("\n   Select department to visualize:");
        List<String> depts = new ArrayList<>(hospital.getEncryptedDepartmentData().keySet());
        for (int i = 0; i < depts.size(); i++) {
            System.out.printf("   [%d] %s%n", i + 1, depts.get(i));
        }
        System.out.print("   Enter choice: ");
        
        int choice = readIntChoice(1, depts.size());
        String dept = depts.get(choice - 1);
        
        byte[] encrypted = hospital.getEncryptedDepartmentData().get(dept);
        long[] original = hospital.getOriginalData().get(dept);
        
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         ENCRYPTION VISUALIZATION                             ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Hospital: %-66s║%n", hospital.getName());
        System.out.printf("║  Department: %-64s║%n", dept);
        System.out.println("╠══════════════════════════════════════════════════════════════════════════════╣");
        
        // Original data
        System.out.println("║                                                                              ║");
        System.out.println("║     ORIGINAL DATA (What the hospital knows):                                 ║");
        System.out.println("║  ┌─────────────────────────────────────────────────────────────────────────┐ ║");
        for (int i = 0; i < DATA_FIELDS.length && i < original.length; i++) {
            System.out.printf("║  │  %-25s : %-10d                            │ ║%n", 
                DATA_FIELDS[i], original[i]);
        }
        System.out.println("║  └─────────────────────────────────────────────────────────────────────────┘ ║");
        
        // Arrow
        System.out.println("║                                  ▼                                           ║");
        System.out.println("║                            SEAL Encryption                                   ║");
        System.out.println("║                                  ▼                                           ║");
        
        // Encrypted data
        System.out.println("║                                                                              ║");
        System.out.println("║     ENCRYPTED DATA (What others see - completely random-looking):            ║");
        System.out.println("║  ┌─────────────────────────────────────────────────────────────────────────┐ ║");
        
        // Show hex dump
        int bytesToShow = Math.min(128, encrypted.length);
        StringBuilder hexLine = new StringBuilder();
        for (int i = 0; i < bytesToShow; i++) {
            hexLine.append(String.format("%02X ", encrypted[i] & 0xFF));
            if ((i + 1) % 24 == 0) {
                System.out.printf("║  │  %s│ ║%n", hexLine.toString());
                hexLine = new StringBuilder();
            }
        }
        if (hexLine.length() > 0) {
            System.out.printf("║  │  %-71s│ ║%n", hexLine.toString());
        }
        System.out.printf("║  │  ... and %,d more bytes                                            │ ║%n", 
            encrypted.length - bytesToShow);
        System.out.println("║  └─────────────────────────────────────────────────────────────────────────┘ ║");
        
        // Stats
        System.out.println("║                                                                              ║");
        System.out.println("║     ENCRYPTION STATISTICS:                                                   ║");
        System.out.printf("║     • Original size: %d values × 8 bytes = %d bytes                          ║%n",
            original.length, original.length * 8);
        System.out.printf("║     • Encrypted size: %,d bytes                                           ║%n",
            encrypted.length);
        System.out.printf("║     • Expansion ratio: %.1fx                                                 ║%n",
            (double) encrypted.length / (original.length * 8));
        System.out.println("║                                                                              ║");
        System.out.println("║      The encrypted data can be shared safely - no one can read the values!   ║");
        System.out.println("║      But we can still COMPUTE on it using homomorphic operations!            ║");
        System.out.println("║                                                                              ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        pressEnterToContinue();
    }
    
    /**
     * Helper: Select a hospital from the list
     */
    private Hospital selectHospital() {
        hospitalManager.displayAllHospitals();
        System.out.print("   Enter Hospital ID: ");
        String id = scanner.nextLine().trim().toUpperCase();
        
        Hospital hospital = hospitalManager.getHospital(id);
        if (hospital == null) {
            System.out.println("      Hospital not found: " + id);
            return null;
        }
        return hospital;
    }
    
    /**
     * Helper: Select a department
     */
    private String selectDepartment() {
        System.out.println("\n   Select department:");
        for (int i = 0; i < DEPARTMENTS.length; i++) {
            System.out.printf("   [%d] %s%n", i + 1, DEPARTMENTS[i]);
        }
        System.out.print("   Enter department number: ");
        
        int choice = readIntChoice(1, DEPARTMENTS.length);
        return DEPARTMENTS[choice - 1];
    }
    
    /**
     * Read an integer choice within range
     */
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
    
    /**
     * Read a long value within range
     */
    private long readLongValue(long min, long max) {
        while (true) {
            try {
                String input = scanner.nextLine().trim();
                long value = Long.parseLong(input);
                if (value >= min && value <= max) {
                    return value;
                }
                System.out.printf("      Please enter a value between %d and %d: ", min, max);
            } catch (NumberFormatException e) {
                System.out.print("      Invalid input. Please enter a number: ");
            }
        }
    }
    
    private void pressEnterToContinue() {
        System.out.print("\n   Press Enter to continue...");
        scanner.nextLine();
    }
}
