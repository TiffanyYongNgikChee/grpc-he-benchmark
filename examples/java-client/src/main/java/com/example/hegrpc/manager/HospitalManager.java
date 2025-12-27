package com.example.hegrpc.manager;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import com.example.hegrpc.model.Hospital;
import com.example.hegrpc.service.HEClientService;

import he_service.HeService.GenerateKeysResponse;

/**
 * Manages hospitals in the privacy-preserving healthcare system.
 * Handles hospital creation, listing, and lookup.
 */
public class HospitalManager {
    
    private final Map<String, Hospital> hospitals;
    private final AtomicInteger hospitalCounter;
    private final HEClientService heClient;
    
    // Predefined regions for demo
    public static final String[] REGIONS = {
        "North Region",
        "South Region", 
        "East Region",
        "West Region",
        "Central Region"
    };
    
    // Predefined hospital name prefixes for demo
    private static final String[] HOSPITAL_PREFIXES = {
        "General", "Memorial", "University", "Community", "Regional", 
        "Children's", "Veterans", "Sacred Heart", "Mercy", "Grace"
    };
    
    public HospitalManager(HEClientService heClient) {
        this.hospitals = new LinkedHashMap<>(); // Maintains insertion order
        this.hospitalCounter = new AtomicInteger(0);
        this.heClient = heClient;
    }
    
    /**
     * Generate a unique hospital ID
     * Format: HSP-XXXX (e.g., HSP-0001)
     */
    public String generateHospitalId() {
        int num = hospitalCounter.incrementAndGet();
        return String.format("HSP-%04d", num);
    }
    
    /**
     * Create a new hospital with auto-generated ID
     */
    public Hospital createHospital(String name, String region) {
        String id = generateHospitalId();
        Hospital hospital = new Hospital(id, name, region);
        
        // Generate encryption keys for this hospital
        try {
            GenerateKeysResponse keys = heClient.generateKeys("SEAL", 8192);
            hospital.setSessionId(keys.getSessionId());
        } catch (Exception e) {
            System.err.println("Warning: Could not generate keys for hospital: " + e.getMessage());
        }
        
        hospitals.put(id, hospital);
        return hospital;
    }
    
    /**
     * Create a hospital with a specific ID (for loading from file)
     */
    public Hospital createHospitalWithId(String id, String name, String region, String sessionId) {
        Hospital hospital = new Hospital(id, name, region);
        hospital.setSessionId(sessionId);
        hospitals.put(id, hospital);
        
        // Update counter if needed
        try {
            int num = Integer.parseInt(id.replace("HSP-", ""));
            if (num >= hospitalCounter.get()) {
                hospitalCounter.set(num);
            }
        } catch (NumberFormatException ignored) {}
        
        return hospital;
    }
    
    /**
     * Get a hospital by ID
     */
    public Hospital getHospital(String id) {
        return hospitals.get(id.toUpperCase());
    }
    
    /**
     * Check if a hospital exists
     */
    public boolean hospitalExists(String id) {
        return hospitals.containsKey(id.toUpperCase());
    }
    
    /**
     * Get all hospitals
     */
    public Collection<Hospital> getAllHospitals() {
        return hospitals.values();
    }
    
    /**
     * Get hospitals by region
     */
    public List<Hospital> getHospitalsByRegion(String region) {
        List<Hospital> result = new ArrayList<>();
        for (Hospital h : hospitals.values()) {
            if (h.getRegion().equalsIgnoreCase(region)) {
                result.add(h);
            }
        }
        return result;
    }
    
    /**
     * Get number of hospitals
     */
    public int getHospitalCount() {
        return hospitals.size();
    }
    
    /**
     * Clear all hospitals (for demo reset)
     */
    public void clearAll() {
        hospitals.clear();
        hospitalCounter.set(0);
    }
    
    /**
     * Generate a random hospital name for demo
     */
    public String generateRandomHospitalName() {
        Random random = new Random();
        String prefix = HOSPITAL_PREFIXES[random.nextInt(HOSPITAL_PREFIXES.length)];
        return prefix + " Hospital";
    }
    
    /**
     * Create demo hospitals for quick testing
     */
    public void createDemoHospitals() {
        createHospital("Singapore General Hospital", "Central Region");
        createHospital("Tan Tock Seng Hospital", "Central Region");
        createHospital("National University Hospital", "West Region");
        createHospital("Changi General Hospital", "East Region");
        createHospital("Khoo Teck Puat Hospital", "North Region");
    }
    
    /**
     * Display all hospitals in a formatted table
     */
    public void displayAllHospitals() {
        if (hospitals.isEmpty()) {
            System.out.println("\n  📭 No hospitals registered yet.\n");
            return;
        }
        
        System.out.println("\n╔════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         🏥 REGISTERED HOSPITALS                            ║");
        System.out.println("╠══════════╦══════════════════════════════════╦════════════════╦═════════════╣");
        System.out.println("║    ID    ║           Hospital Name          ║     Region     ║ Departments ║");
        System.out.println("╠══════════╬══════════════════════════════════╬════════════════╬═════════════╣");
        
        for (Hospital h : hospitals.values()) {
            String name = h.getName();
            if (name.length() > 32) {
                name = name.substring(0, 29) + "...";
            }
            String region = h.getRegion();
            if (region.length() > 14) {
                region = region.substring(0, 11) + "...";
            }
            
            System.out.printf("║ %-8s ║ %-32s ║ %-14s ║     %3d     ║%n",
                h.getId(), name, region, h.getEncryptedDepartmentData().size());
        }
        
        System.out.println("╚══════════╩══════════════════════════════════╩════════════════╩═════════════╝");
        System.out.printf("  Total: %d hospital(s)%n%n", hospitals.size());
    }
    
    /**
     * Display region summary
     */
    public void displayRegionSummary() {
        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║                    📊 REGION SUMMARY                        ║");
        System.out.println("╠════════════════════════════════════════════════════════════╣");
        
        for (String region : REGIONS) {
            List<Hospital> regionHospitals = getHospitalsByRegion(region);
            int totalDepts = regionHospitals.stream()
                .mapToInt(h -> h.getEncryptedDepartmentData().size())
                .sum();
            
            System.out.printf("║  %-20s: %2d hospital(s), %3d department(s)   ║%n",
                region, regionHospitals.size(), totalDepts);
        }
        
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");
    }
}
