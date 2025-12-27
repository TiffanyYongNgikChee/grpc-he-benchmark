package com.example.hegrpc.model;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents a hospital in the privacy-preserving healthcare system.
 * Each hospital has encrypted department data that can be aggregated
 * without revealing individual hospital statistics.
 */
public class Hospital {
    
    private final String id;
    private final String name;
    private final String region;
    private final LocalDateTime createdAt;
    
    // Department name -> encrypted ciphertext (as byte array)
    private final Map<String, byte[]> encryptedDepartmentData;
    
    // Department name -> original values (for demo verification only)
    private final Map<String, long[]> originalData;
    
    // Session ID for this hospital's encryption context
    private String sessionId;
    
    public Hospital(String id, String name, String region) {
        this.id = id;
        this.name = name;
        this.region = region;
        this.createdAt = LocalDateTime.now();
        this.encryptedDepartmentData = new HashMap<>();
        this.originalData = new HashMap<>();
    }
    
    // Getters
    public String getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
    
    public String getRegion() {
        return region;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public String getSessionId() {
        return sessionId;
    }
    
    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }
    
    public Map<String, byte[]> getEncryptedDepartmentData() {
        return encryptedDepartmentData;
    }
    
    public Map<String, long[]> getOriginalData() {
        return originalData;
    }
    
    /**
     * Add encrypted data for a department
     */
    public void addDepartmentData(String department, byte[] encryptedData, long[] original) {
        encryptedDepartmentData.put(department, encryptedData);
        originalData.put(department, original);
    }
    
    /**
     * Check if hospital has data for a department
     */
    public boolean hasDepartment(String department) {
        return encryptedDepartmentData.containsKey(department);
    }
    
    /**
     * Get formatted creation time
     */
    public String getFormattedCreatedAt() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        return createdAt.format(formatter);
    }
    
    @Override
    public String toString() {
        return String.format("Hospital[%s] %s (%s) - %d departments", 
            id, name, region, encryptedDepartmentData.size());
    }
    
    /**
     * Display detailed info about this hospital
     */
    public String getDetailedInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║  Hospital ID: %-45s ║\n", id));
        sb.append(String.format("║  Name: %-50s ║\n", name));
        sb.append(String.format("║  Region: %-48s ║\n", region));
        sb.append(String.format("║  Created: %-47s ║\n", getFormattedCreatedAt()));
        sb.append(String.format("║  Session ID: %-44s ║\n", sessionId != null ? sessionId : "Not initialized"));
        sb.append("╠════════════════════════════════════════════════════════════╣\n");
        
        if (encryptedDepartmentData.isEmpty()) {
            sb.append("║  No department data yet                                    ║\n");
        } else {
            sb.append("║  Departments with encrypted data:                          ║\n");
            for (String dept : encryptedDepartmentData.keySet()) {
                byte[] encrypted = encryptedDepartmentData.get(dept);
                sb.append(String.format("║    • %-20s (%,d bytes encrypted)     ║\n", 
                    dept, encrypted.length));
            }
        }
        sb.append("╚════════════════════════════════════════════════════════════╝");
        return sb.toString();
    }
}
