package com.fyp.hebench.model;

import java.util.List;

public class BenchmarkRequest {
    private String library;
    private int numOperations;
    /** Optional custom integer values to encrypt in the benchmark.
     *  If null or empty, the backend uses default sequential test data. */
    private List<Long> testValues;

    public BenchmarkRequest() {}

    public BenchmarkRequest(String library, int numOperations) {
        this.library = library;
        this.numOperations = numOperations;
    }

    public String getLibrary() { return library; }
    public void setLibrary(String library) { this.library = library; }
    public int getNumOperations() { return numOperations; }
    public void setNumOperations(int numOperations) { this.numOperations = numOperations; }
    public List<Long> getTestValues() { return testValues; }
    public void setTestValues(List<Long> testValues) { this.testValues = testValues; }
}
