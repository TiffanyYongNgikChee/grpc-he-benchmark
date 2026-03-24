package com.fyp.hebench.model;

import java.util.List;

/**
 * PredictRequest - JSON request body for /api/predict endpoint
 * 
 * This is the Java POJO that Spring Boot automatically deserialises from JSON.
 * The frontend sends: {"pixels": [0,0,...,255,...,0], "scaleFactor": 1000}
 * Spring converts it into this object automatically.
 * 
 * Fields:
 *   - pixels: 784 pixel values (28×28 MNIST image, row-major order, values 0-255)
 *   - scaleFactor: quantisation scale factor for BFV integer encoding (default: 1000)
 */
public class PredictRequest {

    private List<Long> pixels;
    private long scaleFactor;
    private int securityLevel;  // 0=128-bit (default), 1=192-bit, 2=256-bit
    private int activationDegree;  // 2=x² (default), 3=cubic, 4=quartic

    // Default constructor required by Spring for JSON deserialisation
    public PredictRequest() {}

    public PredictRequest(List<Long> pixels, long scaleFactor) {
        this.pixels = pixels;
        this.scaleFactor = scaleFactor;
    }

    public PredictRequest(List<Long> pixels, long scaleFactor, int securityLevel) {
        this.pixels = pixels;
        this.scaleFactor = scaleFactor;
        this.securityLevel = securityLevel;
    }

    public PredictRequest(List<Long> pixels, long scaleFactor, int securityLevel, int activationDegree) {
        this.pixels = pixels;
        this.scaleFactor = scaleFactor;
        this.securityLevel = securityLevel;
        this.activationDegree = activationDegree;
    }

    public List<Long> getPixels() { return pixels; }
    public void setPixels(List<Long> pixels) { this.pixels = pixels; }
    public long getScaleFactor() { return scaleFactor; }
    public void setScaleFactor(long scaleFactor) { this.scaleFactor = scaleFactor; }
    public int getSecurityLevel() { return securityLevel; }
    public void setSecurityLevel(int securityLevel) { this.securityLevel = securityLevel; }
    public int getActivationDegree() { return activationDegree; }
    public void setActivationDegree(int activationDegree) { this.activationDegree = activationDegree; }
}
