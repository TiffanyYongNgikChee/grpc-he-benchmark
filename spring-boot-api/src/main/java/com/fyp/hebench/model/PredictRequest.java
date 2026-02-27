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

    // Default constructor required by Spring for JSON deserialisation
    public PredictRequest() {}

    public PredictRequest(List<Long> pixels, long scaleFactor) {
        this.pixels = pixels;
        this.scaleFactor = scaleFactor;
    }

    public List<Long> getPixels() { return pixels; }
    public void setPixels(List<Long> pixels) { this.pixels = pixels; }
    public long getScaleFactor() { return scaleFactor; }
    public void setScaleFactor(long scaleFactor) { this.scaleFactor = scaleFactor; }
}
