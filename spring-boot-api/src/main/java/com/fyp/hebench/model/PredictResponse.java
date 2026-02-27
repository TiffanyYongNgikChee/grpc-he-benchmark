package com.fyp.hebench.model;

import java.util.List;

/**
 * PredictResponse - JSON response body from /api/predict endpoint
 * 
 * Spring Boot automatically converts this POJO to JSON when returning it.
 * The frontend receives something like:
 * {
 *   "predictedDigit": 7,
 *   "confidence": 0.43,
 *   "logits": [12, -5, 3, ...],
 *   "status": "success",
 *   "encryptionMs": 2.1,
 *   "conv1Ms": 5.3,
 *   ...
 *   "totalMs": 48.7
 * }
 */
public class PredictResponse {

    // === Prediction results ===
    private int predictedDigit;       // The digit the model predicted (0-9)
    private double confidence;        // How confident the model is (0.0 - 1.0)
    private List<Long> logits;        // Raw output values for each digit (10 values)
    private String status;            // "success" or error message

    // === Per-layer timing breakdown (milliseconds) ===
    private double encryptionMs;      // Time to encrypt input pixels
    private double conv1Ms;           // Conv1 layer time
    private double bias1Ms;           // Conv1 bias addition time
    private double act1Ms;            // Square activation after Conv1
    private double pool1Ms;           // AvgPool after Conv1
    private double conv2Ms;           // Conv2 layer time
    private double bias2Ms;           // Conv2 bias addition time
    private double act2Ms;            // Square activation after Conv2
    private double pool2Ms;           // AvgPool after Conv2
    private double fcMs;              // Fully connected layer time
    private double biasFcMs;          // FC bias addition time
    private double decryptionMs;      // Time to decrypt output
    private double totalMs;           // Total wall-clock time

    // === Model metadata ===
    private double floatModelAccuracy; // Plaintext model accuracy (e.g., 87.68)

    // Default constructor required by Spring for JSON serialisation
    public PredictResponse() {}

    // Getters and setters — Spring uses these to build the JSON response

    public int getPredictedDigit() { return predictedDigit; }
    public void setPredictedDigit(int predictedDigit) { this.predictedDigit = predictedDigit; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public List<Long> getLogits() { return logits; }
    public void setLogits(List<Long> logits) { this.logits = logits; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public double getEncryptionMs() { return encryptionMs; }
    public void setEncryptionMs(double encryptionMs) { this.encryptionMs = encryptionMs; }

    public double getConv1Ms() { return conv1Ms; }
    public void setConv1Ms(double conv1Ms) { this.conv1Ms = conv1Ms; }

    public double getBias1Ms() { return bias1Ms; }
    public void setBias1Ms(double bias1Ms) { this.bias1Ms = bias1Ms; }

    public double getAct1Ms() { return act1Ms; }
    public void setAct1Ms(double act1Ms) { this.act1Ms = act1Ms; }

    public double getPool1Ms() { return pool1Ms; }
    public void setPool1Ms(double pool1Ms) { this.pool1Ms = pool1Ms; }

    public double getConv2Ms() { return conv2Ms; }
    public void setConv2Ms(double conv2Ms) { this.conv2Ms = conv2Ms; }

    public double getBias2Ms() { return bias2Ms; }
    public void setBias2Ms(double bias2Ms) { this.bias2Ms = bias2Ms; }

    public double getAct2Ms() { return act2Ms; }
    public void setAct2Ms(double act2Ms) { this.act2Ms = act2Ms; }

    public double getPool2Ms() { return pool2Ms; }
    public void setPool2Ms(double pool2Ms) { this.pool2Ms = pool2Ms; }

    public double getFcMs() { return fcMs; }
    public void setFcMs(double fcMs) { this.fcMs = fcMs; }

    public double getBiasFcMs() { return biasFcMs; }
    public void setBiasFcMs(double biasFcMs) { this.biasFcMs = biasFcMs; }

    public double getDecryptionMs() { return decryptionMs; }
    public void setDecryptionMs(double decryptionMs) { this.decryptionMs = decryptionMs; }

    public double getTotalMs() { return totalMs; }
    public void setTotalMs(double totalMs) { this.totalMs = totalMs; }

    public double getFloatModelAccuracy() { return floatModelAccuracy; }
    public void setFloatModelAccuracy(double floatModelAccuracy) { this.floatModelAccuracy = floatModelAccuracy; }
}
