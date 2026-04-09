package com.fyp.hebench.controller;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fyp.hebench.model.BenchmarkResponse;
import com.fyp.hebench.model.PredictResponse;
import com.fyp.hebench.service.GrpcClientService;

/**
 * Integration tests for BenchmarkController.
 *
 * Uses @WebMvcTest to load only the web layer (no gRPC connections needed).
 * GrpcClientService is replaced with a Mockito mock so we can test
 * validation, error handling, and response formatting in isolation.
 */
@WebMvcTest(BenchmarkController.class)
class BenchmarkControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private GrpcClientService grpcClientService;

    /* ══════════════════════════════════════
       GET /api/health
       ══════════════════════════════════════ */
    @Nested
    @DisplayName("GET /api/health")
    class HealthEndpoint {

        @Test
        @DisplayName("returns OK")
        void healthReturnsOk() throws Exception {
            mockMvc.perform(get("/api/health"))
                    .andExpect(status().isOk())
                    .andExpect(content().string("OK"));
        }
    }

    /* ══════════════════════════════════════
       POST /api/predict
       ══════════════════════════════════════ */
    @Nested
    @DisplayName("POST /api/predict")
    class PredictEndpoint {

        /** Helper: build a JSON body with n pixels */
        private String pixelBody(int n, long scaleFactor) {
            String pixels = IntStream.range(0, n)
                    .mapToObj(i -> "0")
                    .collect(Collectors.joining(","));
            return String.format(
                    "{\"pixels\":[%s],\"scaleFactor\":%d,\"securityLevel\":0,\"activationDegree\":2}",
                    pixels, scaleFactor);
        }

        @Test
        @DisplayName("rejects request with wrong pixel count (not 784)")
        void rejectsBadPixelCount() throws Exception {
            // Send only 10 pixels instead of 784
            mockMvc.perform(post("/api/predict")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(pixelBody(10, 1000)))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error", containsString("784")))
                    .andExpect(jsonPath("$.received", is(10)));
        }

        @Test
        @DisplayName("rejects request with null pixels")
        void rejectsNullPixels() throws Exception {
            mockMvc.perform(post("/api/predict")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{\"scaleFactor\":1000}"))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error", containsString("784")));
        }

        @Test
        @DisplayName("accepts valid 784-pixel request and returns prediction")
        void acceptsValid784Pixels() throws Exception {
            // Mock gRPC response
            PredictResponse mockResponse = new PredictResponse();
            mockResponse.setPredictedDigit(7);
            mockResponse.setConfidence(0.95);
            mockResponse.setTotalMs(15000);
            mockResponse.setStatus("success");
            mockResponse.setEncryptionMs(64);
            mockResponse.setConv1Ms(3475);
            mockResponse.setDecryptionMs(26);

            when(grpcClientService.predictDigit(anyList(), eq(1000L), eq(0), eq(2)))
                    .thenReturn(mockResponse);

            mockMvc.perform(post("/api/predict")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(pixelBody(784, 1000)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.predictedDigit", is(7)))
                    .andExpect(jsonPath("$.confidence", is(0.95)))
                    .andExpect(jsonPath("$.totalMs", is(15000.0)));
        }

        @Test
        @DisplayName("returns 500 when gRPC server is unreachable")
        void handlesGrpcConnectionError() throws Exception {
            when(grpcClientService.predictDigit(anyList(), anyLong(), anyInt(), anyInt()))
                    .thenThrow(new io.grpc.StatusRuntimeException(
                            io.grpc.Status.UNAVAILABLE.withDescription("Connection refused")));

            mockMvc.perform(post("/api/predict")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(pixelBody(784, 1000)))
                    .andExpect(status().isInternalServerError())
                    .andExpect(jsonPath("$.error", is("Failed to connect to HE inference engine")))
                    .andExpect(jsonPath("$.hint", containsString("Docker")));
        }

        @Test
        @DisplayName("defaults scaleFactor to 1000 when 0 or negative")
        void defaultsScaleFactor() throws Exception {
            PredictResponse mockResponse = new PredictResponse();
            mockResponse.setPredictedDigit(3);
            mockResponse.setConfidence(0.88);
            mockResponse.setTotalMs(12000);
            mockResponse.setStatus("success");

            // When scaleFactor <= 0, controller should use 1000
            when(grpcClientService.predictDigit(anyList(), eq(1000L), anyInt(), anyInt()))
                    .thenReturn(mockResponse);

            mockMvc.perform(post("/api/predict")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(pixelBody(784, 0)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.predictedDigit", is(3)));
        }
    }

    /* ══════════════════════════════════════
       POST /api/benchmark/run
       ══════════════════════════════════════ */
    @Nested
    @DisplayName("POST /api/benchmark/run")
    class BenchmarkRunEndpoint {

        @Test
        @DisplayName("runs SEAL benchmark and returns timing metrics")
        void runsSealBenchmark() throws Exception {
            BenchmarkResponse mockResponse = new BenchmarkResponse(
                    "SEAL", 5.2, 1.1, 0.3, 2.8, 0.9, 10.3, true, "");

            when(grpcClientService.runBenchmark(eq("SEAL"), eq(10), eq(null)))
                    .thenReturn(mockResponse);

            mockMvc.perform(post("/api/benchmark/run")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{\"library\":\"SEAL\",\"numOperations\":10}"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.library", is("SEAL")))
                    .andExpect(jsonPath("$.keyGenTimeMs", is(5.2)))
                    .andExpect(jsonPath("$.success", is(true)));
        }

        @Test
        @DisplayName("returns 500 when gRPC throws exception")
        void handlesGrpcError() throws Exception {
            when(grpcClientService.runBenchmark(eq("HELib"), anyInt(), eq(null)))
                    .thenThrow(new RuntimeException("Server down"));

            mockMvc.perform(post("/api/benchmark/run")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{\"library\":\"HELib\",\"numOperations\":10}"))
                    .andExpect(status().isInternalServerError());
        }
    }
}
