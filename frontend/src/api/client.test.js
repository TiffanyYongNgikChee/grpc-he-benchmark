/**
 * Tests for the API client module.
 *
 * These tests verify that the frontend correctly constructs requests
 * to the Spring Boot backend, handles errors, and manages timeouts.
 * All fetch calls are mocked — no real server needed.
 */

// AbortSignal.timeout may not exist in JSDOM — polyfill for tests
if (!AbortSignal.timeout) {
  AbortSignal.timeout = (ms) => {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(new DOMException("TimeoutError")), ms);
    return ctrl.signal;
  };
}
if (!AbortSignal.any) {
  AbortSignal.any = (signals) => {
    const ctrl = new AbortController();
    for (const s of signals) {
      if (s.aborted) { ctrl.abort(s.reason); return ctrl.signal; }
      s.addEventListener("abort", () => ctrl.abort(s.reason), { once: true });
    }
    return ctrl.signal;
  };
}

import { checkHealth, predictDigit, runBenchmark, runComparisonBenchmark } from "./client";

// Mock fetch globally
beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.restoreAllMocks();
});

/* ══════════════════════════════════════════
   checkHealth
   ══════════════════════════════════════════ */
describe("checkHealth", () => {
  it("returns true when server responds OK", async () => {
    global.fetch.mockResolvedValue({ ok: true });
    const result = await checkHealth();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/health"));
  });

  it("returns false when server responds with error", async () => {
    global.fetch.mockResolvedValue({ ok: false });
    const result = await checkHealth();
    expect(result).toBe(false);
  });

  it("returns false when fetch throws (server unreachable)", async () => {
    global.fetch.mockRejectedValue(new Error("Network error"));
    const result = await checkHealth();
    expect(result).toBe(false);
  });
});

/* ══════════════════════════════════════════
   predictDigit
   ══════════════════════════════════════════ */
describe("predictDigit", () => {
  const mockPixels = new Array(784).fill(0);
  const mockResponse = {
    predictedDigit: 7,
    confidence: 0.95,
    totalMs: 15000,
    status: "success",
  };

  it("sends correct request body with default parameters", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await predictDigit(mockPixels);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/predict"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pixels: mockPixels,
          scaleFactor: 1000,
          securityLevel: 0,
          activationDegree: 2,
        }),
      })
    );
    expect(result.predictedDigit).toBe(7);
  });

  it("passes custom security level and activation degree", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    await predictDigit(mockPixels, 1000, { securityLevel: 1, activationDegree: 3 });

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.securityLevel).toBe(1);
    expect(callBody.activationDegree).toBe(3);
  });

  it("uses default scale factor of 1000", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    await predictDigit(mockPixels);

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.scaleFactor).toBe(1000);
  });

  it("throws on HTTP error response", async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ error: "Internal server error" }),
    });

    await expect(predictDigit(mockPixels)).rejects.toThrow("Internal server error");
  });

  it("handles non-JSON error response gracefully", async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 502,
      json: () => Promise.reject(new Error("Not JSON")),
    });

    await expect(predictDigit(mockPixels)).rejects.toThrow("Unknown error");
  });
});

/* ══════════════════════════════════════════
   runBenchmark
   ══════════════════════════════════════════ */
describe("runBenchmark", () => {
  it("sends correct library and numOperations", async () => {
    const mockData = { keyGenTimeMs: 5.2, encryptionTimeMs: 1.1 };
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const result = await runBenchmark("SEAL", 20);

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.library).toBe("SEAL");
    expect(callBody.numOperations).toBe(20);
    expect(result.keyGenTimeMs).toBe(5.2);
  });

  it("defaults to 10 operations", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });

    await runBenchmark("OpenFHE");

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.numOperations).toBe(10);
  });
});

/* ══════════════════════════════════════════
   runComparisonBenchmark
   ══════════════════════════════════════════ */
describe("runComparisonBenchmark", () => {
  it("sends ALL library with default parameters", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await runComparisonBenchmark();

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.library).toBe("ALL");
    expect(callBody.numOperations).toBe(10);
    expect(callBody.testValues).toBeUndefined();
  });

  it("includes testValues when provided", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await runComparisonBenchmark(10, [42, 7, 100]);

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.testValues).toEqual([42, 7, 100]);
  });

  it("does not include testValues when null", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await runComparisonBenchmark(10, null);

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.testValues).toBeUndefined();
  });

  it("does not include testValues when empty array", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await runComparisonBenchmark(10, []);

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.testValues).toBeUndefined();
  });
});
