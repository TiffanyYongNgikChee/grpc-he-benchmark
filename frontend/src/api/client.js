/**
 * API client for communicating with the Spring Boot backend.
 * All fetch calls go through here.
 *
 * To switch between local and EC2:
 *   - Local dev:  REACT_APP_API_BASE is unset → defaults to localhost:8080
 *   - EC2 deploy: REACT_APP_API_BASE=http://54.205.254.22:8080/api
 */

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8080/api";

/**
 * Check if the backend server is running.
 * GET /api/health → "OK"
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Run encrypted MNIST digit prediction.
 * POST /api/predict
 * @param {number[]} pixels - 784 pixel values (28×28 image)
 * @param {number} scaleFactor - BFV quantisation scale (default: 1000)
 * @param {object} [options] - Optional parameters
 * @param {number} [options.securityLevel] - 0=128-bit, 1=192-bit, 2=256-bit
 * @param {number} [options.activationDegree] - 2=x², 3=cubic, 4=quartic
 * @param {AbortSignal} [options.signal] - Optional AbortSignal to cancel the request
 */
export async function predictDigit(pixels, scaleFactor = 1000, options = {}) {
  const { securityLevel = 0, activationDegree = 2, signal } = typeof options === 'object' && options !== null && !('aborted' in options)
    ? options
    : { signal: options }; // backward compat: if options is an AbortSignal

  const timeoutSignal = AbortSignal.timeout(300_000); // 5 min — FHE inference can take ~27s+ per image
  // Combine external signal (if provided) with timeout signal
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal;

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels, scaleFactor, securityLevel, activationDegree }),
    signal: combinedSignal,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Run benchmark for a single HE library.
 * POST /api/benchmark/run
 * @param {string} library - "SEAL", "HELib", or "OpenFHE"
 * @param {number} numOperations - How many times to repeat each operation
 */
export async function runBenchmark(library, numOperations = 10) {
  const res = await fetch(`${API_BASE}/benchmark/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ library, numOperations }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Run comparison benchmark for all 3 HE libraries.
 * POST /api/benchmark/compare
 * @param {number} numOperations - How many times to repeat each operation
 * @param {number[]|null} testValues - Custom integers to encrypt (null = use defaults)
 */
export async function runComparisonBenchmark(numOperations = 10, testValues = null, signal) {
  const body = { library: "ALL", numOperations };
  if (testValues && testValues.length > 0) {
    body.testValues = testValues;
  }
  const timeoutSignal = AbortSignal.timeout(180_000); // 3 min timeout
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal;

  const res = await fetch(`${API_BASE}/benchmark/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: combinedSignal,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `HTTP ${res.status}`);
  }
  return res.json();
}
