import { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

// Register the Chart.js components we need for grouped bar charts
ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/**
 * LibraryComparison — Side-by-side benchmark comparison of 3 HE libraries.
 *
 * HOW IT WORKS:
 *   1. The user types a list of numbers (e.g. "42, 7, 100") into the input field.
 *      These are the actual values that get encrypted, added, multiplied, and
 *      decrypted by each library. If left blank, the backend uses its default
 *      test data (sequential integers 0,1,2,…).
 *
 *   2. The user clicks "Run Library Comparison". This sends a POST request to
 *      /api/benchmark/compare with { library: "ALL", numOperations: 10, testValues: [...] }.
 *
 *   3. The Rust gRPC server runs the same 5 operations for each library:
 *        Key Gen   → create encryption keys
 *        Encrypt   → turn the plaintext numbers into ciphertexts
 *        Addition  → add pairs of ciphertexts together (homomorphic add)
 *        Multiply  → multiply pairs of ciphertexts (homomorphic multiply)
 *        Decrypt   → turn ciphertexts back into plaintext numbers
 *      Each operation is repeated 10 times and averaged (to smooth out noise).
 *
 *   4. The results come back as timing data (ms per operation per library).
 *      This component renders:
 *        • Summary cards — one per library, showing total time + "FASTEST" badge
 *        • Grouped bar chart — 5 bars per library, side by side
 *
 * WHERE THE DATA COMES FROM:
 *   Frontend (this file)
 *     → api/client.js  runComparisonBenchmark(10, testValues)
 *       → Spring Boot  POST /api/benchmark/compare
 *         → gRPC stub  RunComparisonBenchmark
 *           → Rust server  run_seal_benchmark() / run_helib_benchmark() / run_openfhe_benchmark()
 *             → Each calls the real C/C++ HE library (SEAL, HELib, OpenFHE)
 *
 * PROPS:
 *   data     — the comparison response from the backend (null until first run)
 *   loading  — true while the benchmark is in progress
 *   error    — error message string, if the benchmark failed
 *   onRun    — callback to trigger the benchmark (called with testValues array)
 */

// Colour palette for the three HE libraries
const LIB_COLORS = {
  SEAL:    { bg: "rgba(59,130,246,0.75)",  border: "#3b82f6"  },   // blue
  HELib:   { bg: "rgba(139,92,246,0.75)",  border: "#8b5cf6"  },   // purple
  OpenFHE: { bg: "rgba(13,183,196,0.75)",  border: "#0db7c4"  },   // teal
};

// The 5 primitive HE operations we benchmark.
// Each `key` matches a field name in the JSON response from the backend.
const OPERATIONS = [
  { key: "keyGenTimeMs",         label: "Key Gen" },    // time to generate public + secret keys
  { key: "encryptionTimeMs",     label: "Encrypt" },    // time to encrypt plaintext → ciphertext
  { key: "additionTimeMs",       label: "Addition" },   // time to add two ciphertexts (homomorphic)
  { key: "multiplicationTimeMs", label: "Multiply" },   // time to multiply two ciphertexts (homomorphic)
  { key: "decryptionTimeMs",     label: "Decrypt" },    // time to decrypt ciphertext → plaintext
];

export default function LibraryComparison({ data, loading, error, onRun }) {
  // The user can type custom numbers to encrypt/benchmark (e.g. "42, 7, 100")
  // If left empty, the backend uses default sequential test data.
  const [inputValues, setInputValues] = useState("");

  /**
   * Parse the user's comma-separated input into an array of integers.
   * Returns null if empty (meaning "use default data").
   */
  function parseTestValues() {
    const trimmed = inputValues.trim();
    if (!trimmed) return null; // empty → let backend use defaults
    return trimmed
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n));
  }

  /* ─── Shared input section (shown in both empty state and results) ─── */
  const inputSection = (
    <div className="mb-5">
      <label className="block text-xs font-medium mb-1.5" style={{ color: "#666" }}>
        Test Values
        <span className="font-normal ml-1" style={{ color: "#aaa" }}>
          (comma-separated integers to encrypt — leave blank for default)
        </span>
      </label>
      <input
        type="text"
        value={inputValues}
        onChange={(e) => setInputValues(e.target.value)}
        placeholder="e.g. 42, 7, 100, 256, 999"
        disabled={loading}
        className="w-full px-3 py-2 rounded-lg text-sm border focus:outline-none focus:ring-2 transition-all font-mono"
        style={{
          borderColor: "#d9d9d9",
          color: "#333",
          background: loading ? "#f5f5f5" : "#fff",
        }}
      />
      {inputValues.trim() && (
        <p className="text-[10px] mt-1" style={{ color: "#999" }}>
          {parseTestValues()?.length || 0} value(s) will be encrypted by each library
        </p>
      )}
    </div>
  );

  /* ─── Empty state (no results yet) ─── */
  if (!data && !loading && !error) {
    return (
      <div className="py-6">
        {inputSection}
        <div className="text-center">
          <p className="text-sm mb-4" style={{ color: "#888" }}>
            Compare primitive HE operations across all three libraries
          </p>
          <button
            onClick={() => onRun(parseTestValues())}
            className="px-5 py-2.5 rounded-lg text-sm font-medium text-white shadow-sm hover:shadow transition-all hover:scale-[1.02] active:scale-[0.98]"
            style={{ background: "#1a1a2e" }}
          >
            Run Library Comparison
          </button>
          {/* Colour legend for the 3 libraries */}
          <div className="flex justify-center gap-6 mt-6">
            {Object.entries(LIB_COLORS).map(([lib, c]) => (
              <span key={lib} className="flex items-center gap-1.5 text-xs" style={{ color: "#888" }}>
                <span className="inline-block w-3 h-3 rounded-sm" style={{ background: c.border }} />
                {lib}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ─── Loading spinner (benchmark is running on the server) ─── */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <svg className="animate-spin h-8 w-8 mb-3" style={{ color: "#1a1a2e" }} viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p className="text-xs" style={{ color: "#999" }}>Running benchmarks for SEAL, HElib, OpenFHE…</p>
        <p className="text-[10px] mt-1" style={{ color: "#bbb" }}>This may take a moment</p>
      </div>
    );
  }

  /* ─── Error state (backend returned an error or is unreachable) ─── */
  if (error) {
    return (
      <div className="text-center py-8">
        <div className="rounded-lg p-4 inline-block mx-auto" style={{ background: "#fef2f2", border: "1px solid #fecaca" }}>
          <p className="text-sm font-medium mb-1" style={{ color: "#dc2626" }}>Benchmark Failed</p>
          <p className="text-xs" style={{ color: "#b91c1c" }}>{error}</p>
        </div>
        <div className="mt-4">
          <button
            onClick={() => onRun(parseTestValues())}
            className="px-4 py-2 rounded-lg text-xs font-medium text-white"
            style={{ background: "#1a1a2e" }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  /* ─── Results ─── */
  // data.results is an array of 3 objects, one per library:
  // [{ library: "SEAL", keyGenTimeMs, encryptionTimeMs, additionTimeMs, ... }, ...]
  const results = data.results || [];
  if (results.length === 0) {
    return (
      <div className="text-center py-8 text-xs" style={{ color: "#999" }}>
        No benchmark results returned. Make sure all 3 libraries are available in Docker.
      </div>
    );
  }

  // Build one Chart.js dataset per library (each dataset has 5 bars — one per operation)
  const datasets = results.map((lib) => {
    const colors = LIB_COLORS[lib.library] || { bg: "rgba(150,150,150,0.6)", border: "#999" };
    return {
      label: lib.library,
      data: OPERATIONS.map((op) => lib[op.key] || 0),
      backgroundColor: colors.bg,
      borderColor: colors.border,
      borderWidth: 1,
      borderRadius: 3,
    };
  });

  // Chart.js data object — labels on the x-axis, 3 datasets (one per library)
  const chartData = {
    labels: OPERATIONS.map((op) => op.label),
    datasets,
  };

  // Chart.js config — grouped bars, y-axis in ms, tooltips, etc.
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
          pointStyle: "rectRounded",
          padding: 16,
          font: { size: 11, family: "'Roboto', sans-serif" },
          color: "#666",
        },
      },
      tooltip: {
        backgroundColor: "#333",
        titleColor: "#fff",
        bodyColor: "#ddd",
        padding: 10,
        cornerRadius: 6,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.raw).toFixed(2)} ms`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#888", font: { size: 11, family: "'Roboto', sans-serif" } },
      },
      y: {
        grid: { color: "rgba(0,0,0,0.06)" },
        ticks: {
          color: "#999",
          font: { size: 10, family: "'Roboto Mono', monospace" },
          callback: (v) => `${v} ms`,
        },
        title: {
          display: true,
          text: "Time (ms)",
          color: "#aaa",
          font: { size: 10, family: "'Roboto', sans-serif" },
        },
      },
    },
  };

  // Find the fastest library (lowest total time) for the "FASTEST" badge
  const fastest = results.reduce((a, b) => (a.totalTimeMs < b.totalTimeMs ? a : b));

  return (
    <div>
      {/* Test values input — lets user provide custom data to encrypt */}
      {inputSection}

      {/* Summary row — one card per library with total time + FASTEST badge */}
      <div className="flex gap-3 mb-5 flex-wrap">
        {results.map((lib) => {
          const colors = LIB_COLORS[lib.library] || { border: "#999" };
          const isFastest = lib.library === fastest.library;
          return (
            <div
              key={lib.library}
              className="flex-1 min-w-[140px] rounded-lg p-3 text-center relative"
              style={{
                background: "#fff",
                border: `2px solid ${isFastest ? colors.border : "#e5e5e5"}`,
              }}
            >
              {isFastest && (
                <span
                  className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full text-[9px] font-bold text-white"
                  style={{ background: colors.border }}
                >
                  FASTEST
                </span>
              )}
              <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>
                {lib.library}
              </p>
              <p className="text-2xl font-bold font-mono" style={{ color: colors.border }}>
                {lib.totalTimeMs.toFixed(1)}
                <span className="text-xs font-normal ml-0.5" style={{ color: "#bbb" }}>ms</span>
              </p>
              <p className="text-[10px] mt-1" style={{ color: lib.success ? "#059669" : "#dc2626" }}>
                {lib.success ? "Success" : lib.errorMessage || "Failed"}
              </p>
            </div>
          );
        })}
      </div>

      {/* Grouped bar chart — 5 operation categories, 3 bars each */}
      <div className="rounded-lg p-4" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
        <div style={{ height: 280 }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Rerun button — re-runs with the current test values */}
      <div className="text-center mt-4">
        <button
          onClick={() => onRun(parseTestValues())}
          className="px-4 py-1.5 rounded-lg text-xs font-medium border hover:bg-gray-50 transition-colors"
          style={{ borderColor: "#ccc", color: "#666" }}
        >
          Run Again
        </button>
      </div>
    </div>
  );
}
