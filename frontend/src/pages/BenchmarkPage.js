import { useState } from "react";
import { runBenchmark, runComparisonBenchmark } from "../api/client";
import BenchmarkChart from "../components/BenchmarkChart";

const LIBRARIES = ["SEAL", "HELib", "OpenFHE"];
const OPERATIONS_OPTIONS = [1, 5, 10, 25, 50, 100];

/**
 * BenchmarkPage – Compare HE library performance.
 *
 * Two modes:
 *   • "single"  – benchmark one library at a time
 *   • "compare" – benchmark all three and compare side-by-side
 *
 * Commit 12: layout, mode toggle, library selector, operations dropdown.
 * Commit 13: API integration — calls runBenchmark / runComparisonBenchmark.
 * Chart and table will follow in Commits 14–15.
 */
export default function BenchmarkPage() {
  const [mode, setMode] = useState("compare"); // "single" | "compare"
  const [library, setLibrary] = useState("SEAL");
  const [numOperations, setNumOperations] = useState(10);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);   // always an array of LibraryResult
  const [error, setError] = useState(null);

  /**
   * Call the correct endpoint based on mode, then normalise into
   * a consistent results[] array for the chart / table (Commits 14–15).
   */
  async function handleRun() {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      if (mode === "compare") {
        const res = await runComparisonBenchmark(numOperations);
        // POST /api/benchmark/compare → { results: [LibraryResult …] }
        setResults(res.results);
      } else {
        const res = await runBenchmark(library, numOperations);
        // POST /api/benchmark/run → single LibraryResult object
        // Wrap in an array so downstream code can always iterate
        setResults([res]);
      }
    } catch (err) {
      setError(err.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-white mb-2">
        Library Benchmark
      </h1>
      <p className="text-slate-400 mb-8">
        Compare encryption performance across SEAL, HELib, and OpenFHE.
      </p>

      {/* ── Controls panel ── */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6 mb-8">
        <div className="flex flex-col sm:flex-row sm:items-end gap-6">

          {/* Mode toggle */}
          <div>
            <label className="block text-xs text-slate-400 mb-2 font-medium">
              Mode
            </label>
            <div className="inline-flex rounded-lg overflow-hidden border border-slate-600">
              <button
                onClick={() => setMode("single")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  mode === "single"
                    ? "bg-emerald-600 text-white"
                    : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                }`}
              >
                Single Library
              </button>
              <button
                onClick={() => setMode("compare")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  mode === "compare"
                    ? "bg-emerald-600 text-white"
                    : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                }`}
              >
                Compare All
              </button>
            </div>
          </div>

          {/* Library selector — only visible in single mode */}
          {mode === "single" && (
            <div>
              <label className="block text-xs text-slate-400 mb-2 font-medium">
                Library
              </label>
              <div className="inline-flex rounded-lg overflow-hidden border border-slate-600">
                {LIBRARIES.map((lib) => (
                  <button
                    key={lib}
                    onClick={() => setLibrary(lib)}
                    className={`px-4 py-2 text-sm font-medium transition-colors ${
                      library === lib
                        ? "bg-violet-600 text-white"
                        : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                    }`}
                  >
                    {lib}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Operations count dropdown */}
          <div>
            <label
              htmlFor="numOps"
              className="block text-xs text-slate-400 mb-2 font-medium"
            >
              Operations
            </label>
            <select
              id="numOps"
              value={numOperations}
              onChange={(e) => setNumOperations(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 text-slate-200 text-sm
                         rounded-lg px-4 py-2 focus:ring-emerald-500 focus:border-emerald-500
                         outline-none"
            >
              {OPERATIONS_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n} {n === 1 ? "operation" : "operations"}
                </option>
              ))}
            </select>
          </div>

          {/* Run button */}
          <div className="sm:ml-auto">
            <button
              onClick={handleRun}
              disabled={loading}
              className="px-6 py-2.5 rounded-lg font-semibold text-sm transition-all
                         bg-emerald-600 hover:bg-emerald-500 text-white
                         disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Running...
                </span>
              ) : mode === "compare" ? (
                "⚡ Compare All Libraries"
              ) : (
                `⚡ Run ${library} Benchmark`
              )}
            </button>
          </div>
        </div>

        {/* Inline description */}
        <p className="text-xs text-slate-500 mt-4">
          {mode === "compare"
            ? "Runs KeyGen → Encrypt → Add → Multiply → Decrypt on SEAL, HELib, and OpenFHE, then compares results side-by-side."
            : `Runs ${numOperations} iteration${numOperations > 1 ? "s" : ""} of each HE operation on ${library}.`}
        </p>
      </div>

      {/* ── Results area (Commits 13–15 will populate this) ── */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-8">
        {/* Error state */}
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-400 text-sm font-medium">Benchmark failed</p>
            <p className="text-red-300 text-xs mt-1">{error}</p>
            <p className="text-red-400/60 text-xs mt-2">
              Make sure Docker and Spring Boot are running.
            </p>
          </div>
        )}

        {/* Empty state */}
        {!results && !error && !loading && (
          <div className="text-center text-slate-500 py-16">
            <p className="text-4xl mb-3">📊</p>
            <p className="text-sm">
              {mode === "compare"
                ? 'Click "Compare All Libraries" to benchmark SEAL, HELib, and OpenFHE'
                : `Click "Run ${library} Benchmark" to start`}
            </p>
          </div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="text-center py-16">
            <div className="inline-flex items-center gap-3 text-emerald-400">
              <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span className="text-sm">
                {mode === "compare"
                  ? "Benchmarking all 3 libraries… this may take a moment"
                  : `Running ${library} benchmark…`}
              </span>
            </div>
          </div>
        )}

        {/* Chart + Table (table coming in Commit 15) */}
        {results && !loading && (
          <div>
            <h3 className="text-sm font-medium text-slate-300 mb-4">
              {results.length > 1
                ? "Performance Comparison — Time per Operation"
                : `${results[0].library} — Time per Operation`}
            </h3>
            <BenchmarkChart results={results} />
          </div>
        )}
      </div>
    </div>
  );
}
