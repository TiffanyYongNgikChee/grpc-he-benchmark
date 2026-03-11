import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/**
 * LibraryComparison — Side-by-side benchmark comparison of 3 HE libraries.
 *
 * Shows grouped bar charts for each primitive operation (keygen, encrypt,
 * add, multiply, decrypt) with SEAL (blue), HElib (purple), OpenFHE (teal).
 */

const LIB_COLORS = {
  SEAL:    { bg: "rgba(59,130,246,0.75)",  border: "#3b82f6"  },
  HELib:   { bg: "rgba(139,92,246,0.75)",  border: "#8b5cf6"  },
  OpenFHE: { bg: "rgba(13,183,196,0.75)",  border: "#0db7c4"  },
};

const OPERATIONS = [
  { key: "keyGenTimeMs",         label: "Key Gen" },
  { key: "encryptionTimeMs",     label: "Encrypt" },
  { key: "additionTimeMs",       label: "Addition" },
  { key: "multiplicationTimeMs", label: "Multiply" },
  { key: "decryptionTimeMs",     label: "Decrypt" },
];

export default function LibraryComparison({ data, loading, error, onRun }) {
  /* ─── Empty state ─── */
  if (!data && !loading && !error) {
    return (
      <div className="text-center py-10">
        <p className="text-sm mb-4" style={{ color: "#888" }}>
          Compare primitive HE operations across all three libraries
        </p>
        <button
          onClick={onRun}
          className="px-5 py-2.5 rounded-lg text-sm font-medium text-white shadow-sm hover:shadow transition-all hover:scale-[1.02] active:scale-[0.98]"
          style={{ background: "#1a1a2e" }}
        >
          Run Library Comparison
        </button>
        <div className="flex justify-center gap-6 mt-6">
          {Object.entries(LIB_COLORS).map(([lib, c]) => (
            <span key={lib} className="flex items-center gap-1.5 text-xs" style={{ color: "#888" }}>
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: c.border }} />
              {lib}
            </span>
          ))}
        </div>
      </div>
    );
  }

  /* ─── Loading ─── */
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

  /* ─── Error ─── */
  if (error) {
    return (
      <div className="text-center py-8">
        <div className="rounded-lg p-4 inline-block mx-auto" style={{ background: "#fef2f2", border: "1px solid #fecaca" }}>
          <p className="text-sm font-medium mb-1" style={{ color: "#dc2626" }}>Benchmark Failed</p>
          <p className="text-xs" style={{ color: "#b91c1c" }}>{error}</p>
        </div>
        <div className="mt-4">
          <button
            onClick={onRun}
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
  const results = data.results || [];
  if (results.length === 0) {
    return (
      <div className="text-center py-8 text-xs" style={{ color: "#999" }}>
        No benchmark results returned. Make sure all 3 libraries are available in Docker.
      </div>
    );
  }

  /* Build datasets for grouped bar chart */
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

  const chartData = {
    labels: OPERATIONS.map((op) => op.label),
    datasets,
  };

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

  /* Summary cards */
  const fastest = results.reduce((a, b) => (a.totalTimeMs < b.totalTimeMs ? a : b));

  return (
    <div>
      {/* Summary row */}
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

      {/* Grouped bar chart */}
      <div className="rounded-lg p-4" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
        <div style={{ height: 280 }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Rerun button */}
      <div className="text-center mt-4">
        <button
          onClick={onRun}
          className="px-4 py-1.5 rounded-lg text-xs font-medium border hover:bg-gray-50 transition-colors"
          style={{ borderColor: "#ccc", color: "#666" }}
        >
          Run Again
        </button>
      </div>
    </div>
  );
}
