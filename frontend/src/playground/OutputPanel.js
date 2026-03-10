import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from "chart.js";
import CountUp from "../components/CountUp";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

/**
 * OutputPanel — The right-hand column of the playground.
 * Shows predicted digit, confidence, logits mini-chart, and status.
 */
export default function OutputPanel({ result, error, loading, pixels }) {
  /* Empty state */
  if (!result && !error && !loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 text-center">
        <p className="text-3xl mb-2">🧠</p>
        <p className="text-xs">
          {pixels
            ? "Press ▶ to run encrypted inference"
            : "Draw a digit on the left to start"}
        </p>
      </div>
    );
  }

  /* Loading state */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <svg className="animate-spin h-8 w-8 text-emerald-400 mb-3" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p className="text-xs text-emerald-400">Encrypting & inferring…</p>
      </div>
    );
  }

  /* Error state */
  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
        <p className="text-red-400 text-sm font-medium mb-1">Error</p>
        <p className="text-red-300 text-xs">{error}</p>
        <p className="text-red-400/50 text-xs mt-2">Make sure Docker + Spring Boot are running.</p>
      </div>
    );
  }

  /* Result */
  const logits = result.logits || [];
  const predicted = result.predictedDigit;

  /* Mini logits chart data */
  const chartData = {
    labels: logits.map((_, i) => String(i)),
    datasets: [
      {
        data: logits,
        backgroundColor: logits.map((_, i) =>
          i === predicted ? "rgba(16,185,129,0.85)" : "rgba(100,116,139,0.35)"
        ),
        borderColor: logits.map((_, i) =>
          i === predicted ? "rgb(16,185,129)" : "rgba(100,116,139,0.3)"
        ),
        borderWidth: 1,
        borderRadius: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (item) => `Logit: ${Number(item.raw).toFixed(3)}`,
        },
        backgroundColor: "rgba(15,23,42,0.95)",
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        padding: 8,
        cornerRadius: 4,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#64748b", font: { size: 10 } },
      },
      y: {
        grid: { color: "rgba(100,116,139,0.1)" },
        ticks: { display: false },
      },
    },
  };

  return (
    <div className="space-y-4">
      {/* Big predicted digit */}
      <div className="text-center bg-slate-800/60 rounded-lg border border-slate-700/50 py-4">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Predicted</p>
        <p className="text-6xl font-bold text-emerald-400 leading-none">
          <CountUp end={predicted} duration={400} decimals={0} />
        </p>
      </div>

      {/* Confidence + Time */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 p-2 text-center">
          <p className="text-[10px] text-slate-500">Confidence</p>
          <p className="text-base font-semibold text-white font-mono">
            <CountUp end={result.confidence * 100} duration={800} decimals={1} suffix="%" />
          </p>
        </div>
        <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 p-2 text-center">
          <p className="text-[10px] text-slate-500">Total</p>
          <p className="text-base font-semibold text-white font-mono">
            <CountUp end={result.totalMs} duration={800} decimals={0} suffix="ms" />
          </p>
        </div>
      </div>

      {/* Status badge */}
      <div className="flex justify-center">
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium ${
            result.status === "success"
              ? "bg-emerald-900/40 text-emerald-400 border border-emerald-800"
              : "bg-red-900/40 text-red-400 border border-red-800"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              result.status === "success" ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
          {result.status}
        </span>
      </div>

      {/* Mini logits chart */}
      <div>
        <p className="text-[10px] text-slate-500 mb-1">Output Logits</p>
        <div style={{ height: 100 }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Float model accuracy */}
      {result.floatModelAccuracy != null && (
        <p className="text-[10px] text-slate-500 text-center">
          Float model accuracy: <span className="text-slate-400">{result.floatModelAccuracy}%</span>
        </p>
      )}
    </div>
  );
}
