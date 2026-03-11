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
 * OutputPanel — Right column showing prediction results.
 * Light theme with clean, minimal style.
 */
export default function OutputPanel({ result, error, loading, pixels }) {
  /* Empty state */
  if (!result && !error && !loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center" style={{ color: "#bbb" }}>
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#ccc" strokeWidth="1.5" className="mb-2">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 12h8M12 8v8" strokeLinecap="round" />
        </svg>
        <p className="text-xs">
          {pixels ? "Press the run button to start encrypted inference" : "Draw a digit on the left to start"}
        </p>
      </div>
    );
  }

  /* Loading */
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <svg className="animate-spin h-8 w-8 mb-3" style={{ color: "#f4743a" }} viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p className="text-xs" style={{ color: "#f4743a" }}>Encrypting & inferring…</p>
      </div>
    );
  }

  /* Error */
  if (error) {
    return (
      <div className="rounded-lg p-4" style={{ background: "#fef2f2", border: "1px solid #fecaca" }}>
        <p className="text-sm font-medium mb-1" style={{ color: "#dc2626" }}>Error</p>
        <p className="text-xs" style={{ color: "#b91c1c" }}>{error}</p>
        <p className="text-xs mt-2" style={{ color: "#f87171" }}>Make sure Docker + Spring Boot are running.</p>
      </div>
    );
  }

  /* Result */
  const logits = result.logits || [];
  const predicted = result.predictedDigit;

  const chartData = {
    labels: logits.map((_, i) => String(i)),
    datasets: [
      {
        data: logits,
        backgroundColor: logits.map((_, i) =>
          i === predicted ? "rgba(244,116,58,0.85)" : "rgba(200,200,200,0.5)"
        ),
        borderColor: logits.map((_, i) =>
          i === predicted ? "#f4743a" : "rgba(200,200,200,0.4)"
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
        backgroundColor: "#333",
        titleColor: "#fff",
        bodyColor: "#ddd",
        padding: 8,
        cornerRadius: 4,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#999", font: { size: 10 } },
      },
      y: {
        grid: { color: "rgba(0,0,0,0.05)" },
        ticks: { display: false },
      },
    },
  };

  return (
    <div className="space-y-4">
      {/* Big predicted digit */}
      <div
        className="text-center rounded-lg py-5"
        style={{ background: "#fff", border: "1px solid #d9d9d9" }}
      >
        <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>Predicted</p>
        <p className="text-6xl font-bold leading-none" style={{ color: "#f4743a" }}>
          <CountUp end={predicted} duration={400} decimals={0} />
        </p>
      </div>

      {/* Confidence + Time */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg p-2 text-center" style={{ background: "#fff", border: "1px solid #d9d9d9" }}>
          <p className="text-[10px]" style={{ color: "#999" }}>Confidence</p>
          <p className="text-base font-semibold font-mono" style={{ color: "#333" }}>
            <CountUp end={result.confidence * 100} duration={800} decimals={1} suffix="%" />
          </p>
        </div>
        <div className="rounded-lg p-2 text-center" style={{ background: "#fff", border: "1px solid #d9d9d9" }}>
          <p className="text-[10px]" style={{ color: "#999" }}>Total</p>
          <p className="text-base font-semibold font-mono" style={{ color: "#333" }}>
            <CountUp end={result.totalMs} duration={800} decimals={0} suffix="ms" />
          </p>
        </div>
      </div>

      {/* Status badge */}
      <div className="flex justify-center">
        <span
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium"
          style={{
            background: result.status === "success" ? "#ecfdf5" : "#fef2f2",
            color: result.status === "success" ? "#059669" : "#dc2626",
            border: `1px solid ${result.status === "success" ? "#a7f3d0" : "#fecaca"}`,
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: result.status === "success" ? "#059669" : "#dc2626" }}
          />
          {result.status}
        </span>
      </div>

      {/* Logits chart */}
      <div>
        <p className="text-[10px] mb-1" style={{ color: "#999" }}>Output Logits</p>
        <div style={{ height: 100 }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Float model accuracy */}
      {result.floatModelAccuracy != null && (
        <p className="text-[10px] text-center" style={{ color: "#bbb" }}>
          Float model accuracy: <span style={{ color: "#999" }}>{result.floatModelAccuracy}%</span>
        </p>
      )}
    </div>
  );
}
