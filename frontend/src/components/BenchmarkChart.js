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
 * The 6 HE operations we benchmark, in display order.
 * Each `key` matches the camelCase field from LibraryResult.
 */
const OPERATIONS = [
  { key: "keyGenTimeMs",          label: "KeyGen" },
  { key: "encryptionTimeMs",      label: "Encrypt" },
  { key: "additionTimeMs",        label: "Add" },
  { key: "multiplicationTimeMs",  label: "Multiply" },
  { key: "decryptionTimeMs",      label: "Decrypt" },
  { key: "totalTimeMs",           label: "Total" },
];

/**
 * One color per library — visually distinct, dark-theme friendly.
 */
const LIBRARY_COLORS = {
  SEAL:    { bg: "rgba(56, 189, 248, 0.75)", border: "rgb(56, 189, 248)"  },  // sky
  HELib:   { bg: "rgba(167, 139, 250, 0.75)", border: "rgb(167, 139, 250)" }, // violet
  OpenFHE: { bg: "rgba(251, 191, 36, 0.75)",  border: "rgb(251, 191, 36)"  }, // amber
};

/**
 * BenchmarkChart — Grouped bar chart comparing HE libraries.
 *
 * Props:
 *  - results: LibraryResult[]  (1 or 3 items)
 */
export default function BenchmarkChart({ results }) {
  const labels = OPERATIONS.map((op) => op.label);

  const datasets = results.map((lib) => {
    const colors = LIBRARY_COLORS[lib.library] || {
      bg: "rgba(148, 163, 184, 0.6)",
      border: "rgb(148, 163, 184)",
    };
    return {
      label: lib.library,
      data: OPERATIONS.map((op) => lib[op.key] ?? 0),
      backgroundColor: colors.bg,
      borderColor: colors.border,
      borderWidth: 1,
      borderRadius: 4,
    };
  });

  const data = { labels, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: results.length > 1,
        position: "top",
        labels: {
          color: "#94a3b8",
          font: { size: 12 },
          padding: 16,
          usePointStyle: true,
          pointStyleWidth: 12,
        },
      },
      tooltip: {
        callbacks: {
          label: (item) =>
            `${item.dataset.label}: ${Number(item.raw).toFixed(2)} ms`,
        },
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        borderColor: "rgba(100, 116, 139, 0.3)",
        borderWidth: 1,
        padding: 10,
        cornerRadius: 6,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          color: "#94a3b8",
          font: { size: 12 },
        },
        border: { display: false },
      },
      y: {
        grid: { color: "rgba(100, 116, 139, 0.12)" },
        ticks: {
          color: "#64748b",
          font: { size: 11 },
          callback: (v) => `${v} ms`,
        },
        border: { display: false },
        title: {
          display: true,
          text: "Time (ms)",
          color: "#64748b",
          font: { size: 11 },
        },
      },
    },
  };

  return (
    <div style={{ height: results.length > 1 ? 360 : 300 }}>
      <Bar data={data} options={options} />
    </div>
  );
}
