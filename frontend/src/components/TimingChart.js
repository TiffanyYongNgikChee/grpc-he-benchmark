import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

/**
 * Color palette — each layer category gets a distinct hue:
 *   Encrypt / Decrypt → cyan
 *   Conv / Bias        → violet
 *   ReLU / Pool        → amber
 *   FC / BiasFc        → rose
 */
const LAYER_META = [
  { key: "encryptionMs", label: "Encryption", color: "rgba(34,211,238,0.8)",  border: "rgb(34,211,238)"  },
  { key: "conv1Ms",      label: "Conv1",      color: "rgba(139,92,246,0.8)",  border: "rgb(139,92,246)"  },
  { key: "bias1Ms",      label: "Bias1",      color: "rgba(167,139,250,0.7)", border: "rgb(167,139,250)" },
  { key: "act1Ms",       label: "ReLU1",      color: "rgba(245,158,11,0.8)",  border: "rgb(245,158,11)"  },
  { key: "pool1Ms",      label: "Pool1",      color: "rgba(251,191,36,0.7)",  border: "rgb(251,191,36)"  },
  { key: "conv2Ms",      label: "Conv2",      color: "rgba(139,92,246,0.8)",  border: "rgb(139,92,246)"  },
  { key: "bias2Ms",      label: "Bias2",      color: "rgba(167,139,250,0.7)", border: "rgb(167,139,250)" },
  { key: "act2Ms",       label: "ReLU2",      color: "rgba(245,158,11,0.8)",  border: "rgb(245,158,11)"  },
  { key: "pool2Ms",      label: "Pool2",      color: "rgba(251,191,36,0.7)",  border: "rgb(251,191,36)"  },
  { key: "fcMs",         label: "FC",         color: "rgba(244,63,94,0.8)",   border: "rgb(244,63,94)"   },
  { key: "biasFcMs",     label: "BiasFc",     color: "rgba(251,113,133,0.7)", border: "rgb(251,113,133)" },
  { key: "decryptionMs", label: "Decryption", color: "rgba(45,212,191,0.8)",  border: "rgb(45,212,191)"  },
];

/**
 * TimingChart — Horizontal bar chart showing per-layer inference times.
 *
 * Props:
 *  - result: the full prediction response object (needs *.Ms fields + totalMs)
 */
export default function TimingChart({ result }) {
  const labels = LAYER_META.map((l) => l.label);
  const values = LAYER_META.map((l) => result[l.key] ?? 0);
  const bgColors = LAYER_META.map((l) => l.color);
  const bdColors = LAYER_META.map((l) => l.border);

  const data = {
    labels,
    datasets: [
      {
        label: "Time (ms)",
        data: values,
        backgroundColor: bgColors,
        borderColor: bdColors,
        borderWidth: 1,
        borderRadius: 3,
        borderSkipped: false,
      },
    ],
  };

  const options = {
    indexAxis: "y",          // horizontal bars
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        callbacks: {
          title: (items) => items[0].label,
          label: (item) => {
            const ms = Number(item.raw).toFixed(2);
            const pct = ((item.raw / result.totalMs) * 100).toFixed(1);
            return `${ms} ms  (${pct}%)`;
          },
        },
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        borderColor: "rgba(100, 116, 139, 0.3)",
        borderWidth: 1,
        padding: 10,
        cornerRadius: 6,
      },
      legend: { display: false },
    },
    scales: {
      x: {
        grid: { color: "rgba(100, 116, 139, 0.12)" },
        ticks: {
          color: "#64748b",
          font: { size: 10 },
          callback: (v) => `${v} ms`,
        },
        border: { display: false },
        title: {
          display: true,
          text: "Time (ms)",
          color: "#64748b",
          font: { size: 10 },
        },
      },
      y: {
        grid: { display: false },
        ticks: {
          color: "#94a3b8",
          font: { size: 11 },
        },
        border: { display: false },
      },
    },
  };

  return (
    <div style={{ height: 340 }}>
      <Bar data={data} options={options} />
    </div>
  );
}
