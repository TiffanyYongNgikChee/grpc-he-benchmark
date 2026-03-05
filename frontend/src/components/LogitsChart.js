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
 * LogitsChart — Vertical bar chart showing the 10 output logits.
 * The predicted digit's bar is highlighted in emerald; others are slate.
 *
 * Props:
 *  - logits: number[10]  (raw model outputs)
 *  - predictedDigit: number (0-9)
 */
export default function LogitsChart({ logits, predictedDigit }) {
  const labels = Array.from({ length: 10 }, (_, i) => String(i));

  const barColors = logits.map((_, i) =>
    i === predictedDigit ? "rgba(52, 211, 153, 0.85)" : "rgba(100, 116, 139, 0.5)"
  );
  const borderColors = logits.map((_, i) =>
    i === predictedDigit ? "rgb(16, 185, 129)" : "rgba(100, 116, 139, 0.2)"
  );

  const data = {
    labels,
    datasets: [
      {
        label: "Logit value",
        data: logits,
        backgroundColor: barColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        callbacks: {
          title: (items) => `Digit ${items[0].label}`,
          label: (item) => `Logit: ${Number(item.raw).toFixed(4)}`,
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
        grid: { display: false },
        ticks: {
          color: (ctx) =>
            ctx.index === predictedDigit ? "#34d399" : "#64748b",
          font: (ctx) => ({
            size: 12,
            weight: ctx.index === predictedDigit ? "bold" : "normal",
          }),
        },
        border: { display: false },
      },
      y: {
        grid: { color: "rgba(100, 116, 139, 0.15)" },
        ticks: {
          color: "#64748b",
          font: { size: 10 },
          maxTicksLimit: 5,
        },
        border: { display: false },
      },
    },
  };

  return (
    <div style={{ height: 220 }}>
      <Bar data={data} options={options} />
    </div>
  );
}
