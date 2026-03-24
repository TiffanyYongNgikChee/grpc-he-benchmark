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

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/**
 * ParameterComparison — Displays experiment results comparing different
 * HE parameter configurations (security level & activation degree).
 *
 * Data is from real EC2 experiments on t3.xlarge (7.6 GB RAM):
 *   - 10 MNIST test images per configuration
 *   - BFV scheme via OpenFHE, plaintext modulus 100,073,473
 *   - Scale factor = 1000
 */

/* ── Real experiment data from EC2 benchmarks ── */
const EXPERIMENT_DATA = {
  activation: [
    {
      parameter: "Activation",
      value: "x² (degree 2)",
      accuracy: "10/10",
      accuracyPct: 100,
      avgInferenceMs: 13919,
      encMs: 64,
      conv1Ms: 3475,
      act1Ms: 342,
      pool1Ms: 474,
      conv2Ms: 3133,
      act2Ms: 307,
      pool2Ms: 485,
      fcMs: 5611,
      decMs: 26,
      security: "128-bit",
      notes: "Best accuracy — polynomial x² matches training activation",
      floatAccuracy: "88.86%",
    },
    {
      parameter: "Activation",
      value: "degree 3",
      accuracy: "3/10",
      accuracyPct: 30,
      avgInferenceMs: 12160,
      encMs: 65,
      conv1Ms: 2777,
      act1Ms: 85,
      pool1Ms: 446,
      conv2Ms: 2885,
      act2Ms: 83,
      pool2Ms: 438,
      fcMs: 5353,
      decMs: 26,
      security: "128-bit",
      notes: "Signal corruption from modular clamping in poly_activate",
      floatAccuracy: "87.26%",
    },
    {
      parameter: "Activation",
      value: "degree 4",
      accuracy: "0/10",
      accuracyPct: 0,
      avgInferenceMs: 13696,
      encMs: 61,
      conv1Ms: 3454,
      act1Ms: 88,
      pool1Ms: 510,
      conv2Ms: 3348,
      act2Ms: 85,
      pool2Ms: 505,
      fcMs: 5617,
      decMs: 26,
      security: "128-bit",
      notes: "Complete signal loss — higher-degree terms overflow modulus",
      floatAccuracy: "86.91%",
    },
  ],
  security: [
    {
      parameter: "Security",
      value: "128-bit",
      accuracy: "10/10",
      accuracyPct: 100,
      avgInferenceMs: 13919,
      keygenNote: "< 5s",
      memoryNote: "~2.5 GB",
      security: "128-bit",
      notes: "Standard NIST security level — works on t3.xlarge",
    },
    {
      parameter: "Security",
      value: "192-bit",
      accuracy: "—",
      accuracyPct: null,
      avgInferenceMs: null,
      keygenNote: "> 60 min (OOM)",
      memoryNote: "> 7.6 GB",
      security: "192-bit",
      notes: "BFV context creation failed — requires larger ring dimension, exceeded 7.6 GB RAM + 15 GB swap",
    },
    {
      parameter: "Security",
      value: "256-bit",
      accuracy: "—",
      accuracyPct: null,
      avgInferenceMs: null,
      keygenNote: "N/A",
      memoryNote: "> 7.6 GB",
      security: "256-bit",
      notes: "Not attempted — 192-bit already infeasible on t3.xlarge",
    },
  ],
};

/* ── Colour helpers ── */
const accuracyColor = (pct) => {
  if (pct === null) return "#999";
  if (pct >= 80) return "#0aa35e";
  if (pct >= 40) return "#e68a00";
  return "#e03e52";
};

const formatMs = (ms) => {
  if (ms === null || ms === undefined) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms.toFixed(0)}ms`;
};

/* ── Main Component ── */
export default function ParameterComparison() {
  const [activeTab, setActiveTab] = useState("activation"); // "activation" | "security"

  const rows = EXPERIMENT_DATA[activeTab];

  return (
    <div>
      {/* Tab selector */}
      <div className="flex gap-2 mb-5">
        {[
          { id: "activation", label: "Activation Degree", desc: "x² vs degree 3 vs degree 4" },
          { id: "security", label: "Security Level", desc: "128-bit vs 192-bit vs 256-bit" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? "bg-[#1a1a2e] text-white shadow-md"
                : "bg-white text-gray-600 border border-gray-200 hover:border-gray-300 hover:bg-gray-50"
            }`}
          >
            {tab.label}
            <span className="block text-[10px] font-normal mt-0.5 opacity-70">{tab.desc}</span>
          </button>
        ))}
      </div>

      {/* Results table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ background: "#f8f8f8" }}>
              <th className="text-left px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                Configuration
              </th>
              <th className="text-center px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                Accuracy
              </th>
              <th className="text-center px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                Avg Inference
              </th>
              {activeTab === "activation" && (
                <th className="text-center px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                  Float Accuracy
                </th>
              )}
              {activeTab === "security" && (
                <>
                  <th className="text-center px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                    Keygen
                  </th>
                  <th className="text-center px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                    Memory
                  </th>
                </>
              )}
              <th className="text-left px-4 py-3 font-medium text-xs uppercase tracking-wider" style={{ color: "#888" }}>
                Notes
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-t transition-colors hover:bg-gray-50"
                style={{ borderColor: "#eee" }}
              >
                {/* Configuration */}
                <td className="px-4 py-3">
                  <div className="font-medium" style={{ color: "#333" }}>{row.value}</div>
                  <div className="text-[11px]" style={{ color: "#aaa" }}>{row.security}</div>
                </td>

                {/* Accuracy */}
                <td className="text-center px-4 py-3">
                  <span
                    className="inline-block px-2.5 py-1 rounded-full text-xs font-bold"
                    style={{
                      color: accuracyColor(row.accuracyPct),
                      background: row.accuracyPct === null ? "#f5f5f5" : `${accuracyColor(row.accuracyPct)}15`,
                    }}
                  >
                    {row.accuracy}
                  </span>
                </td>

                {/* Avg Inference */}
                <td className="text-center px-4 py-3 font-mono text-sm" style={{ color: row.avgInferenceMs ? "#333" : "#999" }}>
                  {row.avgInferenceMs ? formatMs(row.avgInferenceMs) : "—"}
                </td>

                {/* Conditional columns */}
                {activeTab === "activation" && (
                  <td className="text-center px-4 py-3 text-xs" style={{ color: "#666" }}>
                    {row.floatAccuracy || "—"}
                  </td>
                )}
                {activeTab === "security" && (
                  <>
                    <td className="text-center px-4 py-3 text-xs font-mono" style={{ color: row.keygenNote?.includes("OOM") ? "#e03e52" : "#666" }}>
                      {row.keygenNote || "—"}
                    </td>
                    <td className="text-center px-4 py-3 text-xs font-mono" style={{ color: "#666" }}>
                      {row.memoryNote || "—"}
                    </td>
                  </>
                )}

                {/* Notes */}
                <td className="px-4 py-3 text-xs max-w-[300px]" style={{ color: "#888" }}>
                  {row.notes}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-layer breakdown chart (activation tab only) */}
      {activeTab === "activation" && (
        <div className="mt-6">
          <h4 className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: "#888" }}>
            Per-Layer Timing Breakdown (ms)
          </h4>
          <div className="rounded-lg p-4" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
            <div style={{ height: 260 }}>
              <Bar
                data={{
                  labels: EXPERIMENT_DATA.activation.map((r) => r.value),
                  datasets: [
                    { label: "Encrypt", data: EXPERIMENT_DATA.activation.map((r) => r.encMs), backgroundColor: "#0db7c4" },
                    { label: "Conv1", data: EXPERIMENT_DATA.activation.map((r) => r.conv1Ms), backgroundColor: "#7b3ff2" },
                    { label: "Act1", data: EXPERIMENT_DATA.activation.map((r) => r.act1Ms), backgroundColor: "#e68a00" },
                    { label: "Pool1", data: EXPERIMENT_DATA.activation.map((r) => r.pool1Ms), backgroundColor: "#f4b942" },
                    { label: "Conv2", data: EXPERIMENT_DATA.activation.map((r) => r.conv2Ms), backgroundColor: "#9b6dff" },
                    { label: "Act2", data: EXPERIMENT_DATA.activation.map((r) => r.act2Ms), backgroundColor: "#ff9f43" },
                    { label: "Pool2", data: EXPERIMENT_DATA.activation.map((r) => r.pool2Ms), backgroundColor: "#ffc857" },
                    { label: "FC", data: EXPERIMENT_DATA.activation.map((r) => r.fcMs), backgroundColor: "#e03e52" },
                    { label: "Decrypt", data: EXPERIMENT_DATA.activation.map((r) => r.decMs), backgroundColor: "#0aa35e" },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: "bottom",
                      labels: { boxWidth: 12, padding: 10, font: { size: 10 } },
                    },
                    tooltip: {
                      callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(0)}ms`,
                      },
                    },
                  },
                  scales: {
                    x: { stacked: true, grid: { display: false } },
                    y: {
                      stacked: true,
                      title: { display: true, text: "Time (ms)", font: { size: 11 } },
                      grid: { color: "#f0f0f0" },
                    },
                  },
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Key findings callout */}
      <div className="mt-5 rounded-lg p-4" style={{ background: "#fffbe6", border: "1px solid #ffe58f" }}>
        <p className="text-xs font-medium mb-1.5" style={{ color: "#ad6800" }}>
          Key Findings
        </p>
        {activeTab === "activation" ? (
          <ul className="text-xs space-y-1" style={{ color: "#8c6200" }}>
            <li>• <b>x² (degree 2)</b> achieves 100% FHE accuracy (10/10) — matching the activation used during training gives best results.</li>
            <li>• <b>Degree 3</b> drops to 30% accuracy due to modular clamping — intermediate polynomial values overflow the plaintext modulus (100,073,473).</li>
            <li>• <b>Degree 4</b> drops to 0% accuracy — higher-degree terms cause complete signal corruption despite plaintext float model achieving 86.91%.</li>
            <li>• Inference time is similar across degrees (~12–14s) since the bottleneck is Conv/FC layers, not activation.</li>
          </ul>
        ) : (
          <ul className="text-xs space-y-1" style={{ color: "#8c6200" }}>
            <li>• <b>128-bit</b> security works well on t3.xlarge (7.6 GB RAM) with BFV keygen completing in under 5 seconds.</li>
            <li>• <b>192-bit</b> requires a much larger ring dimension (N), causing BFV context creation to exceed available memory (7.6 GB + 15 GB swap) — server never completed keygen after 60+ minutes.</li>
            <li>• <b>256-bit</b> would require even more memory and was not attempted.</li>
            <li>• Practical deployment: 128-bit is the feasible option for typical cloud instances. Higher security levels need 32 GB+ RAM (e.g., r5.xlarge).</li>
          </ul>
        )}
      </div>

      {/* Experiment metadata */}
      <div className="mt-3 text-[10px] flex flex-wrap gap-x-4 gap-y-1" style={{ color: "#bbb" }}>
        <span>EC2: t3.xlarge (4 vCPU, 7.6 GB RAM)</span>
        <span>OpenFHE v1.2.2 (BFV)</span>
        <span>p = 100,073,473</span>
        <span>mult_depth = 6</span>
        <span>scale = 1000</span>
        <span>10 MNIST test images per config</span>
      </div>
    </div>
  );
}
