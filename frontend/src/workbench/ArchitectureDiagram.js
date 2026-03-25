/**
 * ArchitectureDiagram — Educational visual walkthrough of the encrypted CNN pipeline.
 *
 * Features:
 *   1. Pipeline overview — 10-step horizontal flow (encrypt → conv → act → pool → … → decrypt)
 *   2. Interactive convolution demo — CS231n-style animated sliding filter with numbers,
 *      showing exactly how a 3×3 kernel produces one output value at a time.
 *   3. Activation & pooling mini-demos — visual "before → after" for x² and 2×2 avg pool.
 *
 * Pure React + inline SVG, no external dependencies.
 */

import { useState, useEffect, useCallback } from "react";

/* ── Colour palette ── */
const C = {
  io:     "#0aa35e",
  crypto: "#0db7c4",
  conv:   "#7b3ff2",
  act:    "#e68a00",
  fc:     "#e03e52",
  bg:     "#f8f8f8",
  border: "#e5e5e5",
  text:   "#333",
  muted:  "#888",
  faint:  "#bbb",
};

/* ════════════════════════════════════════════════════════════════════════
   Section 1 — Pipeline overview (compact 10-step horizontal strip)
   ════════════════════════════════════════════════════════════════════════ */

function PipelineOverview() {
  const steps = [
    { n: 1,  label: "Input",      dim: "28×28",  color: C.io,     enc: false, desc: "Raw pixels" },
    { n: 2,  label: "Encrypt",    dim: "784 ct",  color: C.crypto, enc: true,  desc: "BFV encode" },
    { n: 3,  label: "Conv1",      dim: "24×24",  color: C.conv,   enc: true,  desc: "5×5 filter" },
    { n: 4,  label: "x² Act",     dim: "24×24",  color: C.act,    enc: true,  desc: "Non-linear" },
    { n: 5,  label: "Pool1",      dim: "12×12",  color: C.act,    enc: true,  desc: "2×2 avg" },
    { n: 6,  label: "Conv2",      dim: "8×8",    color: C.conv,   enc: true,  desc: "5×5 filter" },
    { n: 7,  label: "Act+Pool2",  dim: "4×4",    color: C.act,    enc: true,  desc: "x² + avg" },
    { n: 8,  label: "FC",         dim: "16→10",  color: C.fc,     enc: true,  desc: "Dense layer" },
    { n: 9,  label: "Decrypt",    dim: "10 ints", color: C.crypto, enc: false, desc: "Reveal logits" },
    { n: 10, label: "Predict",    dim: "argmax",  color: C.io,     enc: false, desc: "Winner digit" },
  ];

  return (
    <div className="overflow-x-auto">
      <div className="flex items-start" style={{ minWidth: 820 }}>
        {steps.map((s, i) => (
          <div key={s.n} className="flex items-start">
            <div className="flex flex-col items-center text-center" style={{ width: 76 }}>
              <div className="w-5 h-5 rounded-full flex items-center justify-center text-white text-[9px] font-bold"
                style={{ background: s.color }}>{s.n}</div>
              <p className="text-[10px] font-semibold mt-1" style={{ color: s.color }}>{s.label}</p>
              <div className="rounded px-1.5 py-0.5 mt-0.5 text-[9px] font-mono"
                style={{ background: s.enc ? `${s.color}10` : "#fff", border: `1px ${s.enc ? "dashed" : "solid"} ${s.color}40`, color: s.color }}>
                {s.dim}
              </div>
              <p className="text-[9px] mt-0.5" style={{ color: C.faint }}>{s.desc}</p>
            </div>
            {i < steps.length - 1 && (
              <svg width={12} height={20} viewBox="0 0 12 20" className="mt-2 flex-shrink-0">
                <line x1={0} y1={10} x2={7} y2={10} stroke={C.faint} strokeWidth={1} />
                <polygon points="7,6 12,10 7,14" fill={C.faint} />
              </svg>
            )}
          </div>
        ))}
      </div>
      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-2 text-[10px]" style={{ color: C.muted }}>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0 border-t" style={{ borderColor: C.io }} /> Plaintext
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0 border-t border-dashed" style={{ borderColor: C.crypto }} /> Encrypted (steps 2–8)
        </span>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Section 2 — Interactive Convolution Demo (CS231n style)
   ════════════════════════════════════════════════════════════════════════
   A 7×7 input, 3×3 filter, stride 1, no padding → 5×5 output.
   The filter slides to a new position every 2 seconds.
   Shows connection lines, element-wise multiply, sum = output cell.          */

/* Example data — a "7"-ish pattern in a 7×7 grid */
const INPUT_7x7 = [
  [0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 2, 0],
  [0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 2, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
];

/* Simple edge-detect kernel */
const FILTER_3x3 = [
  [-1,  0,  1],
  [-1,  0,  1],
  [-1,  0,  1],
];

const OUT_SIZE = 5; // 7 - 3 + 1

/* Pre-compute all output values */
function computeConv(input, filter) {
  const out = [];
  for (let r = 0; r < OUT_SIZE; r++) {
    const row = [];
    for (let c = 0; c < OUT_SIZE; c++) {
      let sum = 0;
      for (let fr = 0; fr < 3; fr++)
        for (let fc = 0; fc < 3; fc++)
          sum += input[r + fr][c + fc] * filter[fr][fc];
      row.push(sum);
    }
    out.push(row);
  }
  return out;
}

const OUTPUT_5x5 = computeConv(INPUT_7x7, FILTER_3x3);

/* Total positions the filter visits */
const TOTAL_POS = OUT_SIZE * OUT_SIZE; // 25

function ConvolutionDemo() {
  const [pos, setPos] = useState(0);
  const [paused, setPaused] = useState(false);

  /* Auto-advance every 2 s */
  useEffect(() => {
    if (paused) return;
    const id = setInterval(() => setPos((p) => (p + 1) % TOTAL_POS), 2000);
    return () => clearInterval(id);
  }, [paused]);

  const togglePause = useCallback(() => setPaused((p) => !p), []);

  const outR = Math.floor(pos / OUT_SIZE);
  const outC = pos % OUT_SIZE;
  const inR = outR; // top-left row of the 3×3 patch
  const inC = outC;

  /* Compute element-wise products for the current position */
  const products = [];
  let sum = 0;
  for (let fr = 0; fr < 3; fr++) {
    for (let fc = 0; fc < 3; fc++) {
      const iv = INPUT_7x7[inR + fr][inC + fc];
      const fv = FILTER_3x3[fr][fc];
      const p = iv * fv;
      products.push({ iv, fv, p });
      sum += p;
    }
  }

  const S = 36; // cell size

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-2">
        <p className="text-xs font-semibold" style={{ color: C.conv }}>
          How Convolution Works — Interactive Demo
        </p>
        <button
          onClick={togglePause}
          className="text-[10px] px-2 py-0.5 rounded border"
          style={{ borderColor: C.conv, color: C.conv, background: paused ? `${C.conv}15` : "transparent" }}
        >
          {paused ? "Play" : "Pause"}
        </button>
        <span className="text-[10px]" style={{ color: C.muted }}>
          Position {pos + 1}/{TOTAL_POS}
        </span>
      </div>

      <div className="flex flex-wrap items-start gap-6">

        {/* ── Input 7×7 grid ── */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] font-semibold mb-1" style={{ color: C.text }}>
            Input (7×7)
          </p>
          <svg width={7 * S + 2} height={7 * S + 2} viewBox={`-1 -1 ${7 * S + 2} ${7 * S + 2}`}>
            {/* Grid cells */}
            {INPUT_7x7.map((row, r) =>
              row.map((v, c) => {
                const inPatch = r >= inR && r < inR + 3 && c >= inC && c < inC + 3;
                return (
                  <g key={`${r}-${c}`}>
                    <rect
                      x={c * S} y={r * S} width={S} height={S}
                      fill={inPatch ? `${C.conv}20` : v > 0 ? "#e8e8e8" : "#fff"}
                      stroke={inPatch ? C.conv : "#d0d0d0"}
                      strokeWidth={inPatch ? 2 : 0.5}
                    />
                    <text
                      x={c * S + S / 2} y={r * S + S / 2 + 1}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={13} fontFamily="monospace"
                      fontWeight={inPatch ? 700 : 400}
                      fill={inPatch ? C.conv : v > 0 ? "#333" : "#bbb"}
                    >
                      {v}
                    </text>
                  </g>
                );
              })
            )}
            {/* Sliding window border */}
            <rect
              x={inC * S} y={inR * S} width={3 * S} height={3 * S}
              fill="none" stroke={C.conv} strokeWidth={3} rx={3}
              style={{ transition: "x 0.4s ease, y 0.4s ease" }}
            />
          </svg>
        </div>

        {/* ── Multiplication column ── */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] font-semibold mb-1" style={{ color: C.conv }}>
            Filter (3×3) × Patch
          </p>
          <svg width={3 * S + 2} height={3 * S + 2} viewBox={`-1 -1 ${3 * S + 2} ${3 * S + 2}`}>
            {FILTER_3x3.map((row, r) =>
              row.map((fv, c) => {
                const idx = r * 3 + c;
                const { iv } = products[idx];
                const isNonZero = iv !== 0 && fv !== 0;
                return (
                  <g key={`${r}-${c}`}>
                    <rect
                      x={c * S} y={r * S} width={S} height={S}
                      fill={isNonZero ? "#fce4ec" : fv > 0 ? "#e8f5e9" : fv < 0 ? "#fbe9e7" : "#f5f5f5"}
                      stroke="#ccc" strokeWidth={0.5}
                    />
                    <text
                      x={c * S + S / 2} y={r * S + S / 2 + 1}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={13} fontFamily="monospace" fontWeight={600}
                      fill={fv > 0 ? "#2e7d32" : fv < 0 ? "#c62828" : "#999"}
                    >
                      {fv > 0 ? `+${fv}` : fv}
                    </text>
                  </g>
                );
              })
            )}
          </svg>
          {/* Calculation breakdown */}
          <div className="mt-2 rounded px-2 py-1.5" style={{ background: "#fff", border: `1px solid ${C.border}` }}>
            <p className="text-[9px] font-mono leading-relaxed" style={{ color: C.text }}>
              {products.map((p, i) => (
                <span key={i}>
                  <span style={{ color: C.conv }}>{p.iv}</span>
                  <span style={{ color: "#999" }}>×</span>
                  <span style={{ color: p.fv >= 0 ? "#2e7d32" : "#c62828" }}>{p.fv > 0 ? `+${p.fv}` : p.fv}</span>
                  {i < 8 ? <span style={{ color: "#ccc" }}> + </span> : null}
                </span>
              ))}
            </p>
            <p className="text-[11px] font-mono font-bold mt-0.5" style={{ color: C.conv }}>
              = {sum}
            </p>
          </div>
        </div>

        {/* ── Output 5×5 grid ── */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] font-semibold mb-1" style={{ color: C.text }}>
            Output (5×5)
          </p>
          <svg width={5 * S + 2} height={5 * S + 2} viewBox={`-1 -1 ${5 * S + 2} ${5 * S + 2}`}>
            {OUTPUT_5x5.map((row, r) =>
              row.map((v, c) => {
                const flatIdx = r * OUT_SIZE + c;
                const isCurrent = r === outR && c === outC;
                const isComputed = flatIdx <= pos;
                return (
                  <g key={`${r}-${c}`}>
                    <rect
                      x={c * S} y={r * S} width={S} height={S}
                      fill={isCurrent ? `${C.conv}30` : isComputed ? "#f0f0ff" : "#fafafa"}
                      stroke={isCurrent ? C.conv : "#d0d0d0"}
                      strokeWidth={isCurrent ? 2.5 : 0.5}
                    />
                    {isComputed && (
                      <text
                        x={c * S + S / 2} y={r * S + S / 2 + 1}
                        textAnchor="middle" dominantBaseline="middle"
                        fontSize={13} fontFamily="monospace"
                        fontWeight={isCurrent ? 800 : 500}
                        fill={isCurrent ? C.conv : "#555"}
                      >
                        {v}
                      </text>
                    )}
                    {!isComputed && (
                      <text
                        x={c * S + S / 2} y={r * S + S / 2 + 1}
                        textAnchor="middle" dominantBaseline="middle"
                        fontSize={11} fill="#ddd"
                      >
                        ?
                      </text>
                    )}
                  </g>
                );
              })
            )}
          </svg>
        </div>
      </div>

      {/* Explanation strip */}
      <p className="text-[10px] mt-3 leading-relaxed" style={{ color: C.muted }}>
        The <strong style={{ color: C.conv }}>3×3 filter</strong> slides across the input
        one cell at a time (stride=1). At each position it multiplies the overlapping patch
        element-wise, sums the 9 products, and writes the result into the output grid.
        In our encrypted CNN we use a <strong>5×5 filter</strong> on the real 28×28 image —
        same idea, just larger. The entire computation happens on <em>encrypted</em> numbers.
      </p>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Section 3 — Activation (x²) & Pooling mini-demos
   ════════════════════════════════════════════════════════════════════════ */

function ActivationPoolDemo() {
  const S = 34;
  const before = [
    [ 3, -1, 2, 4],
    [-2,  5, 0, 1],
    [ 1, -3, 4,-1],
    [ 2,  0,-2, 3],
  ];
  const afterAct = before.map(r => r.map(v => v * v));
  const afterPool = [
    [Math.round((afterAct[0][0] + afterAct[0][1] + afterAct[1][0] + afterAct[1][1]) / 4),
     Math.round((afterAct[0][2] + afterAct[0][3] + afterAct[1][2] + afterAct[1][3]) / 4)],
    [Math.round((afterAct[2][0] + afterAct[2][1] + afterAct[3][0] + afterAct[3][1]) / 4),
     Math.round((afterAct[2][2] + afterAct[2][3] + afterAct[3][2] + afterAct[3][3]) / 4)],
  ];

  function Grid({ data, color, size, label }) {
    const rows = data.length, cols = data[0].length;
    return (
      <div className="flex flex-col items-center">
        <p className="text-[10px] font-semibold mb-1" style={{ color }}>{label}</p>
        <svg width={cols * size + 2} height={rows * size + 2} viewBox={`-1 -1 ${cols * size + 2} ${rows * size + 2}`}>
          {data.map((row, r) =>
            row.map((v, c) => (
              <g key={`${r}-${c}`}>
                <rect x={c * size} y={r * size} width={size} height={size}
                  fill={v < 0 ? "#fbe9e7" : v > 0 ? `${color}10` : "#fafafa"}
                  stroke="#d0d0d0" strokeWidth={0.5} />
                <text x={c * size + size / 2} y={r * size + size / 2 + 1}
                  textAnchor="middle" dominantBaseline="middle"
                  fontSize={11} fontFamily="monospace" fontWeight={500}
                  fill={v < 0 ? "#c62828" : "#333"}>
                  {v}
                </text>
              </g>
            ))
          )}
        </svg>
      </div>
    );
  }

  function Arrow({ label }) {
    return (
      <div className="flex flex-col items-center mx-2">
        <svg width={36} height={20} viewBox="0 0 36 20">
          <line x1={2} y1={10} x2={28} y2={10} stroke={C.faint} strokeWidth={1.5} />
          <polygon points="28,6 36,10 28,14" fill={C.faint} />
        </svg>
        <p className="text-[9px] font-semibold -mt-0.5" style={{ color: C.act }}>{label}</p>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <p className="text-xs font-semibold mb-2" style={{ color: C.act }}>
        Activation (x²) and Average Pooling — Visual Example
      </p>
      <div className="flex flex-wrap items-center gap-0">
        <Grid data={before} color={C.conv} size={S} label="After Conv (4×4)" />
        <Arrow label="x²" />
        <Grid data={afterAct} color={C.act} size={S} label="After x² (4×4)" />
        <Arrow label="2×2 avg" />
        <Grid data={afterPool} color={C.act} size={42} label="After Pool (2×2)" />
      </div>
      <p className="text-[10px] mt-2 leading-relaxed" style={{ color: C.muted }}>
        <strong style={{ color: C.act }}>x² activation</strong>: every value is squared
        (e.g. −3 → 9). This adds non-linearity — the network can learn curves, not just
        straight lines. Normal CNNs use ReLU, but ReLU is impossible on encrypted data so
        we use a polynomial (x²) instead.{" "}
        <strong style={{ color: C.act }}>Average pooling</strong>: each 2×2 block is
        averaged into one number, halving the grid size and making the features more
        robust to small shifts.
      </p>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Section 4 — FC Layer + Decrypt explanation
   ════════════════════════════════════════════════════════════════════════ */

function FcDecryptDemo() {
  return (
    <div className="mt-4">
      <p className="text-xs font-semibold mb-1" style={{ color: C.fc }}>
        Fully Connected Layer + Decryption
      </p>
      <div className="flex items-center gap-4">
        {/* Flatten */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] mb-1" style={{ color: C.muted }}>Flatten 4×4</p>
          <svg width={20} height={100} viewBox="0 0 20 100">
            {Array.from({ length: 16 }, (_, i) => (
              <rect key={i} x={2} y={i * 6 + 1} width={16} height={5} rx={1}
                fill={C.fc} fillOpacity={0.15 + (i % 3) * 0.1} stroke={C.fc} strokeWidth={0.3} />
            ))}
          </svg>
          <p className="text-[9px] font-mono mt-0.5" style={{ color: C.muted }}>16 values</p>
        </div>

        {/* Arrow */}
        <svg width={36} height={20} viewBox="0 0 36 20">
          <line x1={2} y1={10} x2={28} y2={10} stroke={C.faint} strokeWidth={1.5} />
          <polygon points="28,6 36,10 28,14" fill={C.faint} />
        </svg>

        {/* Weight matrix hint */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] mb-1" style={{ color: C.muted }}>×Weights</p>
          <div className="rounded px-2 py-1" style={{ background: `${C.fc}08`, border: `1px dashed ${C.fc}40` }}>
            <p className="text-[10px] font-mono" style={{ color: C.fc }}>16×10 matrix</p>
          </div>
        </div>

        {/* Arrow */}
        <svg width={36} height={20} viewBox="0 0 36 20">
          <line x1={2} y1={10} x2={28} y2={10} stroke={C.faint} strokeWidth={1.5} />
          <polygon points="28,6 36,10 28,14" fill={C.faint} />
        </svg>

        {/* 10 logits */}
        <div className="flex flex-col items-center">
          <p className="text-[10px] mb-1" style={{ color: C.muted }}>10 logits</p>
          <svg width={80} height={60} viewBox="0 0 80 60">
            {[2, 1, 3, 1, 2, 1, 0, 9, 1, 2].map((v, i) => {
              const barH = (v / 9) * 42;
              const isMax = i === 7;
              return (
                <g key={i}>
                  <rect x={i * 8} y={48 - barH} width={6} height={barH} rx={1}
                    fill={isMax ? C.io : "#ddd"} />
                  <text x={i * 8 + 3} y={56} textAnchor="middle"
                    fontSize={6} fill={isMax ? C.io : C.faint} fontWeight={isMax ? 700 : 400}>
                    {i}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="text-[9px] font-mono" style={{ color: C.muted }}>argmax → <strong style={{ color: C.io }}>7</strong></p>
        </div>
      </div>
      <p className="text-[10px] mt-2 leading-relaxed" style={{ color: C.muted }}>
        After the last pooling layer, the 4×4 grid is flattened into a vector of 16
        numbers. A <strong style={{ color: C.fc }}>fully connected layer</strong> multiplies
        this vector by a 16×10 weight matrix to produce 10 scores (logits) — one per
        digit class (0–9). Then the ciphertext is <strong style={{ color: C.crypto }}>decrypted</strong> with
        the secret key, and <strong>argmax</strong> picks the digit with the highest score.
      </p>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════
   Main export
   ════════════════════════════════════════════════════════════════════════ */

export default function ArchitectureDiagram() {
  return (
    <div className="rounded-lg p-5 mt-4" style={{ background: C.bg, border: `1px solid ${C.border}` }}>
      {/* Title */}
      <p className="text-sm font-medium mb-1" style={{ color: C.text }}>
        Step-by-step: What happens to your drawn digit
      </p>
      <p className="text-xs mb-4" style={{ color: C.muted }}>
        Follow the data as it transforms from raw pixels through encrypted neural
        network layers to a final prediction. Each section below explains one part of the pipeline with real numbers.
      </p>

      {/* 1. Pipeline overview strip */}
      <PipelineOverview />

      {/* Divider */}
      <div className="my-4 h-px" style={{ background: C.border }} />

      {/* 2. Interactive convolution demo */}
      <ConvolutionDemo />

      {/* Divider */}
      <div className="my-4 h-px" style={{ background: C.border }} />

      {/* 3. Activation + pooling demo */}
      <ActivationPoolDemo />

      {/* Divider */}
      <div className="my-4 h-px" style={{ background: C.border }} />

      {/* 4. FC + decrypt demo */}
      <FcDecryptDemo />

      {/* Footer note */}
      <div className="mt-4 pt-3" style={{ borderTop: `1px solid ${C.border}` }}>
        <p className="text-[11px] text-center" style={{ color: C.faint }}>
          Steps 2–8 all run on encrypted data (ciphertext). The server performs convolutions,
          activations, pooling, and matrix multiplication without ever seeing the original pixel values.
          Only the final 10 logit numbers are decrypted — your image stays private throughout.
        </p>
      </div>
    </div>
  );
}
