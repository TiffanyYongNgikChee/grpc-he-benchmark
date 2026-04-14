import { useState } from "react";

// ─── Library colour palette ────────────────────────────────────────────────────
const LIB = {
  SEAL:    { bg: "#e8f0fe", bar: "#4285f4", dark: "#1a56c4", scheme: "BFV" },
  HELib:   { bg: "#f3e8ff", bar: "#9333ea", dark: "#6b21a8", scheme: "BGV" },
  OpenFHE: { bg: "#e0fdf4", bar: "#10b981", dark: "#047857", scheme: "BFV" },
};

// ─── Operation metadata — with real HE explanations ──────────────────────────
const OPS = [
  {
    key:   "keyGenTimeMs",
    short: "Key Generation",
    emoji: "🔑",
    color: { bg: "#fff7ed", accent: "#f97316", light: "#fed7aa", dark: "#9a3412" },
    what:  "Generates the public/secret key pair and relinearisation keys.",
    how:   "Samples random polynomials in ring Rq, computes public key as (−a·s + e, a). Relinearisation keys encode s² so that ciphertexts can be kept to degree 1 after multiplication.",
    why:   "Larger polynomial degree n → bigger key matrices → slower sampling. Done once per session — cost amortises over many operations.",
    param: "n = 4096 · |Rq| = 60-bit",
  },
  {
    key:   "encryptionTimeMs",
    short: "Encrypt Input",
    emoji: "🔒",
    color: { bg: "#eff6ff", accent: "#3b82f6", light: "#bfdbfe", dark: "#1e3a8a" },
    what:  "Encodes 784 MNIST pixel values into a single BFV/BGV ciphertext.",
    how:   "Pixels are packed into plaintext slots via NTT (Number Theoretic Transform), then masked: ct = (pk₀·u + e₁ + Δ·m,  pk₁·u + e₂) where Δ = ⌊q/t⌋ lifts the message into the high-order bits of q.",
    why:   "BFV/BGV batch all 4096 slots — 784 pixels fit in one ciphertext. NTT is O(n log n) so encryption is fast but dominated by random polynomial sampling.",
    param: "Δ = ⌊q/t⌋ · t = 65537",
  },
  {
    key:   "additionTimeMs",
    short: "HE Addition",
    emoji: "➕",
    color: { bg: "#f0fdf4", accent: "#22c55e", light: "#bbf7d0", dark: "#14532d" },
    what:  "Adds two ciphertexts — equivalent to adding the underlying plaintexts.",
    how:   "Simple component-wise polynomial addition mod q: (c₀ + c₀′, c₁ + c₁′) mod q. No noise growth beyond additive — error stays small regardless of how many additions you chain.",
    why:   "Pure polynomial addition is O(n). This is the cheapest HE operation and is used for bias addition in FC layers (bias vector + encrypted activations).",
    param: "cost = O(n) · no relin needed",
  },
  {
    key:   "multiplicationTimeMs",
    short: "HE Multiply",
    emoji: "✖️",
    color: { bg: "#fdf4ff", accent: "#a855f7", light: "#e9d5ff", dark: "#581c87" },
    what:  "Multiplies two ciphertexts — the most expensive HE primitive.",
    how:   "Computes tensor product of the two degree-1 ciphertexts → degree-2 ciphertext, then relinearises using key-switching keys to bring it back to degree 1. Requires NTT-domain multiplication + modulus switching to keep noise bounded.",
    why:   "Key-switching involves multiplying against large gadget matrices. Noise grows quadratically with each multiplication, limiting the number of sequential multiplications (circuit depth). Used for weight multiply in Conv/FC.",
    param: "cost = O(n log n) · relin keys",
  },
  {
    key:   "decryptionTimeMs",
    short: "Decrypt Output",
    emoji: "🔓",
    color: { bg: "#fff1f2", accent: "#f43f5e", light: "#fecdd3", dark: "#881337" },
    what:  "Recovers the 10 prediction logits from the output ciphertext.",
    how:   "Inner product with secret key polynomial s: m̃ = c₀ + c₁·s mod q, then scale back: m = ⌊m̃ · t/q⌉ mod t. Argmax over the 10 output slots gives the predicted digit 0–9.",
    why:   "Single polynomial evaluation — very fast. The secret key never leaves the client in a real privacy-preserving deployment.",
    param: "output: 10 logits → digit 0–9",
  },
];

// ─── SVG gradient defs ────────────────────────────────────────────────────────
function SvgDefs() {
  return (
    <defs>
      {Object.entries(LIB).map(([name, c]) => (
        <linearGradient key={name} id={`grad-${name}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={c.bar} stopOpacity="1" />
          <stop offset="100%" stopColor={c.dark} stopOpacity="0.85" />
        </linearGradient>
      ))}
    </defs>
  );
}

// ─── Radar / spider chart  ────────────────────────────────────────────────────
function RadarChart({ results }) {
  const W = 220, H = 200, CX = 110, CY = 100, R = 75;
  const n = OPS.length;
  const angles = OPS.map((_, i) => (i / n) * 2 * Math.PI - Math.PI / 2);
  const pt = (angle, radius) => [CX + radius * Math.cos(angle), CY + radius * Math.sin(angle)];
  const opMaxes = OPS.map(op => Math.max(...results.map(r => r[op.key] || 0), 1));
  const score = (lib, opIdx) => 1 - (lib[OPS[opIdx].key] || 0) / opMaxes[opIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <SvgDefs />
      {[0.25, 0.5, 0.75, 1].map(level => (
        <polygon key={level}
          points={angles.map(a => pt(a, R * level).join(",")).join(" ")}
          fill="none" stroke="#e2e8f0" strokeWidth={0.7} />
      ))}
      {angles.map((a, i) => {
        const [x, y] = pt(a, R);
        return <line key={i} x1={CX} y1={CY} x2={x} y2={y} stroke="#e2e8f0" strokeWidth={0.7} />;
      })}
      {results.map(lib => {
        const lc = LIB[lib.library] || LIB.SEAL;
        const pts = angles.map((a, i) => pt(a, R * Math.max(0.05, score(lib, i))));
        return (
          <g key={lib.library}>
            <polygon points={pts.map(p => p.join(",")).join(" ")}
              fill={lc.bar} fillOpacity={0.15} stroke={lc.bar} strokeWidth={1.8} />
            {pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={3} fill={lc.bar} />)}
          </g>
        );
      })}
      {angles.map((a, i) => {
        const [x, y] = pt(a, R + 13);
        return (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fontSize={12} fontFamily="sans-serif">{OPS[i].emoji}</text>
        );
      })}
      <text x={CX} y={CY} textAnchor="middle" dominantBaseline="middle"
        fontSize={7} fill="#94a3b8" fontFamily="sans-serif">speed</text>
    </svg>
  );
}

// ─── Total time horizontal bars ───────────────────────────────────────────────
function TotalTimeBars({ results }) {
  const maxTotal = Math.max(...results.map(r => r.totalTimeMs || 0), 1);
  const sorted = [...results].sort((a, b) => a.totalTimeMs - b.totalTimeMs);
  return (
    <div className="flex flex-col gap-2.5">
      {sorted.map((lib, i) => {
        const lc = LIB[lib.library] || LIB.SEAL;
        const pct = (lib.totalTimeMs / maxTotal) * 100;
        return (
          <div key={lib.library}>
            <div className="flex justify-between items-center mb-1">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full inline-block shrink-0" style={{ background: lc.bar }} />
                <span className="text-xs font-bold" style={{ color: lc.dark }}>{lib.library}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded font-mono" style={{ background: lc.bg, color: lc.dark }}>{lc.scheme}</span>
                {i === 0 && <span className="text-[8px] px-1.5 py-0.5 rounded font-bold text-white" style={{ background: lc.bar }}>⚡ FASTEST</span>}
              </div>
              <span className="text-xs font-mono font-bold" style={{ color: lc.dark }}>{lib.totalTimeMs.toFixed(1)} ms</span>
            </div>
            <div className="h-5 rounded-full overflow-hidden" style={{ background: "#f1f5f9" }}>
              <div className="h-5 rounded-full relative flex items-center"
                style={{ width: `${pct}%`, background: `linear-gradient(90deg,${lc.bar},${lc.dark})` }}>
                <div className="absolute inset-0 rounded-full" style={{ background: "linear-gradient(180deg,rgba(255,255,255,0.25) 0%,transparent 60%)" }} />
                {pct > 20 && <span className="text-[8px] font-mono font-bold text-white ml-auto mr-2 relative z-10">{lib.totalTimeMs.toFixed(0)}ms</span>}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function LibraryComparison({ data, loading, error, onRun }) {
  const [activeOp, setActiveOp] = useState(null);

  const Banner = () => (
    <div className="rounded-xl px-4 py-3 mb-5 flex items-start gap-3"
      style={{ background: "linear-gradient(135deg,#1e3a5f,#1a1a2e)", border: "1px solid #334155" }}>
      <div className="mt-0.5 text-xl shrink-0">🧮</div>
      <div>
        <p className="text-xs font-bold mb-0.5 text-white">MNIST Encrypted Inference — HE Primitive Benchmarks</p>
        <p className="text-[11px] leading-relaxed" style={{ color: "#94a3b8" }}>
          Each library runs <strong className="text-white">5 HE operations</strong> repeated <strong className="text-white">10×</strong> and averaged.
          Parameters: <span className="font-mono text-blue-300">n = 4096</span>, <span className="font-mono text-blue-300">128-bit</span> security.
          SEAL &amp; OpenFHE → <span className="font-mono text-green-300">BFV</span>; HELib → <span className="font-mono text-purple-300">BGV</span>.
        </p>
      </div>
    </div>
  );

  if (!data && !loading && !error) {
    return (
      <div className="py-2">
        <Banner />
        <div className="text-center py-8">
          <div className="inline-flex flex-col items-center gap-4">
            <div className="flex gap-3">
              {Object.entries(LIB).map(([name, c]) => (
                <div key={name} className="rounded-xl px-5 py-3 text-center"
                  style={{ background: c.bg, border: `2px solid ${c.bar}33` }}>
                  <div className="text-sm font-black mb-0.5" style={{ color: c.dark }}>{name}</div>
                  <div className="text-[9px] font-mono" style={{ color: c.bar }}>{c.scheme}</div>
                </div>
              ))}
            </div>
            <button onClick={() => onRun(null)}
              className="px-7 py-2.5 rounded-xl text-sm font-black text-white shadow-lg hover:shadow-xl transition-all hover:scale-[1.03] active:scale-[0.97]"
              style={{ background: "linear-gradient(135deg,#4285f4,#9333ea)" }}>
              ▶ Run Library Comparison
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-14">
        <div className="relative w-14 h-14 mb-4">
          <svg className="animate-spin absolute inset-0 w-14 h-14" viewBox="0 0 56 56">
            <circle cx="28" cy="28" r="24" fill="none" stroke="#e2e8f0" strokeWidth="4" />
            <circle cx="28" cy="28" r="24" fill="none" stroke="url(#grad-SEAL)" strokeWidth="4"
              strokeDasharray="40 120" strokeLinecap="round" />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center text-2xl">🔐</span>
        </div>
        <p className="text-sm font-bold" style={{ color: "#1e3a5f" }}>Running benchmarks…</p>
        <p className="text-[10px] mt-1" style={{ color: "#94a3b8" }}>SEAL → HELib → OpenFHE (sequential, ~30–90 s total)</p>
        <svg width="0" height="0"><SvgDefs /></svg>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="inline-block rounded-xl p-4" style={{ background: "#fef2f2", border: "1px solid #fca5a5" }}>
          <p className="text-sm font-bold mb-1" style={{ color: "#dc2626" }}>Benchmark Failed</p>
          <p className="text-xs" style={{ color: "#b91c1c" }}>{error}</p>
        </div>
        <div className="mt-4">
          <button onClick={() => onRun(null)} className="px-5 py-2 rounded-xl text-xs font-bold text-white"
            style={{ background: "#1e3a5f" }}>Retry</button>
        </div>
      </div>
    );
  }

  const results = data?.results || [];
  if (!results.length) return <div className="text-center py-8 text-xs text-gray-400">No results returned.</div>;

  const sortedResults = [...results].sort((a, b) => a.totalTimeMs - b.totalTimeMs);
  const fastest = sortedResults[0];
  const slowest = sortedResults[sortedResults.length - 1];
  const speedup = (slowest.totalTimeMs / fastest.totalTimeMs).toFixed(1);

  return (
    <div>
      <Banner />

      {/* ══ Section A: Total time + radar ══ */}
      <div className="grid grid-cols-3 gap-4 mb-5">
        <div className="col-span-2 rounded-xl p-4" style={{ background: "#fff", border: "1.5px solid #e2e8f0" }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: "#94a3b8" }}>
            ⏱ Total Pipeline Time — 5 operations summed
          </p>
          <TotalTimeBars results={results} />
          {results.length >= 2 && (() => {
            const lc = LIB[fastest.library] || LIB.SEAL;
            return (
              <div className="mt-3 rounded-lg px-3 py-2 flex items-center gap-2"
                style={{ background: lc.bg, border: `1px solid ${lc.bar}44` }}>
                <span>⚡</span>
                <p className="text-[10px]" style={{ color: lc.dark }}>
                  <strong>{fastest.library}</strong> is <strong>{speedup}×</strong> faster than{" "}
                  <strong>{slowest.library}</strong> overall
                </p>
              </div>
            );
          })()}
        </div>
        <div className="rounded-xl p-3" style={{ background: "#fff", border: "1.5px solid #e2e8f0" }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: "#94a3b8" }}>Speed Radar</p>
          <p className="text-[9px] mb-2" style={{ color: "#cbd5e1" }}>Larger area = faster per operation</p>
          <RadarChart results={results} />
          <div className="flex flex-col gap-1 mt-1">
            {results.map(r => {
              const lc = LIB[r.library] || LIB.SEAL;
              return (
                <div key={r.library} className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: lc.bar }} />
                  <span className="text-[9px] font-bold" style={{ color: lc.dark }}>{r.library}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ══ Section B: Operation breakdown — clickable rows ══ */}
      <div className="rounded-xl overflow-hidden mb-4" style={{ background: "#fff", border: "1.5px solid #e2e8f0" }}>
        <div className="px-4 pt-3 pb-2 border-b border-slate-100 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#94a3b8" }}>
              Operation Breakdown
            </span>
            <span className="text-xs font-semibold ml-2" style={{ color: "#334155" }}>
              Per-primitive timing — click any row for details
            </span>
          </div>
          <div className="flex gap-3">
            {results.map(r => {
              const lc = LIB[r.library] || LIB.SEAL;
              return (
                <span key={r.library} className="flex items-center gap-1 text-[9px] font-bold" style={{ color: lc.dark }}>
                  <span className="inline-block w-3 h-2 rounded-sm"
                    style={{ background: `linear-gradient(90deg,${lc.bar},${lc.dark})` }} />
                  {r.library}
                </span>
              );
            })}
          </div>
        </div>

        <div className="px-2 py-2">
          {OPS.map(op => {
            const isActive = activeOp === op.key;
            const maxVal = Math.max(...results.map(r => r[op.key] || 0), 1);
            const opSorted = [...results].sort((a, b) => (a[op.key] || 0) - (b[op.key] || 0));

            return (
              <div key={op.key}
                onClick={() => setActiveOp(isActive ? null : op.key)}
                className="rounded-xl mb-1.5 px-3 py-2.5 transition-all cursor-pointer"
                style={{
                  background: isActive
                    ? `linear-gradient(135deg,${op.color.bg},#ffffff)`
                    : "transparent",
                  border: `1.5px solid ${isActive ? op.color.light : "transparent"}`,
                }}>

                {/* Op header row */}
                <div className="flex items-center gap-2 mb-2">
                  <span style={{ fontSize: 16 }}>{op.emoji}</span>
                  <div className="flex-1 min-w-0">
                    <span className="text-[11px] font-black" style={{ color: op.color.dark }}>{op.short}</span>
                    <span className="ml-2 text-[9px] font-mono px-1.5 py-0.5 rounded"
                      style={{ background: op.color.light + "88", color: op.color.dark }}>{op.param}</span>
                  </div>
                  <span className="text-[9px] shrink-0" style={{ color: "#94a3b8" }}>
                    {isActive ? "▲ hide" : "▼ explain"}
                  </span>
                </div>

                {/* Bars */}
                <div className="flex flex-col gap-1.5">
                  {results.map(lib => {
                    const lc = LIB[lib.library] || LIB.SEAL;
                    const val = lib[op.key] || 0;
                    const pct = Math.max(2, (val / maxVal) * 100);
                    return (
                      <div key={lib.library} className="flex items-center gap-2">
                        <span className="text-[8px] w-12 text-right font-bold shrink-0" style={{ color: lc.dark }}>{lib.library}</span>
                        <div className="flex-1 h-4 rounded-full overflow-hidden relative" style={{ background: "#f1f5f9" }}>
                          <div className="h-4 rounded-full relative"
                            style={{ width: `${pct}%`, background: `linear-gradient(90deg,${lc.bar},${lc.dark})` }}>
                            <div className="absolute inset-0 rounded-full"
                              style={{ background: "linear-gradient(180deg,rgba(255,255,255,0.3) 0%,transparent 55%)" }} />
                          </div>
                        </div>
                        <span className="text-[9px] font-mono font-bold w-14 shrink-0 text-right"
                          style={{ color: lc.dark }}>{val.toFixed(1)} ms</span>
                      </div>
                    );
                  })}
                </div>

                {/* ── Expanded detail ── */}
                {isActive && (
                  <div className="mt-3 pt-3 border-t" style={{ borderColor: op.color.light }}>
                    {/* What / How / Why */}
                    <div className="grid grid-cols-3 gap-2 mb-3">
                      {[
                        { icon: "📌", title: "What it does", body: op.what },
                        { icon: "⚙️", title: "How (math)", body: op.how },
                        { icon: "💡", title: "Why it costs time", body: op.why },
                      ].map(card => (
                        <div key={card.title} className="rounded-xl p-3"
                          style={{ background: "rgba(255,255,255,0.9)", border: `1px solid ${op.color.light}` }}>
                          <p className="text-[8px] font-black uppercase tracking-wider mb-1.5"
                            style={{ color: op.color.accent }}>{card.icon} {card.title}</p>
                          <p className="text-[10px] leading-relaxed" style={{ color: "#475569" }}>{card.body}</p>
                        </div>
                      ))}
                    </div>
                    {/* Per-library timing pills ranked */}
                    <div className="flex gap-2">
                      {opSorted.map((lib, idx) => {
                        const lc = LIB[lib.library] || LIB.SEAL;
                        const val = lib[op.key] || 0;
                        const pctOfMax = (val / maxVal) * 100;
                        return (
                          <div key={lib.library} className="flex-1 rounded-xl p-3 text-center"
                            style={{
                              background: lc.bg,
                              border: `2px solid ${idx === 0 ? lc.bar : lc.bar + "33"}`,
                            }}>
                            {idx === 0 && <div className="text-[8px] font-black mb-1 text-white px-1.5 py-0.5 rounded-full inline-block" style={{ background: lc.bar }}>⚡ fastest</div>}
                            <div className="text-[10px] font-black" style={{ color: lc.dark }}>{lib.library}</div>
                            <div className="text-xl font-black font-mono mt-0.5 leading-none" style={{ color: lc.dark }}>{val.toFixed(1)}</div>
                            <div className="text-[9px] font-bold" style={{ color: lc.bar }}>ms</div>
                            {/* Mini bar */}
                            <div className="mt-1.5 h-1.5 rounded-full" style={{ background: "#e2e8f0" }}>
                              <div className="h-1.5 rounded-full" style={{ width: `${pctOfMax}%`, background: lc.bar }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <p className="text-[9px] px-4 pb-3 italic" style={{ color: "#cbd5e1" }}>
          Fig. 1 — HE primitive latencies at n = 4096, t = 65537, 128-bit security, averaged over 10 repetitions. SEAL &amp; OpenFHE: BFV. HELib: BGV.
        </p>
      </div>

      {/* ══ Section C: Per-library summary cards ══ */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        {sortedResults.map((lib, rank) => {
          const lc = LIB[lib.library] || LIB.SEAL;
          return (
            <div key={lib.library} className="rounded-xl p-4 relative overflow-hidden"
              style={{
                background: `linear-gradient(145deg,${lc.bg} 0%,#ffffff 100%)`,
                border: `2px solid ${rank === 0 ? lc.bar : lc.bar + "33"}`,
              }}>
              {/* Decorative blob */}
              <div className="absolute -top-5 -right-5 w-20 h-20 rounded-full opacity-[0.08]"
                style={{ background: lc.bar }} />
              {/* Rank badge */}
              <span className="absolute top-2.5 right-2.5 text-[8px] font-black px-1.5 py-0.5 rounded-full"
                style={{ background: rank === 0 ? lc.bar : "#f1f5f9", color: rank === 0 ? "#fff" : lc.dark }}>
                {rank === 0 ? "⚡ #1" : `#${rank + 1}`}
              </span>
              <p className="text-xs font-black" style={{ color: lc.dark }}>{lib.library}</p>
              <p className="text-[9px] font-mono mb-2" style={{ color: lc.bar }}>
                {lib.schemeInfo || `${lc.scheme} · n=4096`}
              </p>
              <p className="text-2xl font-black font-mono leading-none" style={{ color: lc.dark }}>
                {lib.totalTimeMs.toFixed(1)}
                <span className="text-[10px] font-semibold ml-1" style={{ color: lc.bar }}>ms</span>
              </p>
              <div className="mt-3 space-y-1.5">
                {OPS.map(op => (
                  <div key={op.key} className="flex items-center justify-between">
                    <span className="text-[9px]" style={{ color: "#94a3b8" }}>{op.emoji} {op.short}</span>
                    <span className="text-[9px] font-mono font-bold" style={{ color: lc.dark }}>
                      {(lib[op.key] || 0).toFixed(1)} ms
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[9px] mt-2.5 font-semibold" style={{ color: lib.success ? "#059669" : "#dc2626" }}>
                {lib.success ? "✓ All operations passed" : lib.errorMessage || "Failed"}
              </p>
            </div>
          );
        })}
      </div>

      <div className="text-center">
        <button onClick={() => onRun(null)}
          className="px-5 py-2 rounded-xl text-xs font-bold border-2 hover:bg-slate-50 transition-colors"
          style={{ borderColor: "#e2e8f0", color: "#64748b" }}>
          ↺ Run Again
        </button>
      </div>
    </div>
  );
}

