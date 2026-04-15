import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import CnnPipeline, { LAYERS } from "./CnnPipeline";
import MiniCanvas from "./MiniCanvas";
import OutputPanel from "./OutputPanel";
import MetricsStrip from "./MetricsStrip";
import LibraryComparison from "./LibraryComparison";
import LiveStatusFeed from "./LiveStatusFeed";

import MnistBatchBenchmark from "./MnistBatchBenchmark";
import ParameterComparison from "./ParameterComparison";
import BenchmarkResultsDashboard from "./BenchmarkResultsDashboard";
import ArchitectureDiagram from "./ArchitectureDiagram";
import NeuralHero from "./NeuralHero";
import useInferenceProgress from "./useInferenceProgress";
import { checkHealth, predictDigit, runComparisonBenchmark } from "../api/client";

/**
 * Workbench — Single-page encrypted ML inference & benchmarking interface.
 *
 * Visual hierarchy (top → bottom):
 *   1. Header — hero text + tagline
 *   2. Controls bar — play, reset, epoch, dropdowns (scheme, model, scale)
 *   3. Main area — INPUT | CNN PIPELINE (vertical groups) | OUTPUT
 *   4. Bottom metrics — timing stacked bar
 *   5. Library comparison — SEAL vs HElib vs OpenFHE
 *   6. Info section — "What is HE?" + "About this project"
 */
export default function Workbench() {
  /* ─── State ─── */
  const [pixels, setPixels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [healthy, setHealthy] = useState(null);

  /* Fixed inference parameters — only degree 2 + 128-bit work on this server */
  const securityLevel = 0;    // 128-bit
  const activationDegree = 2; // x²

  /* Animation — layer-by-layer progress */
  const progress = useInferenceProgress();
  const [activeStep, setActiveStep] = useState(-1);
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const [runCount, setRunCount] = useState(0);
  const animTimers = useRef([]);

  /* Run history — array of { log, result, runIndex, timestamp } */
  const [runHistory, setRunHistory] = useState([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyIndex, setHistoryIndex] = useState(0); // which run to view

  /* Library comparison */
  const [compData, setCompData] = useState(null);
  const [compLoading, setCompLoading] = useState(false);
  const [compError, setCompError] = useState(null);
  const compAbortRef = useRef(null);

  /* Health check */
  useEffect(() => {
    const check = () => checkHealth().then(setHealthy);
    check();
    const id = setInterval(check, 10_000);
    return () => clearInterval(id);
  }, []);

  /* ─── Run inference ─── */
  const handleRun = useCallback(async () => {
    if (!pixels || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveStep(0);

    try {
      /* Try real SSE streaming first — returns the prediction result via the stream */
      const requestBody = { pixels: pixels, scaleFactor: 1000, securityLevel, activationDegree };
      const response = await progress.startSSE(requestBody);

      setResult(response);
      setRunCount((c) => c + 1);

      /* Post-result animation: replay layers with real timing ratios */
      let elapsed = 0;
      const timingKeys = LAYERS.map((l) => l.key);
      animTimers.current.forEach(clearTimeout);
      animTimers.current = [];

      for (let step = 1; step <= LAYERS.length; step++) {
        const key = timingKeys[step - 1];
        const ms = key && response[key] ? response[key] : 60;
        elapsed += Math.min(ms * 1.5, 500);
        const t = setTimeout(() => setActiveStep(step), elapsed);
        animTimers.current.push(t);
      }
    } catch (sseErr) {
      /* SSE failed — fall back to simulated animation + regular predict */
      console.warn("SSE streaming unavailable, falling back to simulated:", sseErr.message);
      progress.reset();
      progress.startSimulated();

      try {
        const response = await predictDigit(pixels, 1000, { securityLevel, activationDegree });
        setResult(response);
        setRunCount((c) => c + 1);
        progress.markAllDone();

        let elapsed = 0;
        const timingKeys = LAYERS.map((l) => l.key);
        animTimers.current.forEach(clearTimeout);
        animTimers.current = [];

        for (let step = 1; step <= LAYERS.length; step++) {
          const key = timingKeys[step - 1];
          const ms = key && response[key] ? response[key] : 60;
          elapsed += Math.min(ms * 1.5, 500);
          const t = setTimeout(() => setActiveStep(step), elapsed);
          animTimers.current.push(t);
        }
      } catch (err) {
        setError(err.message);
        setActiveStep(-1);
        progress.reset();
      }
    } finally {
      setLoading(false);
    }
  }, [pixels, loading, progress, securityLevel, activationDegree]);

  const handleReset = () => {
    setResult(null);
    setError(null);
    setActiveStep(-1);
    progress.reset();
  };

  /* ─── Run library comparison ─── */
  const handleComparison = useCallback(async (testValues) => {
    if (compLoading) return;
    setCompLoading(true);
    setCompError(null);
    const ac = new AbortController();
    compAbortRef.current = ac;
    try {
      const data = await runComparisonBenchmark(10, testValues, ac.signal);
      setCompData(data);
    } catch (err) {
      if (err.name === "AbortError") {
        setCompError("Benchmark cancelled.");
      } else if (err.name === "TimeoutError") {
        setCompError("Benchmark timed out after 3 minutes. The server may be overloaded — try again or reduce operations.");
      } else {
        setCompError(err.message);
      }
    } finally {
      compAbortRef.current = null;
      setCompLoading(false);
    }
  }, [compLoading]);

  const handleCompCancel = useCallback(() => {
    if (compAbortRef.current) compAbortRef.current.abort();
  }, []);

  /* Save completed run log to history */
  const handleLogComplete = useCallback((log, runResult) => {
    setRunHistory(prev => {
      const entry = {
        runIndex: prev.length + 1,
        timestamp: new Date().toLocaleTimeString(),
        log,
        result: runResult,
      };
      return [...prev, entry];
    });
  }, []);

  useEffect(() => () => animTimers.current.forEach(clearTimeout), []);

  /* Tooltip */
  const tl = hoveredLayer !== null ? LAYERS[hoveredLayer] : null;
  const tTime = tl && tl.key && result ? result[tl.key] : null;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#f7f7f7" }}>

      {/* ═══════════ HERO — NEURAL NETWORK ANIMATION ═══════════ */}
      <NeuralHero
        hasHistory={runHistory.length > 0}
        onOpenHistory={() => { setHistoryIndex(runHistory.length - 1); setHistoryOpen(true); }}
      />

      {/* ═══════════ STICKY NAV BAR ═══════════ */}
      <div
        className="sticky top-0 z-30 border-b backdrop-blur-xl"
        style={{
          background: "rgba(3,7,18,0.88)",
          borderColor: "rgba(99,102,241,0.08)",
        }}
      >
        <div className="max-w-[1440px] mx-auto px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#6366f1", boxShadow: "0 0 8px #6366f1" }} />
            <span className="text-[10px] font-semibold tracking-[0.2em] text-white/50 uppercase">
              HE Benchmark
            </span>
          </div>
          <div className="flex items-center gap-4 text-[10px] text-white/30">
            <a href="#workbench" className="hover:text-white/80 transition-colors">Workbench</a>
            <a href="#benchmarks" className="hover:text-white/80 transition-colors">Benchmarks</a>
            <a href="#results" className="hover:text-white/80 transition-colors">Results</a>
            <a href="#info" className="hover:text-white/80 transition-colors">About</a>
            <div className="flex items-center gap-1.5 ml-2">
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: healthy === null ? "#555" : healthy ? "#06d6a0" : "#e03e52",
                  boxShadow: healthy ? "0 0 6px rgba(6,214,160,0.5)" : "none",
                }}
              />
              <span className="text-[9px]">{healthy === null ? "…" : healthy ? "Online" : "Offline"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════ HABBO-STYLE WORKBENCH ═══════════ */}
      <div
        id="workbench"
        style={{
          background: "#5a5a5a",
          borderTop: "3px solid #888",
          borderBottom: "3px solid #333",
          padding: "10px 12px 14px",
        }}
      >
        {/* ── Top toolbar bar — gold Habbo header style ── */}
        <div style={{
          background: "linear-gradient(180deg, #d4a800 0%, #b08a00 100%)",
          border: "2px solid #7a5e00",
          borderRadius: "4px 4px 0 0",
          padding: "5px 12px",
          display: "flex", alignItems: "center", gap: 8,
          marginBottom: 2,
        }}>
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.62rem", letterSpacing: "0.18em",
            color: "#1a0e00", textTransform: "uppercase",
            textShadow: "0 1px 0 rgba(255,220,80,0.4)",
          }}>
            ◈ Encrypted Inference Workbench
          </span>
          <div style={{ flex: 1 }} />
          {/* health */}
          <div style={{ display:"flex", alignItems:"center", gap:5 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: healthy === null ? "#888" : healthy ? "#22cc66" : "#cc2222",
              border: "1.5px solid rgba(0,0,0,0.3)",
              boxShadow: healthy ? "0 0 5px #22cc66" : "none",
            }} />
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.42rem", letterSpacing: "0.12em",
              color: "#1a0e00",
            }}>{healthy === null ? "..." : healthy ? "ONLINE" : "OFFLINE"}</span>
          </div>
        </div>

        {/* ── Controls row — Habbo dark panel style ── */}
        <div style={{
          background: "linear-gradient(180deg, #404040 0%, #383838 100%)",
          border: "2px solid #222",
          borderTop: "1px solid #666",
          padding: "7px 10px",
          display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
          marginBottom: 8,
        }}>
          {/* Reset */}
          <button
            onClick={handleReset}
            title="Reset"
            style={{
              height: 30, paddingLeft: 10, paddingRight: 10,
              borderRadius: 3, flexShrink: 0,
              display: "flex", alignItems: "center", gap: 5,
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.42rem", letterSpacing: "0.1em",
              color: "#ddd",
              background: "linear-gradient(180deg,#5a5a5a,#3a3a3a)",
              border: "2px solid #222",
              borderTop: "2px solid #888",
              borderLeft: "2px solid #777",
              cursor: "pointer",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "linear-gradient(180deg,#6a6a6a,#4a4a4a)"}
            onMouseLeave={e => e.currentTarget.style.background = "linear-gradient(180deg,#5a5a5a,#3a3a3a)"}
          >
            <svg style={{ width:10, height:10 }} fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M5.09 15A8 8 0 0119 9M18.91 9A8 8 0 015 15" />
            </svg>
            RESET
          </button>

          {/* Run button — Habbo "Let's GO" style */}
          <button
            onClick={handleRun}
            disabled={!pixels || loading}
            style={{
              height: 34, paddingLeft: 16, paddingRight: 16,
              borderRadius: 3, flexShrink: 0,
              display: "flex", alignItems: "center", gap: 7,
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.6rem", letterSpacing: "0.14em",
              color: !pixels || loading ? "#888" : "#1a0e00",
              background: !pixels || loading
                ? "linear-gradient(180deg,#555,#3a3a3a)"
                : "linear-gradient(180deg,#f0c030,#c89800)",
              border: "2px solid",
              borderColor: !pixels || loading ? "#222" : "#7a5e00",
              borderTop: `2px solid ${!pixels || loading ? "#666" : "#f8e060"}`,
              borderLeft: `2px solid ${!pixels || loading ? "#555" : "#e8b820"}`,
              cursor: !pixels || loading ? "not-allowed" : "pointer",
              boxShadow: !pixels || loading ? "none" : "0 2px 6px rgba(0,0,0,0.5)",
            }}
            onMouseEnter={e => { if (pixels && !loading) e.currentTarget.style.background = "linear-gradient(180deg,#f8d040,#d4a400)"; }}
            onMouseLeave={e => { if (pixels && !loading) e.currentTarget.style.background = "linear-gradient(180deg,#f0c030,#c89800)"; }}
          >
            {loading ? (
              <svg className="animate-spin" style={{ width:10, height:10 }} viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg style={{ width:10, height:10 }} fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
            {loading ? "RUNNING..." : "RUN ▶"}
          </button>

          <div style={{ width: 1, height: 20, background: "#555", margin: "0 4px", flexShrink: 0 }} />

          {/* Spec pills — Habbo info box style */}
          {[
            { label:"Library",    val:"OpenFHE",    color:"#44cc88" },
            { label:"Scheme",     val:"BFV",         color:"#6aabf7" },
            { label:"Security",   val:"128-bit",     color:"#cc88ff" },
            { label:"Activation", val:"x²",          color:"#f0c030" },
            { label:"Model",      val:"LeNet-5",     color:"#cccccc" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{
              display: "flex", flexDirection: "column", gap: 1,
              padding: "3px 9px",
              background: "linear-gradient(180deg,#2a2a2a,#1e1e1e)",
              border: "1px solid #111",
              borderTop: "1px solid #444",
              borderLeft: "1px solid #3a3a3a",
              borderRadius: 2,
            }}>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.36rem", letterSpacing: "0.16em",
                color: "#888", textTransform: "uppercase",
              }}>{label}</span>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.52rem", letterSpacing: "0.08em",
                color,
              }}>{val}</span>
            </div>
          ))}

          {/* Run count */}
          <div style={{
            display: "flex", flexDirection: "column", gap: 1,
            padding: "3px 9px",
            background: "linear-gradient(180deg,#2a2a2a,#1e1e1e)",
            border: "1px solid #111",
            borderTop: "1px solid #444",
            borderLeft: "1px solid #3a3a3a",
            borderRadius: 2,
          }}>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.36rem", letterSpacing: "0.16em",
              color: "#888", textTransform: "uppercase",
            }}>Runs</span>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.52rem", letterSpacing: "0.08em",
              color: "#eee",
            }}>{String(runCount).padStart(4,"0")}</span>
          </div>

          <div style={{ flex: 1 }} />

          {/* Total time — Habbo highlighted box */}
          {result && (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 1,
              padding: "3px 12px",
              background: "linear-gradient(180deg,#1a3a1a,#122012)",
              border: "2px solid #0a4a0a",
              borderTop: "2px solid #2a6a2a",
              borderRadius: 2,
            }}>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.36rem", letterSpacing: "0.2em",
                color: "#44aa44", textTransform: "uppercase",
              }}>Total Time</span>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.9rem", letterSpacing: "0.06em",
                color: "#44ee88",
                textShadow: "0 0 8px rgba(68,238,136,0.5)",
              }}>
                {result.totalMs?.toFixed(1)}ms
              </span>
            </div>
          )}
        </div>

        {/* ── Three-column Habbo panel layout ── */}
        <div style={{
          display: "flex", gap: 6, alignItems: "stretch", minHeight: 620,
        }}>

          {/* ── INPUT PANEL — Habbo-style titled box ── */}
          <div style={{
            flexShrink: 0, width: 240,
            display: "flex", flexDirection: "column",
            border: "2px solid #222",
            borderTop: "2px solid #777",
            borderLeft: "2px solid #666",
            borderRadius: 2,
            overflow: "hidden",
          }}>
            {/* Panel title bar */}
            <div style={{
              background: "linear-gradient(180deg,#6aabf7,#3a7acc)",
              padding: "4px 10px",
              borderBottom: "2px solid #1a4a8a",
            }}>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.48rem", letterSpacing: "0.16em",
                color: "#fff",
                textShadow: "0 1px 0 rgba(0,0,0,0.4)",
                textTransform: "uppercase",
              }}>◈ Input</span>
            </div>
            {/* Panel body */}
            <div style={{
              flex: 1, background: "#d8d0c0",
              padding: "10px 10px",
              display: "flex", flexDirection: "column", gap: 10,
            }}>
              <p style={{
                fontFamily: "system-ui, sans-serif",
                fontSize: "0.78rem", fontWeight: 600,
                color: "#2a2a2a", lineHeight: 1.55, margin: 0,
              }}>
                Draw any digit (0–9) below.
              </p>

              {/* Canvas */}
              <div style={{
                border: "3px solid #555",
                borderTop: "3px solid #888",
                borderLeft: "3px solid #777",
                borderRadius: 2, overflow: "hidden",
                background: "#111",
                boxShadow: "inset 2px 2px 4px rgba(0,0,0,0.5)",
              }}>
                <MiniCanvas onPixelsReady={setPixels} disabled={loading} />
              </div>

              {/* 28×28 preview */}
              {pixels && (
                <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                  <canvas
                    width={28} height={28}
                    style={{
                      width: 56, height: 56,
                      imageRendering: "pixelated",
                      border: "2px solid #555",
                      borderTop: "2px solid #888",
                      background: "#111",
                    }}
                    ref={(el) => {
                      if (!el || !pixels) return;
                      const ctx = el.getContext("2d");
                      const img = ctx.createImageData(28, 28);
                      for (let i = 0; i < 784; i++) {
                        const v = pixels[i];
                        img.data[i*4] = v; img.data[i*4+1] = v;
                        img.data[i*4+2] = v; img.data[i*4+3] = 255;
                      }
                      ctx.putImageData(img, 0, 0);
                    }}
                  />
                  <div>
                    <p style={{
                      fontFamily: "'Press Start 2P', monospace",
                      fontSize: "0.4rem", letterSpacing: "0.1em",
                      color: "#555", margin: "0 0 3px",
                    }}>28×28 px</p>
                    <p style={{
                      fontFamily: "system-ui,sans-serif",
                      fontSize: "0.75rem", color: "#666", margin: 0,
                    }}>model input</p>
                  </div>
                </div>
              )}

              {/* Layer legend */}
              <div style={{
                marginTop: "auto",
                background: "#c8c0b0",
                border: "2px solid #999",
                borderTop: "2px solid #bbb",
                borderLeft: "2px solid #b0b0b0",
                borderRadius: 2,
                padding: "8px 10px",
              }}>
                <p style={{
                  fontFamily: "'Press Start 2P', monospace",
                  fontSize: "0.4rem", letterSpacing: "0.16em",
                  color: "#555", textTransform: "uppercase",
                  margin: "0 0 7px",
                }}>Layer types</p>
                {[
                  ["Crypto",            "#2266cc"],
                  ["Conv / Bias",       "#882299"],
                  ["Activation / Pool", "#aa8800"],
                  ["Fully Connected",   "#cc2222"],
                  ["I/O",               "#117744"],
                ].map(([label, color]) => (
                  <div key={label} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
                    <span style={{
                      width:9, height:9, borderRadius:1, flexShrink:0,
                      background:color, border:"1px solid rgba(0,0,0,0.3)",
                    }} />
                    <span style={{
                      fontFamily: "system-ui,sans-serif",
                      fontSize: "0.74rem", fontWeight:600, color:"#333",
                    }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── PIPELINE PANEL ── */}
          <div style={{
            flex: 1, minWidth: 0,
            display: "flex", flexDirection: "column",
            border: "2px solid #222",
            borderTop: "2px solid #777",
            borderLeft: "2px solid #666",
            borderRadius: 2,
            overflow: "hidden",
          }}>
            {/* Title bar */}
            <div style={{
              background: "linear-gradient(180deg,#c890f0,#8844bb)",
              padding: "4px 10px",
              borderBottom: "2px solid #5a1a99",
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.48rem", letterSpacing: "0.16em",
                color: "#fff",
                textShadow: "0 1px 0 rgba(0,0,0,0.4)",
                textTransform: "uppercase",
              }}>◈ Encrypted CNN Pipeline</span>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.36rem", color: "rgba(255,255,255,0.6)",
              }}>{LAYERS.length} layers</span>
            </div>
            {/* Body */}
            <div style={{
              flex: 1, background: "#ccc8d8",
              display: "flex", flexDirection: "column", gap: 6,
            }}>
              {/* Pipeline diagram */}
              <div style={{
                flex: 1,
                background: "#1a1428",
                borderBottom: "2px solid #444",
                display: "flex", alignItems: "center",
                overflow: "auto",
                padding: "clamp(10px,2vw,20px)",
              }}>
                <CnnPipeline
                  timings={result}
                  activeStep={activeStep}
                  hovered={hoveredLayer}
                  onHover={setHoveredLayer}
                  layerStatus={progress.running || result ? progress.layerStatus : null}
                />
              </div>

              {/* Server log feed */}
              <div style={{ padding: "0 8px 6px" }}>
                <LiveStatusFeed
                  layerStatus={progress.layerStatus}
                  elapsedMs={progress.elapsedMs}
                  running={progress.running}
                  result={result}
                  onLogComplete={handleLogComplete}
                />
              </div>

              {/* Hover tooltip — Habbo dialogue box style */}
              <AnimatePresence>
                {tl && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    style={{
                      margin: "0 8px 8px",
                      padding: "8px 12px",
                      background: "#f0ead8",
                      border: "2px solid #888",
                      borderTop: "2px solid #ccc",
                      borderLeft: "2px solid #bbb",
                      borderRadius: 2,
                      display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
                    }}
                  >
                    <span style={{
                      fontFamily: "'Press Start 2P', monospace",
                      fontSize: "0.52rem", letterSpacing: "0.08em",
                      color: "#222",
                    }}>{tl.label}</span>
                    <span style={{
                      fontFamily: "'Press Start 2P', monospace",
                      fontSize: "0.4rem", color: "#888", letterSpacing: "0.06em",
                    }}>{tl.sub}</span>
                    {tTime !== null && (
                      <span style={{
                        fontFamily: "'Press Start 2P', monospace",
                        fontSize: "0.5rem", color: "#117744",
                        letterSpacing: "0.06em",
                      }}>
                        {tTime.toFixed(2)}ms
                        <span style={{ color:"#888", marginLeft:5 }}>
                          ({((tTime / result.totalMs) * 100).toFixed(1)}%)
                        </span>
                      </span>
                    )}
                    <span style={{
                      fontFamily: "system-ui,sans-serif",
                      fontSize: "0.78rem", color: "#555",
                    }}>
                      — {getLayerDescription(tl.id)}
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* ── OUTPUT PANEL ── */}
          <div style={{
            flexShrink: 0, width: 290,
            display: "flex", flexDirection: "column",
            border: "2px solid #222",
            borderTop: "2px solid #777",
            borderLeft: "2px solid #666",
            borderRadius: 2,
            overflow: "hidden",
          }}>
            {/* Title bar */}
            <div style={{
              background: "linear-gradient(180deg,#58c896,#229955)",
              padding: "4px 10px",
              borderBottom: "2px solid #116633",
            }}>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.48rem", letterSpacing: "0.16em",
                color: "#fff",
                textShadow: "0 1px 0 rgba(0,0,0,0.4)",
                textTransform: "uppercase",
              }}>◈ Output</span>
            </div>
            {/* Body */}
            <div style={{
              flex: 1, background: "#ccd8cc",
              padding: "clamp(8px,1.5vw,14px)",
            }}>
              <OutputPanel
                result={result}
                error={error}
                loading={loading}
                pixels={pixels}
                layerStatus={progress.layerStatus}
                elapsedMs={progress.elapsedMs}
              />
            </div>
          </div>
        </div>

        {/* ── Metrics strip inside the workbench frame ── */}
        <div style={{
          marginTop: 6,
          border: "2px solid #222",
          borderTop: "2px solid #777",
          borderLeft: "2px solid #666",
          borderRadius: 2,
          overflow: "hidden",
        }}>
          <div style={{
            background: "linear-gradient(180deg,#d4a800,#b08a00)",
            padding: "4px 10px",
            borderBottom: "2px solid #7a5e00",
          }}>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.42rem", letterSpacing: "0.16em",
              color: "#1a0e00", textTransform: "uppercase",
            }}>◈ Layer Timing</span>
          </div>
          <div style={{ background: "#1a1a2a", padding: "10px 14px" }}>
            <MetricsStrip result={result} />
          </div>
        </div>
      </div>

      {/* ═══════════ MNIST BATCH BENCHMARK ═══════════ */}
      <div id="benchmarks" style={{
        background: "#5a5a5a",
        borderTop: "4px solid #888",
        borderBottom: "4px solid #2a2a2a",
        padding: "14px 18px 18px",
      }}>
        <div className="max-w-[1100px] mx-auto">
          {/* Habbo-style outer frame */}
          <div style={{
            background: "#4a4a4a",
            borderTop: "3px solid #777",
            borderLeft: "3px solid #666",
            borderBottom: "3px solid #1a1a1a",
            borderRight: "3px solid #1a1a1a",
            borderRadius: 3,
            overflow: "hidden",
          }}>
            {/* Orange nav bar — Habbo 2007 top bar */}
            <div style={{
              background: "linear-gradient(180deg, #f0a030 0%, #c87000 100%)",
              borderBottom: "3px solid #7a4000",
              borderTop: "2px solid #f8c060",
              padding: "6px 14px",
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{
                  display: "inline-block",
                  width: 10, height: 10,
                  background: "linear-gradient(135deg, #ffe080, #c07000)",
                  border: "2px solid #7a4000",
                  borderTop: "2px solid #ffe8a0",
                  borderRadius: 2,
                  flexShrink: 0,
                }} />
                <span style={{
                  fontFamily: "'Press Start 2P', monospace",
                  fontSize: "0.65rem", letterSpacing: "0.16em",
                  color: "#1a0800",
                  textShadow: "0 1px 0 rgba(255,200,80,0.4)",
                }}>MNIST BATCH BENCHMARK</span>
              </div>
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.38rem", letterSpacing: "0.08em",
                color: "rgba(26,8,0,0.6)",
              }}>10 test images · Encrypted CNN Inference</span>
            </div>

            {/* Cream content body */}
            <div style={{
              background: "#d8d0c0",
              borderTop: "2px solid #c0b8a8",
              padding: "14px",
            }}>
              <MnistBatchBenchmark />
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════ BENCHMARK RESULTS DASHBOARD ═══════════ */}
      <div id="results" className="px-4 md:px-8 py-8" style={{ background: "#f7f7f7", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <h3
              className="text-xs font-medium uppercase tracking-widest"
              style={{ color: "#888", letterSpacing: "0.12em" }}
            >
              BENCHMARK RESULTS
            </h3>
            <span className="text-[10px]" style={{ color: "#bbb" }}>
              FHE CNN Performance across Activation Degrees
            </span>
          </div>
          <div
            className="rounded-lg p-5"
            style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}
          >
            <BenchmarkResultsDashboard />
          </div>
        </div>
      </div>

      {/* ═══════════ PARAMETER COMPARISON ═══════════ */}
      <div className="px-4 md:px-8 py-8" style={{ background: "#fff", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <h3
              className="text-xs font-medium uppercase tracking-widest"
              style={{ color: "#888", letterSpacing: "0.12em" }}
            >
              PARAMETER EXPERIMENTS
            </h3>
            <span className="text-[10px]" style={{ color: "#bbb" }}>
              Activation Degree · Security Level
            </span>
          </div>
          <div
            className="rounded-lg p-5"
            style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}
          >
            <ParameterComparison />
          </div>
        </div>
      </div>

      {/* ═══════════ LIBRARY COMPARISON ═══════════ */}
      <div className="px-4 md:px-8 py-8" style={{ background: "#ebeaea", borderTop: "1px solid #d9d9d9" }}>
        <div className="max-w-[1100px] mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <h3
              className="text-xs font-medium uppercase tracking-widest"
              style={{ color: "#888", letterSpacing: "0.12em" }}
            >
              LIBRARY COMPARISON
            </h3>
            <span className="text-[10px]" style={{ color: "#bbb" }}>
              SEAL vs HElib vs OpenFHE
            </span>
          </div>
          <div
            className="rounded-lg p-5"
            style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}
          >
            <LibraryComparison
              data={compData}
              loading={compLoading}
              error={compError}
              onRun={handleComparison}
              onCancel={handleCompCancel}
            />
          </div>
        </div>
      </div>

      {/* ═══════════ EXPAND ARROW ═══════════ */}
      <div className="flex justify-center py-3" style={{ background: "#ebeaea" }}>
        <a href="#info" className="w-9 h-9 rounded-full border border-gray-300 bg-white flex items-center justify-center hover:bg-gray-100 transition">
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </a>
      </div>

      {/* ═══════════ INFO SECTION ═══════════ */}
      <section id="info" className="px-4 md:px-8 py-12" style={{ background: "#fff" }}>
        <div className="max-w-3xl mx-auto space-y-12 text-[15px] leading-relaxed" style={{ color: "#555" }}>

          {/* ── What This System Does ── */}
          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              What Does This System Do?
            </h2>
            <p className="mb-3">
              This is an <b>Encrypted Machine Learning Benchmark Framework</b> — a Final Year
              Project that demonstrates and benchmarks <b>machine learning inference on
              fully encrypted data</b>. It has two main capabilities:
            </p>
            <div className="space-y-3 ml-1">
              <div className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white" style={{ background: "#0db7c4" }}>1</span>
                <p>
                  <b style={{ color: "#333" }}>Encrypted Digit Recognition</b> — You draw a handwritten
                  digit (0–9) on the canvas. The system encrypts your pixel data, runs a full
                  convolutional neural network (CNN) <em>entirely on the encrypted data</em>, then
                  decrypts only the final result. The server <b>never sees your raw input</b>.
                </p>
              </div>
              <div className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white" style={{ background: "#3b82f6" }}>2</span>
                <p>
                  <b style={{ color: "#333" }}>Library Benchmarking</b> — The system compares three
                  leading homomorphic encryption libraries —{" "}
                  <b style={{ color: "#0db7c4" }}>OpenFHE</b>,{" "}
                  <b style={{ color: "#3b82f6" }}>Microsoft SEAL</b>, and{" "}
                  <b style={{ color: "#8b5cf6" }}>HElib</b> — on identical cryptographic operations
                  (key generation, encryption, addition, multiplication, decryption), so you
                  can see which library is fastest for each task.
                </p>
              </div>
            </div>
          </div>

          {/* ── What Is Homomorphic Encryption? ── */}
          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              What Is Homomorphic Encryption?
            </h2>
            <p className="mb-3">
              <b>Homomorphic Encryption (HE)</b> is a form of encryption that allows you to
              perform computations on encrypted data (called <b>ciphertext</b>) without
              decrypting it first. The result, when decrypted, is the same as if you had
              performed the computation on the original unencrypted data (called{" "}
              <b>plaintext</b>).
            </p>
            <p className="mb-3">
              In simple terms: the server can do maths on your data <b>without ever seeing
              it</b>. This is extremely useful for privacy-sensitive applications like
              medical records, financial data, or biometric analysis.
            </p>
            <p className="mb-3">
              This project uses the <b>BFV (Brakerski/Fan–Vercauteren) scheme</b>, which
              supports addition and multiplication on encrypted integers. These two
              operations are enough to build a full neural network — convolutions, pooling,
              and fully-connected layers are all just combinations of additions and
              multiplications.
            </p>
            <div className="rounded-lg p-4 text-sm" style={{ background: "#f8f8f8", border: "1px solid #e5e5e5" }}>
              <p className="font-medium mb-2" style={{ color: "#333" }}>The three HE libraries compared:</p>
              <ul className="space-y-1.5 ml-1">
                <li className="flex gap-2">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm mt-1.5 flex-shrink-0" style={{ background: "#0db7c4" }} />
                  <span><b style={{ color: "#0db7c4" }}>OpenFHE</b> — A modern, open-source library supporting BFV, CKKS, and TFHE schemes. Developed by Duality Technologies and a broad academic consortium.</span>
                </li>
                <li className="flex gap-2">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm mt-1.5 flex-shrink-0" style={{ background: "#3b82f6" }} />
                  <span><b style={{ color: "#3b82f6" }}>Microsoft SEAL</b> — Microsoft Research's widely used HE library supporting BFV and CKKS schemes.</span>
                </li>
                <li className="flex gap-2">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm mt-1.5 flex-shrink-0" style={{ background: "#8b5cf6" }} />
                  <span><b style={{ color: "#8b5cf6" }}>HElib</b> — IBM's pioneering HE library, primarily supporting the BGV scheme.</span>
                </li>
              </ul>
            </div>
          </div>

          {/* ── How Does the Encrypted Inference Work? ── */}
          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              How Does the Encrypted Inference Work?
            </h2>
            <p className="mb-2">
              When you draw a digit and press run, the following pipeline executes
              end-to-end:
            </p>
            <ol className="list-decimal ml-5 space-y-1.5 mb-4">
              <li>Your 28×28 pixel drawing is extracted and sent to the backend server.</li>
              <li>The server <b>encrypts</b> all 784 pixel values into BFV ciphertext using OpenFHE.</li>
              <li>A full <b>CNN (Convolutional Neural Network)</b> runs on the encrypted data — 12 layers, all computed without decrypting.</li>
              <li>The server <b>decrypts</b> only the 10 final output values (logits) and returns the predicted digit.</li>
            </ol>
            <p>
              The model was trained in PyTorch on the MNIST dataset with a{" "}
              <b>polynomial activation function</b> instead of the usual ReLU. This is because
              HE can only do addition and multiplication — it cannot do comparisons like
              max(0, x). This demo uses the <b>x² (degree 2)</b> activation, which gives the
              best accuracy and runs within the server's memory constraints. The benchmark
              section below shows results across degree 2, 3, and 4 for comparison.
            </p>

            {/* Visual pipeline walkthrough */}
            <div className="mt-6">
              <ArchitectureDiagram />
            </div>
          </div>

          {/* ── CNN Layer Glossary ── */}
          <div>
            <h2 className="text-xl font-medium mb-4" style={{ color: "#333" }}>
              CNN Layer Glossary
            </h2>
            <p className="mb-4">
              Each bubble in the network visualisation above represents a layer. Here is
              what each one does:
            </p>
            <div className="space-y-4">
              {[
                {
                  term: "Input",
                  color: "#0aa35e",
                  desc: "The raw 28×28 grayscale image (784 pixel values, 0 = black, 255 = white). This is the handwritten digit you drew on the canvas.",
                },
                {
                  term: "Encrypt",
                  color: "#0db7c4",
                  desc: "Encodes and encrypts the 784 pixel values into BFV ciphertext using the OpenFHE library. After this step, the server can no longer see the raw pixel values — all subsequent computation happens on encrypted data.",
                },
                {
                  term: "Convolution (Conv1, Conv2)",
                  color: "#7b3ff2",
                  desc: "A convolution slides a small filter (5×5 kernel) across the image and computes a weighted sum at each position. This detects local features like edges, curves, and corners. Conv1 reduces 28×28 → 24×24, and Conv2 reduces 12×12 → 8×8. In encrypted mode, each multiply-and-add is done on ciphertext.",
                },
                {
                  term: "Bias (Bias1, Bias2, BiasFc)",
                  color: "#7b3ff2",
                  desc: "A bias is a fixed value added to each output of the previous layer. It shifts the activation, allowing the network to learn patterns that don't pass through zero. In encrypted mode, this is a simple ciphertext addition.",
                },
                {
                  term: "Activation — x² (Square)",
                  color: "#e68a00",
                  desc: "An activation function introduces non-linearity so the network can learn complex patterns. Standard CNNs use ReLU (max(0, x)), but ReLU requires a comparison operation, which is impossible in HE. Instead, we use x² (squaring), which is a simple ciphertext multiplication and works natively in BFV.",
                },
                {
                  term: "Average Pooling (Pool1, Pool2)",
                  color: "#e68a00",
                  desc: "Pooling reduces the spatial dimensions by averaging groups of pixels. A 2×2 average pool takes each 2×2 block and replaces it with the mean of the four values, halving the width and height. This is just addition and division (scaling), which HE supports natively.",
                },
                {
                  term: "Fully Connected (FC)",
                  color: "#e03e52",
                  desc: "The fully-connected layer takes all remaining values (16, after the second pooling) and multiplies them by a weight matrix to produce 10 output values — one for each digit class (0–9). This is a standard matrix multiplication, done entirely on encrypted data.",
                },
                {
                  term: "Decrypt",
                  color: "#0db7c4",
                  desc: "Decrypts the 10 encrypted output values (logits) back to plaintext using the secret key. This is the only point where the data becomes readable again.",
                },
                {
                  term: "Output (Argmax)",
                  color: "#0aa35e",
                  desc: "Takes the 10 decrypted logit values and picks the index with the highest value — that index is the predicted digit. For example, if logit[7] is the largest, the prediction is \"7\".",
                },
              ].map(({ term, color, desc }) => (
                <div key={term} className="flex gap-3">
                  <span
                    className="flex-shrink-0 w-3 h-3 rounded-full mt-1.5"
                    style={{ background: color }}
                  />
                  <div>
                    <p className="font-medium text-sm" style={{ color: "#333" }}>{term}</p>
                    <p className="text-sm" style={{ color: "#666" }}>{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Can I Repurpose It? ── */}
          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              Can I Repurpose This?
            </h2>
            <p>
              Absolutely. This is an open-source Final Year Project — an{" "}
              <b>Encrypted Machine Learning Benchmark Framework</b> comparing three HE
              libraries on standardised workloads. The entire system (Rust gRPC server,
              Spring Boot API, React frontend) is available on{" "}
              <a
                href="https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework"
                target="_blank"
                rel="noreferrer"
                className="underline"
                style={{ color: "#f4743a" }}
              >
                GitHub
              </a>.
            </p>
          </div>
        </div>
      </section>

      {/* ═══════════ FOOTER ═══════════ */}
      <footer className="text-center py-4 text-xs" style={{ color: "#999", background: "#f7f7f7", borderTop: "1px solid #e5e5e5" }}>
        Built by Tiffany Yong · FYP 2025-2026 · Powered by OpenFHE, SEAL, HElib, Spring Boot &amp; React
      </footer>

      {/* ═══════════ RUN LOG HISTORY MODAL ═══════════ */}
      <AnimatePresence>
        {historyOpen && runHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: "fixed", inset: 0, zIndex: 9999,
              background: "rgba(5,10,20,0.82)",
              backdropFilter: "blur(6px)",
              display: "flex", alignItems: "center", justifyContent: "center",
              padding: "20px",
            }}
            onClick={() => setHistoryOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.93, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ type: "spring", stiffness: 320, damping: 28 }}
              onClick={e => e.stopPropagation()}
              style={{
                width: "100%", maxWidth: 620, maxHeight: "80vh",
                borderRadius: 12,
                background: "#0d1826",
                border: "1.5px solid rgba(240,192,48,0.22)",
                boxShadow: "0 0 60px rgba(240,192,48,0.08), 0 24px 80px rgba(0,0,0,0.6)",
                display: "flex", flexDirection: "column",
                overflow: "hidden",
              }}
            >
              {/* Modal header */}
              <div style={{
                padding: "14px 18px",
                borderBottom: "1px solid rgba(240,192,48,0.12)",
                display: "flex", alignItems: "center", justifyContent: "space-between",
                flexShrink: 0,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{
                    fontFamily: "'Press Start 2P', monospace",
                    fontSize: "0.58rem", letterSpacing: "0.2em", textTransform: "uppercase",
                    color: "#f0c030",
                  }}>Server Logs</span>
                  <span style={{
                    fontFamily: "'Press Start 2P', monospace",
                    fontSize: "0.42rem", letterSpacing: "0.1em",
                    color: "rgba(240,192,48,0.4)",
                  }}>{runHistory.length} run{runHistory.length !== 1 ? "s" : ""}</span>
                </div>
                <button
                  onClick={() => setHistoryOpen(false)}
                  style={{
                    background: "none", border: "none", cursor: "pointer",
                    color: "rgba(255,248,220,0.4)", fontSize: "1.1rem",
                    lineHeight: 1, padding: "2px 6px",
                    transition: "color 0.15s",
                  }}
                  onMouseEnter={e => e.target.style.color = "rgba(255,248,220,0.9)"}
                  onMouseLeave={e => e.target.style.color = "rgba(255,248,220,0.4)"}
                >✕</button>
              </div>

              {/* Run tabs */}
              {runHistory.length > 1 && (
                <div style={{
                  display: "flex", gap: 4, padding: "8px 14px",
                  borderBottom: "1px solid rgba(240,192,48,0.08)",
                  overflowX: "auto", flexShrink: 0,
                }}>
                  {runHistory.map((run, i) => (
                    <button
                      key={i}
                      onClick={() => setHistoryIndex(i)}
                      style={{
                        fontFamily: "'Press Start 2P', monospace",
                        fontSize: "0.4rem", letterSpacing: "0.1em",
                        padding: "5px 10px", borderRadius: 4, cursor: "pointer",
                        border: i === historyIndex
                          ? "1.5px solid rgba(240,192,48,0.5)"
                          : "1.5px solid rgba(255,248,220,0.08)",
                        background: i === historyIndex
                          ? "rgba(240,192,48,0.1)"
                          : "rgba(255,255,255,0.03)",
                        color: i === historyIndex
                          ? "#f0c030"
                          : "rgba(255,248,220,0.35)",
                        flexShrink: 0, transition: "all 0.15s",
                        whiteSpace: "nowrap",
                      }}
                    >
                      Run #{run.runIndex}
                      <span style={{ opacity: 0.6, marginLeft: 6 }}>{run.timestamp}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* Log body */}
              {(() => {
                const run = runHistory[historyIndex];
                if (!run) return null;
                return (
                  <ModalLogBody run={run} />
                );
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ─── Reusable sub-components ─── */

/**
 * ModalLogBody — renders the run log in the history modal.
 * Rows match the live Server Log feed exactly.
 * Hovering a row expands the full cryptographic description.
 */
function ModalLogBody({ run }) {
  const [hoveredRow, setHoveredRow] = useState(null);

  return (
    <div style={{
      overflowY: "auto", padding: "14px 18px",
      display: "flex", flexDirection: "column", gap: 0,
      flex: 1,
    }}>
      {/* ── Header matching the live feed ── */}
      <div style={{
        padding: "8px 14px",
        borderBottom: "1px solid rgba(240,192,48,0.1)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 8,
      }}>
        <span style={{
          fontFamily: "'Press Start 2P', monospace",
          fontSize: "0.55rem", letterSpacing: "0.2em", textTransform: "uppercase",
          color: "rgba(240,192,48,0.5)",
        }}>Server Log</span>
        <span style={{
          fontFamily: "'Press Start 2P', monospace",
          fontSize: "0.45rem", letterSpacing: "0.1em",
          color: "#58c896",
        }}>done · {run.result?.totalMs ? (run.result.totalMs / 1000).toFixed(2) + "s" : "—"}</span>
      </div>

      {/* ── Compact log rows — identical to live feed ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "0 4px" }}>
        {run.log.map((entry, i) => (
          <div
            key={i}
            onMouseEnter={() => setHoveredRow(i)}
            onMouseLeave={() => setHoveredRow(null)}
            style={{
              borderRadius: 6,
              transition: "background 0.15s",
              background: hoveredRow === i ? `${entry.color}0d` : "transparent",
              border: hoveredRow === i ? `1px solid ${entry.color}22` : "1px solid transparent",
              cursor: "default",
              overflow: "hidden",
            }}
          >
            {/* ── Compact row — same layout as LiveStatusFeed compact log ── */}
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "7px 10px",
            }}>
              {/* Colour dot */}
              <span style={{
                width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
                background: entry.color,
                boxShadow: `0 0 5px ${entry.color}66`,
              }} />
              {/* Timestamp */}
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.4rem", letterSpacing: "0.06em",
                color: "rgba(255,248,220,0.28)", flexShrink: 0, width: 38,
              }}>{(entry.atMs / 1000).toFixed(1)}s</span>
              {/* Label */}
              <span style={{
                fontFamily: "system-ui, sans-serif",
                fontSize: "0.72rem", color: "rgba(255,248,220,0.72)",
                flex: 1,
              }}>{entry.label}</span>
              {/* Duration */}
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.38rem", letterSpacing: "0.06em",
                color: entry.color, opacity: 0.85, flexShrink: 0,
              }}>{(entry.durationMs / 1000).toFixed(1)}s</span>
              {/* Tick */}
              <span style={{ color: "#58c896", fontSize: "0.75rem", flexShrink: 0 }}>✓</span>
            </div>

            {/* ── Description — expands on hover ── */}
            <AnimatePresence>
              {hoveredRow === i && entry.desc && (
                <motion.div
                  key="desc"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.22, ease: "easeOut" }}
                  style={{ overflow: "hidden" }}
                >
                  <p style={{
                    fontFamily: "Georgia, 'Times New Roman', serif",
                    fontSize: "0.8rem",
                    color: "rgba(255,248,210,0.65)",
                    lineHeight: 1.8, margin: 0,
                    padding: "0 10px 10px 24px",
                    borderTop: `1px solid ${entry.color}18`,
                    paddingTop: 8,
                  }}>{entry.desc}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>

      {/* ── All done summary — same as live feed ── */}
      <div style={{
        marginTop: 10,
        padding: "8px 12px",
        borderRadius: 6,
        background: "rgba(88,200,150,0.06)",
        border: "1px solid rgba(88,200,150,0.2)",
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <span style={{ color: "#58c896", fontSize: "0.85rem" }}>★</span>
        <span style={{
          fontFamily: "'Press Start 2P', monospace",
          fontSize: "0.42rem", letterSpacing: "0.1em",
          color: "#58c896",
        }}>
          Inference complete — {run.log.length} layers · {run.result?.totalMs ? (run.result.totalMs / 1000).toFixed(2) + "s" : "—"} total
        </span>
      </div>
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h3
      className="text-xs font-medium uppercase tracking-widest mb-1"
      style={{ color: "#888", letterSpacing: "0.12em" }}
    >
      {children}
    </h3>
  );
}

function ControlLabel({ label, value, mono }) {
  return (
    <div className="px-3">
      <div className="text-[10px] uppercase tracking-wide" style={{ color: "#999" }}>{label}</div>
      <div
        className={`text-sm font-medium ${mono ? "font-mono" : ""}`}
        style={{ color: "#333" }}
      >
        {value}
      </div>
    </div>
  );
}

function ControlStaticLabel({ label, value }) {
  return (
    <div className="px-2 md:px-3">
      <div className="text-[10px] uppercase tracking-wide mb-0.5" style={{ color: "#999" }}>{label}</div>
      <div
        className="text-sm font-medium px-2 py-0.5 rounded border"
        style={{ color: "#555", borderColor: "#e0e0e0", background: "#f5f5f5", whiteSpace: "nowrap" }}
      >
        {value}
      </div>
    </div>
  );
}

function Divider() {
  return <div className="w-px h-8 mx-1" style={{ background: "#ccc" }} />;
}

/* ── Layer descriptions ── */
function getLayerDescription(id) {
  const map = {
    input:   "Raw 28×28 pixel image in plaintext",
    encrypt: "Encode & encrypt using BFV scheme (OpenFHE)",
    conv1:   "Encrypted 5×5 convolution, single channel",
    bias1:   "Add encrypted bias vector to conv1 output",
    relu1:   "Polynomial activation (configurable degree)",
    pool1:   "2×2 average pooling on encrypted data",
    conv2:   "Encrypted 5×5 convolution, single channel",
    bias2:   "Add encrypted bias vector to conv2 output",
    relu2:   "Polynomial activation (configurable degree)",
    pool2:   "2×2 average pooling on encrypted data",
    fc:      "Encrypted fully-connected layer → 10 logits",
    biasfc:  "Add encrypted bias vector to FC output",
    decrypt: "Decrypt BFV ciphertext back to plaintext",
    output:  "Softmax + argmax to get predicted digit",
  };
  return map[id] || "";
}
