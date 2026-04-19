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
import NeuralHero from "./NeuralHero";
import CnnClassroom from "./CnnClassroom";
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

      {/* ═══════════ MNIST SECTION HEADER ═══════════ */}
      <SkyBanner
        label="Interactive Demo"
        title="MNIST Encrypted Inference"
        subtitle="From your drawn digit to a prediction — every step happens on encrypted data. Scroll through the tutorial below to see exactly how."
      />

      {/* ═══════════ CNN CLASSROOM — animated explainer ═══════════ */}
      <CnnClassroom/>

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

      {/* ═══════════ LIBRARY COMPARISON ═══════════ */}
      <SkyBanner
        label="Live Benchmark"
        title="Library Comparison"
        subtitle="Run identical HE operations across OpenFHE, SEAL, and HElib — see which is fastest."
      />
      <div style={{ background: "#1a0f06", borderTop: "1px solid #5a3510", overflow: "hidden" }}>
        <LibraryComparison
          data={compData}
          loading={compLoading}
          error={compError}
          onRun={handleComparison}
          onCancel={handleCompCancel}
        />
      </div>

      {/* ═══════════ MNIST BATCH BENCHMARK ═══════════ */}
      <SkyBanner
        label="Batch Testing"
        title="MNIST Batch Benchmark"
        subtitle="10 real test images encrypted and classified end-to-end. Real timings, real results."
      />
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
      <SkyBanner
        label="Deep Dive"
        title="FHE CNN Benchmark Results"
        subtitle="Encrypted CNN inference benchmarked across polynomial activation degrees. x² is the validated result. x³ and x⁴ show what breaks and why."
      />
      <div id="results" className="px-4 md:px-8 py-8" style={{ background: "#f7f7f7", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">

          {/* ── Context banner ── */}
          <div className="rounded-lg p-4 mb-4" style={{ background: "#0f172a", border: "1px solid #1e293b" }}>
            <p className="text-sm leading-relaxed" style={{ color: "#94a3b8", margin: 0 }}>
              <span style={{ color: "#e2e8f0", fontWeight: 600 }}>How to read these results: </span>
              These benchmarks show what happens when you push BFV encryption beyond its stable operating zone.
              x² is the fully validated configuration — high accuracy, complete run, production-ready.
              x³ and x⁴ are deliberate boundary-finding experiments: they reveal exactly where and why the scheme breaks down,
              which is a concrete finding in itself. A 0% accuracy result with a known cause is not a failure — it is data.
            </p>
          </div>

          {/* ── Hardware note ── */}
          <div className="rounded-lg p-4 mb-4 text-sm leading-relaxed" style={{ background: "#fffbeb", border: "1.5px solid #f59e0b" }}>
            <p className="font-semibold mb-1" style={{ color: "#92400e" }}>Hardware note</p>
            <p style={{ color: "#78350f", margin: 0 }}>
              All runs required an <b>r6i.large instance (16 GB RAM)</b> — a t3.small runs out of memory during BFV key generation at n = 4096.
              Due to EC2 budget constraints the full 100-image run could not be completed for every configuration.
              x² and x³ results are from partial but valid runs. x⁴ data is <b>directional only — treat with caution</b>.
            </p>
          </div>

          {/* ── What is this ── */}
          <div className="rounded-lg p-4 mb-6 text-sm leading-relaxed" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
            <p className="font-semibold mb-1" style={{ color: "#333" }}>Why polynomial degree matters</p>
            <p style={{ color: "#555", margin: 0 }}>
              Standard neural networks use ReLU (<code>max(0,x)</code>), which requires a comparison — impossible on ciphertext.
              FHE forces us to replace it with a low-degree polynomial evaluated directly on encrypted values.
              Degree 2 (x²) is the most stable: minimal noise growth, highest accuracy, fully validated.
              Degree 3 amplifies intermediate values into the overflow zone of the plaintext modulus, corrupting activations.
              Degree 4 makes this worse — the signal is destroyed before it even reaches the fully-connected layer.
              The accuracy drop you see is not a model quality issue. It is BFV arithmetic hitting its ceiling.
            </p>
          </div>

          <div className="rounded-lg p-5" style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}>
            <BenchmarkResultsDashboard />
          </div>
        </div>
      </div>

      {/* ═══════════ PARAMETER COMPARISON ═══════════ */}
      <SkyBanner
        label="Findings"
        title="Parameter Exploration & Limitations"
        subtitle="A systematic search for the practical limits of BFV on commodity hardware. Every wall found was found on purpose."
      />
      <div className="px-4 md:px-8 py-8" style={{ background: "#fff", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">

          {/* ── Framing callout ── */}
          <div className="rounded-lg p-4 mb-6 text-sm leading-relaxed" style={{ background: "#f8fafc", border: "1px solid #e2e8f0" }}>
            <p className="font-semibold mb-1" style={{ color: "#0f172a" }}>What this section is</p>
            <p style={{ color: "#475569", margin: 0 }}>
              Rather than only running the configuration that works, this project deliberately tested combinations
              known to be borderline or risky — higher polynomial degrees, stronger security levels, extreme scale factors.
              The goal was to find the feasibility ceiling of single-node BFV deployment and document exactly where it sits.
              The results below are that map.
            </p>
          </div>

          {/* ── Feasibility map ── */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold mb-1" style={{ color: "#0f172a" }}>Feasibility map — Activation Degree × Security Level</h3>
            <p className="text-xs mb-3" style={{ color: "#64748b" }}>
              Each cell shows whether the combination runs correctly on a single r6i.large instance.
            </p>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "#f1f5f9" }}>
                    <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#475569", border: "1px solid #e2e8f0" }}>Activation</th>
                    <th style={{ padding: "10px 16px", textAlign: "center", fontWeight: 600, color: "#475569", border: "1px solid #e2e8f0" }}>128-bit security</th>
                    <th style={{ padding: "10px 16px", textAlign: "center", fontWeight: 600, color: "#475569", border: "1px solid #e2e8f0" }}>192-bit security</th>
                    <th style={{ padding: "10px 16px", textAlign: "center", fontWeight: 600, color: "#475569", border: "1px solid #e2e8f0" }}>256-bit security</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      degree: "x² (degree 2)",
                      cells: [
                        { status: "works",    label: "Validated", note: ">80% accuracy, <5s keygen", bg: "#f0fdf4", border: "#86efac", text: "#15803d" },
                        { status: "oom",      label: "OOM",       note: "n ≥ 8192 required — exceeds 16 GB", bg: "#fff1f2", border: "#fca5a5", text: "#dc2626" },
                        { status: "oom",      label: "OOM",       note: "n ≥ 16384 — not attempted", bg: "#fff1f2", border: "#fca5a5", text: "#dc2626" },
                      ],
                    },
                    {
                      degree: "x³ (degree 3)",
                      cells: [
                        { status: "partial",  label: "Partial",   note: "Runs but accuracy collapses — plaintext modulus overflow", bg: "#fffbeb", border: "#fcd34d", text: "#b45309" },
                        { status: "oom",      label: "OOM",       note: "Not tested — 128-bit already shows accuracy failure", bg: "#f8fafc", border: "#e2e8f0", text: "#94a3b8" },
                        { status: "oom",      label: "N/A",       note: "Not tested", bg: "#f8fafc", border: "#e2e8f0", text: "#94a3b8" },
                      ],
                    },
                    {
                      degree: "x⁴ (degree 4)",
                      cells: [
                        { status: "incomplete", label: "Incomplete", note: "10-image run only — signal destroyed before FC layer", bg: "#fffbeb", border: "#fcd34d", text: "#b45309" },
                        { status: "oom",        label: "N/A",        note: "Not tested", bg: "#f8fafc", border: "#e2e8f0", text: "#94a3b8" },
                        { status: "oom",        label: "N/A",        note: "Not tested", bg: "#f8fafc", border: "#e2e8f0", text: "#94a3b8" },
                      ],
                    },
                  ].map(({ degree, cells }) => (
                    <tr key={degree}>
                      <td style={{ padding: "10px 16px", fontWeight: 600, color: "#0f172a", border: "1px solid #e2e8f0", whiteSpace: "nowrap" }}>{degree}</td>
                      {cells.map((cell, ci) => (
                        <td key={ci} style={{ padding: "10px 16px", textAlign: "center", border: "1px solid #e2e8f0", background: cell.bg }}>
                          <div style={{ display: "inline-block", padding: "2px 10px", borderRadius: 4, border: `1px solid ${cell.border}`, color: cell.text, fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                            {cell.label}
                          </div>
                          <div style={{ fontSize: 11, color: "#64748b", lineHeight: 1.5 }}>{cell.note}</div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ── x⁴ honest callout ── */}
          <div className="rounded-lg p-4 mb-6 text-sm" style={{ background: "#fffbeb", border: "1.5px solid #f59e0b" }}>
            <p className="font-semibold mb-1" style={{ color: "#92400e" }}>About the x⁴ result</p>
            <p style={{ color: "#78350f", lineHeight: 1.75, margin: 0 }}>
              The 100-image x⁴ benchmark was stopped mid-way due to EC2 budget constraints.
              The accuracy figure shown in the charts comes from a <b>10-image partial run only</b> — treat it as directional, not conclusive.
              What is conclusive: the x⁴ signal collapses before the FC layer because the fourth-power term amplifies
              intermediate ciphertext values far beyond the plaintext modulus (100,073,473), causing silent modular wrap-around
              at every activation. This is a known BFV property, not a model or implementation bug.
            </p>
          </div>

          {/* ── Noise budget visualisation ── */}
          <div className="rounded-lg p-4 mb-6 text-sm" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
            <p className="font-semibold mb-1" style={{ color: "#0f172a" }}>Why higher degrees break — noise budget per layer</p>
            <p className="text-xs mb-4" style={{ color: "#64748b" }}>
              BFV tracks a "noise budget" (bits). Each ciphertext multiplication consumes it. When it hits zero, decryption produces garbage.
              This is a conceptual illustration of relative consumption — the exact values depend on n, p, and scale.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {[
                {
                  degree: "x² — 2 activations", color: "#16a34a", bg: "#dcfce7",
                  stages: [
                    { label: "Conv1", pct: 15 }, { label: "Act1 (x²)", pct: 20 },
                    { label: "Conv2", pct: 15 }, { label: "Act2 (x²)", pct: 20 },
                    { label: "FC", pct: 20 }, { label: "Spare", pct: 10, spare: true },
                  ],
                  note: "Budget survives to FC. Decryption succeeds.",
                },
                {
                  degree: "x³ — 2 activations", color: "#b45309", bg: "#fef3c7",
                  stages: [
                    { label: "Conv1", pct: 15 }, { label: "Act1 (x³)", pct: 38 },
                    { label: "Conv2", pct: 15 }, { label: "Act2 (x³)", pct: 38, overflow: true },
                  ],
                  note: "Budget exhausted at Act2. Values wrap around the modulus, corrupting all activations from this point.",
                },
                {
                  degree: "x⁴ — 2 activations", color: "#dc2626", bg: "#fee2e2",
                  stages: [
                    { label: "Conv1", pct: 15 }, { label: "Act1 (x⁴)", pct: 90, overflow: true },
                  ],
                  note: "Budget gone at first activation. Everything after is noise.",
                },
              ].map(({ degree, color, bg, stages, note }) => {
                const total = stages.reduce((s, st) => s + st.pct, 0);
                const capped = Math.min(total, 100);
                return (
                  <div key={degree}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color }}>{degree}</span>
                      <span style={{ fontSize: 11, color: total > 100 ? "#dc2626" : "#64748b" }}>
                        {total > 100 ? "OVERFLOW" : `${capped}% consumed`}
                      </span>
                    </div>
                    <div style={{ display: "flex", height: 22, borderRadius: 4, overflow: "hidden", border: "1px solid #e2e8f0", background: "#f8fafc" }}>
                      {stages.map((st, si) => {
                        const runningTotal = stages.slice(0, si + 1).reduce((s, x) => s + x.pct, 0);
                        const clampedPct = Math.max(0, Math.min(st.pct, 100 - stages.slice(0, si).reduce((s, x) => s + x.pct, 0)));
                        if (clampedPct <= 0) return null;
                        return (
                          <div key={st.label} title={`${st.label}: ${st.pct}% budget`} style={{
                            width: `${clampedPct}%`, background: st.overflow ? "#dc2626" : st.spare ? "#e2e8f0" : color,
                            opacity: st.spare ? 0.4 : 1,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 9, color: st.spare ? "#94a3b8" : "#fff", fontWeight: 600,
                            overflow: "hidden", whiteSpace: "nowrap", minWidth: 0,
                            borderRight: si < stages.length - 1 ? "1px solid rgba(255,255,255,0.3)" : "none",
                          }}>
                            {clampedPct > 8 ? st.label : ""}
                          </div>
                        );
                      })}
                      {total < 100 && (
                        <div style={{ flex: 1, background: "#f1f5f9" }} />
                      )}
                    </div>
                    <p style={{ fontSize: 11, color: "#64748b", marginTop: 4, marginBottom: 0 }}>{note}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Scale / modulus "what if you get it wrong" ── */}
          <div className="rounded-lg p-4 mb-6 text-sm" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
            <p className="font-semibold mb-1" style={{ color: "#0f172a" }}>Scale factor — what happens when you get it wrong</p>
            <p className="text-xs mb-3" style={{ color: "#64748b" }}>
              The scale factor S multiplies weights before integer encoding. Too small = precision loss. Too large = overflow.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 0, border: "1px solid #e2e8f0", borderRadius: 8, overflow: "hidden" }}>
              {[
                { val: "S = 100", verdict: "Precision loss", color: "#b45309", bg: "#fffbeb", border: "#fcd34d", why: "Weight values rounded too aggressively during integer encoding — quantization error accumulates across layers and degrades accuracy." },
                { val: "S = 1,000", verdict: "Stable", color: "#16a34a", bg: "#f0fdf4", border: "#86efac", why: "Enough precision to preserve meaningful weight differences, small enough to stay well clear of the plaintext modulus ceiling. This is the validated baseline." },
                { val: "S = 10,000", verdict: "Overflow risk", color: "#dc2626", bg: "#fff1f2", border: "#fca5a5", why: "Larger encoded values push intermediate computation results closer to p = 100,073,473. At higher polynomial degrees, this causes modular wrap-around and silent corruption." },
              ].map(({ val, verdict, color, bg, border, why }, i, arr) => (
                <div key={val} style={{ display: "flex", alignItems: "flex-start", gap: 16, padding: "12px 16px", background: bg, borderBottom: i < arr.length - 1 ? `1px solid ${border}` : "none" }}>
                  <div style={{ flexShrink: 0, minWidth: 80, fontWeight: 700, fontSize: 13, color }}>{val}</div>
                  <div style={{ flexShrink: 0, minWidth: 100 }}>
                    <span style={{ display: "inline-block", padding: "1px 8px", borderRadius: 4, border: `1px solid ${border}`, color, fontWeight: 600, fontSize: 11 }}>{verdict}</span>
                  </div>
                  <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.65 }}>{why}</div>
                </div>
              ))}
            </div>

            <p className="font-semibold mb-1 mt-4" style={{ color: "#0f172a" }}>Plaintext modulus — what happens when you get it wrong</p>
            <p className="text-xs mb-3" style={{ color: "#64748b" }}>
              p is the ceiling for all intermediate arithmetic values. Every activation and convolution output must stay below it.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 0, border: "1px solid #e2e8f0", borderRadius: 8, overflow: "hidden" }}>
              {[
                { val: "p = 65,537", verdict: "Too small", color: "#dc2626", bg: "#fff1f2", border: "#fca5a5", why: "16-bit modulus — intermediate values from conv and activation overflow almost immediately. Results are garbage after the first layer." },
                { val: "p = 100,073,473", verdict: "Stable", color: "#16a34a", bg: "#f0fdf4", border: "#86efac", why: "Baseline modulus, chosen to give enough headroom for scale = 1,000 and degree 2 activations. Validated end-to-end." },
                { val: "p = 4,294,967,311", verdict: "Headroom, slower", color: "#b45309", bg: "#fffbeb", border: "#fcd34d", why: "32-bit modulus gives more overflow headroom but increases the cost of every ciphertext operation proportionally. Useful if you raise the scale factor or degree." },
              ].map(({ val, verdict, color, bg, border, why }, i, arr) => (
                <div key={val} style={{ display: "flex", alignItems: "flex-start", gap: 16, padding: "12px 16px", background: bg, borderBottom: i < arr.length - 1 ? `1px solid ${border}` : "none" }}>
                  <div style={{ flexShrink: 0, minWidth: 140, fontWeight: 700, fontSize: 13, color }}>{val}</div>
                  <div style={{ flexShrink: 0, minWidth: 100 }}>
                    <span style={{ display: "inline-block", padding: "1px 8px", borderRadius: 4, border: `1px solid ${border}`, color, fontWeight: 600, fontSize: 11 }}>{verdict}</span>
                  </div>
                  <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.65 }}>{why}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg p-5" style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}>
            <ParameterComparison />
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

      {/* ═══════════ INFO SECTION — PIXEL GUIDEBOOK ═══════════ */}
      <BookGuidebook />

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

/**
 * SkyBanner — Stardew Valley–style sky-blue section divider.
 * Golden pixel-art title with chunky dark drop shadow, soft clouds.
 */

/* ─── BookGuidebook ─────────────────────────────────────────────────────────── */
function BookGuidebook() {
  const [openPipeline, setOpenPipeline] = useState(null);
  const [openParam, setOpenParam] = useState(null);

  const pipelineSteps = [
    { label: "Draw", sub: "28×28 pixels", color: "#16a34a", detail: "You sketch a digit on the canvas. The app reads 784 pixel values (0–255) from the 28×28 grid — the same format the original MNIST dataset uses." },
    { label: "Encrypt", sub: "BFV ciphertext", color: "#0891b2", detail: "The pixel array is encoded into a polynomial and encrypted using the BFV (Brakerski–Fan–Vercauteren) scheme. From this point on the server cannot read any values — only operate on them." },
    { label: "Conv 1", sub: "5×5 kernel", color: "#7c3aed", detail: "A 5×5 convolutional kernel slides over the encrypted feature map. Each output pixel is a linear combination of 25 ciphertext additions and scalar multiplications — no data is revealed." },
    { label: "Activate", sub: "x² (no ReLU)", color: "#b45309", detail: "Standard ReLU requires comparing a value to zero — impossible on ciphertext. Instead we evaluate x², a polynomial that approximates nonlinearity and can be computed with a single ciphertext multiplication." },
    { label: "Pool", sub: "avg 2×2", color: "#ca8a04", detail: "2×2 average pooling halves the spatial resolution. On encrypted data this is just a weighted sum — four ciphertext additions and a scalar divide, no branching needed." },
    { label: "Conv 2", sub: "5×5 kernel", color: "#6d28d9", detail: "A second convolutional layer extracts higher-level features. The computation is identical to Conv 1 but operates on the (encrypted) pooled output of the first layer." },
    { label: "Activate", sub: "x² again", color: "#d97706", detail: "The same x² activation is applied again. This is the second ciphertext multiplication in the multiplicative depth chain — BFV can handle this at n = 4096 without exhausting the noise budget." },
    { label: "FC Layer", sub: "10 logits", color: "#dc2626", detail: "A fully-connected layer maps the flattened feature vector to 10 output logits — one per digit class. This is the most expensive step: hundreds of ciphertext additions per output neuron." },
    { label: "Decrypt", sub: "secret key", color: "#0891b2", detail: "The 10 encrypted logits are sent back to the client. The secret key (which never left your session) decrypts them into 10 plaintext scores." },
    { label: "Result", sub: "predicted digit", color: "#16a34a", detail: "Argmax picks the highest logit. That index is your predicted digit. The entire pipeline ran on locked data — the server learned nothing about your drawing." },
  ];

  const params = [
    { term: "Polynomial Degree", value: "x² (degree 2)", color: "#7c3aed", verdict: "Only valid choice", detail: "Replaces ReLU in FHE. Degree 2 is the only option that preserves accuracy end-to-end — degree 3 and 4 cause intermediate values to overflow the plaintext modulus, corrupting every activation silently." },
    { term: "Security Level", value: "128-bit", color: "#0891b2", verdict: "Hardware ceiling", detail: "128-bit NIST security works on a single r6i.large (16 GB RAM). 192-bit requires a larger ring dimension (n ≥ 8192) and OOM'd at 7.6 GB + 15 GB swap after 60+ minutes — never completed." },
    { term: "Ring Dimension n", value: "n = 4096", color: "#b45309", verdict: "Fixed", detail: "The size of the polynomial ring. Larger n gives more security and noise budget but exponentially more memory. n = 4096 is the minimum viable for 128-bit BFV with mult_depth = 6." },
    { term: "Plaintext Modulus p", value: "100,073,473", color: "#dc2626", verdict: "Baseline is safest", detail: "The ceiling for intermediate computation values. Too small and activations overflow (modular wrap-around silently corrupts results). The baseline value gives enough headroom at scale = 1,000." },
    { term: "Scale Factor S", value: "S = 1,000", color: "#16a34a", verdict: "Sweet spot", detail: "Multiplies weights before integer encoding to preserve decimal precision. S = 100 loses too much precision; S = 10,000 risks overflow into the modulus. S = 1,000 hits the right balance." },
    { term: "Multiplication Depth", value: "depth = 6", color: "#ca8a04", verdict: "Fixed", detail: "The maximum number of sequential ciphertext multiplications before noise overwhelms the signal. Our CNN uses: Conv1 × Act1 × Conv2 × Act2 × FC × 1 spare = depth 6." },
  ];

  return (
    <section id="info" style={{ background: "#f8fafc" }}>

      {/* ── Hero ── */}
      <div style={{ background: "linear-gradient(135deg, #14532d 0%, #166534 40%, #15803d 100%)", padding: "64px 24px 56px", textAlign: "center" }}>
        <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 12, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "rgba(187,247,208,0.8)", marginBottom: 18 }}>
          About This Project
        </p>
        <h2 style={{ fontFamily: "system-ui,sans-serif", fontSize: "clamp(1.6rem,4vw,2.6rem)", fontWeight: 800, color: "#fff", lineHeight: 1.25, marginBottom: 20, maxWidth: 700, margin: "0 auto 20px" }}>
          What if an AI could classify your data without ever seeing it?
        </h2>
        <p style={{ fontFamily: "system-ui,sans-serif", fontSize: "clamp(0.95rem,1.8vw,1.1rem)", color: "rgba(187,247,208,0.9)", maxWidth: 580, margin: "0 auto 32px", lineHeight: 1.8 }}>
          This demo runs a neural network entirely on <strong style={{ color: "#fff" }}>encrypted data</strong> using Fully Homomorphic Encryption — the server computes the answer without ever decrypting your input.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
          {[
            { label: "Draw a digit", desc: "your input, your key" },
            { label: "Server encrypts + infers", desc: "never sees plaintext" },
            { label: "You decrypt the result", desc: "server learns nothing" },
          ].map(({ label, desc }) => (
            <div key={label} style={{ background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "12px 20px", minWidth: 160 }}>
              <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 14, color: "#fff", marginBottom: 4 }}>{label}</div>
              <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 12, color: "rgba(187,247,208,0.8)" }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "0 24px 72px" }}>

        {/* ── What you can do ── */}
        <div style={{ padding: "52px 0 44px", borderBottom: "1px solid #e2e8f0" }}>
          <h3 style={{ fontFamily: "system-ui,sans-serif", fontSize: 22, fontWeight: 700, color: "#0f172a", marginBottom: 8 }}>What you can do here</h3>
          <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 14, color: "#64748b", marginBottom: 32, lineHeight: 1.7 }}>Four sections, each independently useful. Start anywhere.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: 16 }}>
            {[
              { title: "Draw & Predict", desc: "Sketch any digit (0–9) and watch the full encrypted CNN inference run live — every layer, real timings.", color: "#16a34a", bg: "#f0fdf4", border: "#86efac" },
              { title: "Compare Libraries", desc: "Benchmark OpenFHE, SEAL, and HElib on five identical HE operations. See which library is fastest at each.", color: "#0891b2", bg: "#ecfeff", border: "#a5f3fc" },
              { title: "Explore Parameters", desc: "See what happens when you change activation degree, security level, or scale factor — and exactly why some break.", color: "#7c3aed", bg: "#faf5ff", border: "#d8b4fe" },
              { title: "Read the Results", desc: "Real benchmark data from 10–100 MNIST images. Accuracy and latency across three polynomial degrees.", color: "#b45309", bg: "#fffbeb", border: "#fcd34d" },
            ].map(({ title, desc, color, bg, border }) => (
              <div key={title} style={{ background: bg, border: `1.5px solid ${border}`, borderRadius: 10, padding: "20px 20px 18px" }}>
                <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 14, color, marginBottom: 8 }}>{title}</div>
                <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#475569", lineHeight: 1.7 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Pipeline ── */}
        <div style={{ padding: "48px 0 44px", borderBottom: "1px solid #e2e8f0" }}>
          <div style={{ background: "linear-gradient(135deg,#0c4a6e,#0369a1)", borderRadius: 12, padding: "32px 32px 28px", marginBottom: 32 }}>
            <h3 style={{ fontFamily: "system-ui,sans-serif", fontSize: 20, fontWeight: 700, color: "#fff", marginBottom: 8 }}>How it works — the pipeline</h3>
            <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "rgba(186,230,253,0.9)", lineHeight: 1.7, marginBottom: 0 }}>
              Every step from Encrypt to Decrypt runs on the server — but on locked data. The secret key never leaves your browser. Tap any stage to read what happens inside.
            </p>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "flex-start" }}>
            {pipelineSteps.map(({ label, sub, color, detail }, i) => (
              <div key={label + i} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                <button
                  onClick={() => setOpenPipeline(openPipeline === i ? null : i)}
                  style={{
                    background: openPipeline === i ? color : "#fff",
                    border: `2px solid ${color}`,
                    borderRadius: 8, padding: "10px 14px", cursor: "pointer",
                    textAlign: "center", minWidth: 80, transition: "all 0.15s",
                  }}
                >
                  <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 12, color: openPipeline === i ? "#fff" : color, marginBottom: 3 }}>{label}</div>
                  <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 11, color: openPipeline === i ? "rgba(255,255,255,0.8)" : "#94a3b8" }}>{sub}</div>
                </button>
                {i < pipelineSteps.length - 1 && (
                  <span style={{ color: "#cbd5e1", fontSize: 18, lineHeight: "42px", flexShrink: 0 }}>›</span>
                )}
              </div>
            ))}
          </div>

          {openPipeline !== null && (
            <div style={{ marginTop: 16, background: "#f1f5f9", border: `1.5px solid ${pipelineSteps[openPipeline].color}44`, borderRadius: 8, padding: "16px 20px" }}>
              <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 13, color: pipelineSteps[openPipeline].color, marginBottom: 6 }}>
                {pipelineSteps[openPipeline].label} — {pipelineSteps[openPipeline].sub}
              </div>
              <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#334155", lineHeight: 1.75 }}>
                {pipelineSteps[openPipeline].detail}
              </div>
            </div>
          )}

          <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#64748b", marginTop: 20, lineHeight: 1.7 }}>
            Conv and FC layers account for <strong style={{ color: "#0f172a" }}>&gt;90% of inference time</strong>. The activation function is just one ciphertext multiplication — the bottleneck is linear algebra, not nonlinearity.
          </p>
        </div>

        {/* ── Parameters ── */}
        <div style={{ padding: "48px 0 44px", borderBottom: "1px solid #e2e8f0" }}>
          <div style={{ background: "linear-gradient(135deg,#3b0764,#6d28d9)", borderRadius: 12, padding: "32px 32px 28px", marginBottom: 32 }}>
            <h3 style={{ fontFamily: "system-ui,sans-serif", fontSize: 20, fontWeight: 700, color: "#fff", marginBottom: 8 }}>Key parameters — what they mean and why they matter</h3>
            <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "rgba(221,214,254,0.9)", lineHeight: 1.7, marginBottom: 0 }}>
              FHE has more knobs than a traditional neural network. Tap any parameter to expand the explanation.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {params.map(({ term, value, color, verdict, detail }, i) => (
              <div key={term}
                style={{ background: "#fff", border: `1.5px solid ${openParam === i ? color : "#e2e8f0"}`, borderRadius: 10, overflow: "hidden", transition: "border-color 0.15s" }}>
                <button
                  onClick={() => setOpenParam(openParam === i ? null : i)}
                  style={{ width: "100%", background: "none", border: "none", cursor: "pointer", padding: "16px 20px", display: "flex", alignItems: "center", gap: 16, textAlign: "left" }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 13, color: "#0f172a", marginBottom: 2 }}>{term}</div>
                    <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 12, color: "#64748b" }}>Value used: <strong style={{ color }}>{value}</strong></div>
                  </div>
                  <div style={{ fontFamily: "system-ui,sans-serif", fontSize: 11, fontWeight: 600, color, background: `${color}15`, border: `1px solid ${color}44`, borderRadius: 6, padding: "3px 10px", flexShrink: 0 }}>
                    {verdict}
                  </div>
                  <span style={{ color: "#94a3b8", fontSize: 18, flexShrink: 0, transform: openParam === i ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>›</span>
                </button>
                {openParam === i && (
                  <div style={{ padding: "0 20px 18px", borderTop: `1px solid ${color}22` }}>
                    <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#334155", lineHeight: 1.8, margin: "12px 0 0" }}>{detail}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── Findings ── */}
        <div style={{ padding: "48px 0 0" }}>
          <div style={{ background: "linear-gradient(135deg,#78350f,#b45309)", borderRadius: 12, padding: "32px 32px 28px", marginBottom: 32 }}>
            <h3 style={{ fontFamily: "system-ui,sans-serif", fontSize: 20, fontWeight: 700, color: "#fff", marginBottom: 8 }}>Experiment findings — what worked, what broke, and why</h3>
            <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "rgba(253,230,138,0.9)", lineHeight: 1.7, marginBottom: 0 }}>
              Not everything that was attempted worked. The failures are as informative as the successes.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 16, marginBottom: 28 }}>
            {[
              {
                heading: "What worked",
                color: "#16a34a", bg: "#f0fdf4", border: "#86efac",
                items: [
                  "x² (degree 2) — >80% FHE accuracy end-to-end",
                  "128-bit security on a single r6i.large (16 GB RAM)",
                  "BFV keygen under 5 seconds at n = 4096",
                  "Scale = 1,000, p = 100,073,473 — no overflow",
                  "Full gRPC inference pipeline verified working",
                ],
              },
              {
                heading: "What failed",
                color: "#dc2626", bg: "#fff1f2", border: "#fca5a5",
                items: [
                  "x³ — cubic term overflows plaintext modulus, accuracy collapses",
                  "x⁴ — fourth-power compounds overflow; signal destroyed before FC",
                  "192-bit security — context creation OOM at 7.6 GB + 15 GB swap",
                  "256-bit — never reached; 192-bit was already infeasible",
                  "x⁴ full 100-image run — EC2 budget ran out before completion",
                ],
              },
              {
                heading: "Design decisions",
                color: "#7c3aed", bg: "#faf5ff", border: "#d8b4fe",
                items: [
                  "Activation fixed to x² — the only degree that works in BFV at n = 4096",
                  "Security fixed to 128-bit — the hardware ceiling for one node",
                  "Scale and modulus locked to validated values",
                  "OOM failures framed as feasibility findings, not bugs",
                  "Future work: 4-node EC2 cluster for n = 8192 and 192-bit",
                ],
              },
            ].map(({ heading, color, bg, border, items }) => (
              <div key={heading} style={{ background: bg, border: `1.5px solid ${border}`, borderRadius: 10, padding: "20px 20px 18px" }}>
                <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 14, color, marginBottom: 14 }}>{heading}</div>
                <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
                  {items.map((item) => (
                    <li key={item} style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#334155", lineHeight: 1.75, paddingLeft: 16, position: "relative", marginBottom: 4 }}>
                      <span style={{ position: "absolute", left: 0, color, fontWeight: 700 }}>›</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Limitations callout */}
          <div style={{ background: "#fffbeb", border: "1.5px solid #fcd34d", borderRadius: 10, padding: "20px 24px", marginBottom: 28 }}>
            <div style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 13, color: "#92400e", marginBottom: 8 }}>Known limitations</div>
            <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 13, color: "#78350f", lineHeight: 1.8, margin: 0 }}>
              <strong>x⁴ data is incomplete</strong> — the 100-image run was stopped due to EC2 budget constraints. Do not draw accuracy conclusions from degree-4 results shown in the dashboard.
              The <strong>192-bit and 256-bit OOM failures</strong> are intentional findings, not bugs — they establish the feasibility ceiling for single-node commodity hardware running BFV at n = 4096.
              All results were produced on a single <strong>r6i.large (16 GB RAM)</strong> AWS instance.
            </p>
          </div>

          {/* GitHub link */}
          <div style={{ textAlign: "center", paddingBottom: 8 }}>
            <p style={{ fontFamily: "system-ui,sans-serif", fontSize: 12, color: "#94a3b8", marginBottom: 16 }}>Open Source · Final Year Project 2025–2026</p>
            <a
              href="https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework"
              target="_blank" rel="noreferrer"
              style={{ fontFamily: "system-ui,sans-serif", fontWeight: 700, fontSize: 14, color: "#fff", background: "#0f172a", borderRadius: 8, padding: "12px 28px", display: "inline-block", textDecoration: "none", transition: "background 0.15s" }}
              onMouseEnter={e => { e.currentTarget.style.background = "#1e293b"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "#0f172a"; }}
            >
              View on GitHub
            </a>
          </div>
        </div>

      </div>
    </section>
  );
}
function SkyBanner({ label, title, subtitle }) {
  return (
    <div style={{
      background: "linear-gradient(180deg, #5bb8f5 0%, #82cef7 30%, #b8e4ff 70%, #d6f0ff 100%)",
      borderTop: "4px solid #3a8abf",
      borderBottom: "4px solid #2a6a9f",
      padding: "52px 24px 44px",
      textAlign: "center",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Pixel-art sun glow top-right */}
      <div style={{
        position: "absolute", top: -20, right: "12%",
        width: 90, height: 90,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(255,240,160,0.55) 0%, transparent 70%)",
        pointerEvents: "none",
      }} />

      {/* Cloud shapes */}
      {[
        { top: 14, left: "5%",  w: 80,  h: 28, op: 0.55 },
        { top: 8,  left: "18%", w: 52,  h: 18, op: 0.4  },
        { top: 20, left: "60%", w: 100, h: 34, op: 0.5  },
        { top: 10, left: "78%", w: 60,  h: 22, op: 0.38 },
        { top: 30, left: "88%", w: 40,  h: 14, op: 0.3  },
      ].map((c, i) => (
        <div key={i} style={{
          position: "absolute", top: c.top, left: c.left,
          width: c.w, height: c.h,
          borderRadius: "50%",
          background: "#fff",
          opacity: c.op,
          filter: "blur(4px)",
          pointerEvents: "none",
        }} />
      ))}

      {/* Label */}
      <p style={{
        fontFamily: "'Press Start 2P', monospace",
        fontSize: "0.5rem", letterSpacing: "0.3em",
        color: "#1a5a8a",
        textTransform: "uppercase",
        margin: "0 0 18px",
        textShadow: "0 1px 0 rgba(255,255,255,0.7)",
      }}>
        ✦ {label} ✦
      </p>

      {/* Stardew-style title — golden with thick dark drop shadow */}
      <h2 style={{
        fontFamily: "'Press Start 2P', monospace",
        fontSize: "clamp(1.1rem, 3vw, 1.9rem)",
        letterSpacing: "0.06em",
        lineHeight: 1.45,
        margin: "0 0 20px",
        color: "#f5d020",
        /* layered drop shadows for chunky pixel-art look */
        textShadow: [
          "3px 3px 0 #5a3200",
          "4px 4px 0 #3a1e00",
          "-1px -1px 0 #c8900a",
          "0 0 24px rgba(245,208,32,0.35)",
        ].join(", "),
      }}>
        {title}
      </h2>

      {/* Subtitle */}
      <p style={{
        fontFamily: "system-ui, sans-serif",
        fontSize: "clamp(0.85rem, 1.4vw, 1rem)",
        color: "#0d3a5c",
        maxWidth: 520,
        margin: "0 auto 28px",
        lineHeight: 1.75,
        fontWeight: 500,
        textShadow: "0 1px 0 rgba(255,255,255,0.5)",
      }}>
        {subtitle}
      </p>

      {/* Bouncing pixel arrow */}
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
        style={{ display: "inline-block", opacity: 0.55,
          animation: "bounce 1.6s ease-in-out infinite" }}
      >
        <path d="M12 5v14M5 12l7 7 7-7" stroke="#0d3a5c" strokeWidth="3"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(5px); }
        }
      `}</style>
    </div>
  );
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
