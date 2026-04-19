import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion, useInView } from "framer-motion";
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
        subtitle="Encrypted CNN inference benchmarked across polynomial activation degrees. x² and x³ results are valid. x⁴ experiment was incomplete — see notes."
      />
      <div id="results" className="px-4 md:px-8 py-8" style={{ background: "#f7f7f7", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">

          {/* ── Reproducibility / hardware callout ── */}
          <div className="rounded-lg p-4 mb-4 flex gap-3 text-sm leading-relaxed" style={{ background: "#fffbeb", border: "1.5px solid #f59e0b" }}>
            <span style={{ fontSize: "1.1rem", flexShrink: 0 }}>⚠️</span>
            <div>
              <p className="font-semibold mb-1" style={{ color: "#92400e" }}>Reproducibility note — hardware requirements</p>
              <p style={{ color: "#78350f" }}>
                These benchmarks require an <b>r6i.large instance (≥16 GB RAM)</b> for the Rust compute server — a t3.small
                runs out of memory during BFV key generation at n = 4096. The recommended deployment uses <b>3 EC2 instances</b>:
                frontend + Nginx (t3.small), Spring Boot API (t3.small), and Rust gRPC compute server (r6i.large).
                Due to budget constraints the full 100-image benchmark could not be completed during development.
                The x² and x³ results below are from partial runs and are valid; x⁴ data is <b>incomplete and should not be used for comparison</b>.
              </p>
            </div>
          </div>

          {/* ── What is this ── */}
          <div className="rounded-lg p-4 mb-6 text-sm leading-relaxed" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
            <p className="font-semibold mb-1" style={{ color: "#333" }}>What is this?</p>
            <p style={{ color: "#555" }}>
              MNIST test images were run through the full encrypted CNN pipeline and every layer's timing was recorded.
              The charts compare three <b>polynomial activation degrees</b> (x², x³, x⁴) — a design choice forced by FHE.
              Standard neural networks use ReLU (<code>max(0,x)</code>), which requires a comparison that is impossible on ciphertext.
              Instead, we approximate it with a low-degree polynomial evaluated directly on the encrypted values.
              Degree 2 (x²) is the most stable: it produces the least noise growth, the highest accuracy, and is the only
              configuration fully validated end-to-end on this hardware. Degree 3 shows increased noise but partial accuracy.
              Degree 4 was not reproducible under current resource constraints and its results are marked accordingly.
            </p>
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
      <SkyBanner
        label="Findings"
        title="Parameter Exploration & Limitations"
        subtitle="What was attempted, what worked, what broke — and what each failure revealed about the limits of practical FHE deployment."
      />
      <div className="px-4 md:px-8 py-8" style={{ background: "#fff", borderTop: "1px solid #e5e5e5" }}>
        <div className="max-w-[1100px] mx-auto">

          {/* ── 4-part findings overview ── */}
          <div className="grid grid-cols-2 gap-3 mb-6" style={{ gridTemplateColumns: "1fr 1fr" }}>

            {/* What was attempted */}
            <div className="rounded-lg p-4 text-sm leading-relaxed" style={{ background: "#f8f9ff", border: "1px solid #c7d2fe" }}>
              <p className="font-semibold mb-2 flex items-center gap-2" style={{ color: "#3730a3" }}>
                <span>🧪</span> What was attempted
              </p>
              <ul className="space-y-1 text-xs" style={{ color: "#4338ca" }}>
                <li>• Polynomial activation degrees: <b>x²</b>, <b>x³</b>, <b>x⁴</b></li>
                <li>• Security levels: <b>128-bit</b>, <b>192-bit</b>, <b>256-bit</b></li>
                <li>• Scale factors: <b>100</b>, <b>1,000</b>, <b>10,000</b></li>
                <li>• Plaintext moduli: <b>16-bit</b>, <b>baseline</b>, <b>32-bit</b></li>
                <li>• Full 100-image MNIST inference run per configuration</li>
              </ul>
            </div>

            {/* What succeeded */}
            <div className="rounded-lg p-4 text-sm leading-relaxed" style={{ background: "#f0fdf4", border: "1px solid #86efac" }}>
              <p className="font-semibold mb-2 flex items-center gap-2" style={{ color: "#14532d" }}>
                <span>✅</span> What succeeded
              </p>
              <ul className="space-y-1 text-xs" style={{ color: "#166534" }}>
                <li>• <b>x² (degree 2)</b> — {`>`}80% FHE accuracy, matches training activation exactly</li>
                <li>• <b>128-bit security</b> — keygen in &lt;5s, ~2.5 GB RAM on r6i.large</li>
                <li>• <b>Scale = 1,000</b> — sufficient precision without overflow</li>
                <li>• <b>p = 100,073,473</b> — headroom for intermediate values at scale 1,000</li>
                <li>• End-to-end gRPC inference pipeline verified working</li>
              </ul>
            </div>

            {/* What failed & why */}
            <div className="rounded-lg p-4 text-sm leading-relaxed" style={{ background: "#fff7ed", border: "1px solid #fed7aa" }}>
              <p className="font-semibold mb-2 flex items-center gap-2" style={{ color: "#7c2d12" }}>
                <span>❌</span> What failed & why
              </p>
              <ul className="space-y-1 text-xs" style={{ color: "#9a3412" }}>
                <li>• <b>x³ / x⁴</b> — Higher-degree polynomial terms overflow the plaintext modulus during activation — noise budget consumed faster than BFV can tolerate at n = 4096</li>
                <li>• <b>192-bit / 256-bit</b> — Larger ring dimension required ({">"}n = 8192); context creation OOM at 7.6 GB + 15 GB swap</li>
                <li>• <b>x⁴ full run</b> — 100-image evaluation never completed due to EC2 budget constraints</li>
              </ul>
            </div>

            {/* Design decisions */}
            <div className="rounded-lg p-4 text-sm leading-relaxed" style={{ background: "#fdf4ff", border: "1px solid #e9d5ff" }}>
              <p className="font-semibold mb-2 flex items-center gap-2" style={{ color: "#581c87" }}>
                <span>🎯</span> What this meant for the final design
              </p>
              <ul className="space-y-1 text-xs" style={{ color: "#6b21a8" }}>
                <li>• Fixed activation to <b>x²</b> — the only degree that preserves accuracy end-to-end in BFV</li>
                <li>• Fixed security to <b>128-bit</b> — the hardware ceiling for a single-node r6i.large deployment</li>
                <li>• Locked scale and modulus to values validated at 128-bit, degree 2</li>
                <li>• Framed 192/256-bit OOM as an explicit <b>feasibility boundary finding</b>, not a bug</li>
                <li>• Future work: 4-node EC2 split to enable n = 8192 and 192-bit security</li>
              </ul>
            </div>
          </div>
          <div
            className="rounded-lg p-5"
            style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}
          >
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

/* ─── BookPageAnim ─────────────────────────────────────────────────────────── */
function BookPageAnim({ children, direction = "left" }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.div
      ref={ref}
      style={{ perspective: 1200, transformStyle: "preserve-3d" }}
      initial={{ opacity: 0, rotateY: direction === "left" ? -12 : 12, y: 40 }}
      animate={isInView ? { opacity: 1, rotateY: 0, y: 0 } : {}}
      transition={{ type: "spring", stiffness: 60, damping: 18, mass: 1.2 }}
    >
      {children}
    </motion.div>
  );
}

/* ─── BookGuidebook ─────────────────────────────────────────────────────────── */
function BookGuidebook() {
  return (
    <section id="info" style={{ background: "#e8d5a3", borderTop: "4px solid #b8902a" }}>
      {/* ── Sky-blue About banner ── */}
      <div style={{
        background: "linear-gradient(180deg, #5bb8f5 0%, #82cef7 30%, #b8e4ff 70%, #d6f0ff 100%)",
        borderTop: "4px solid #3a8abf",
        borderBottom: "4px solid #2a6a9f",
        padding: "52px 24px 44px",
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
      }}>
        {[
          { top: "18%", left: "6%", w: 90, h: 28, op: 0.55 },
          { top: "38%", left: "22%", w: 60, h: 20, op: 0.4 },
          { top: "12%", right: "8%", w: 110, h: 32, op: 0.5 },
          { top: "50%", right: "18%", w: 70, h: 22, op: 0.38 },
        ].map((c, i) => (
          <div key={i} style={{
            position: "absolute", top: c.top, left: c.left, right: c.right,
            width: c.w, height: c.h, background: "rgba(255,255,255,0.9)",
            borderRadius: 999, opacity: c.op,
          }} />
        ))}
        <div style={{ position: "relative", zIndex: 2 }}>
          <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: "clamp(0.55rem,1.4vw,0.75rem)", letterSpacing: "0.25em", textTransform: "uppercase", color: "#1a4a6e", marginBottom: 14, opacity: 0.75 }}>
            About This Project
          </div>
          <h2 style={{ fontFamily: "'Press Start 2P', monospace", fontSize: "clamp(1.1rem,3.5vw,2rem)", color: "#1a2e1a", textShadow: "3px 3px 0 #b8d4f0, 5px 5px 0 #8ab8e0", lineHeight: 1.4, marginBottom: 16 }}>
            The Guidebook
          </h2>
          <p style={{ fontFamily: "system-ui, sans-serif", fontSize: "clamp(0.85rem,1.8vw,1.05rem)", color: "#1a3a5e", maxWidth: 560, margin: "0 auto", opacity: 0.85 }}>
            Everything you need to understand this demo — the hook, the how, and the honest limitations.
          </p>
        </div>
      </div>

      {/* ── Book wrapper ── */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px 64px" }}>
        <style>{`
          .book-spread { display:grid; grid-template-columns:1fr 1fr; gap:0; background:#f5e6c0; border:4px solid #8b6914; box-shadow:8px 8px 0 #5a4208,12px 12px 0 rgba(0,0,0,0.25); margin-bottom:36px; position:relative; }
          .book-spread::after { content:''; position:absolute; top:0; bottom:0; left:50%; width:6px; margin-left:-3px; background:linear-gradient(90deg,#8b6914 0%,#c8a84b 40%,#a07820 60%,#5a4208 100%); box-shadow:0 0 12px rgba(0,0,0,0.3); z-index:10; }
          .book-page { padding:40px 44px; background:#fdf6e3; position:relative; overflow:hidden; }
          .book-page-right { background:#faf0d0; }
          .book-page::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#d4a830 0%,#f0c84a 50%,#d4a830 100%); }
          .book-page-right::before { background:linear-gradient(90deg,#c8a030 0%,#e8c040 50%,#c8a030 100%); }
          .pixel-chapter { font-family:'Press Start 2P',monospace; font-size:0.55rem; letter-spacing:0.2em; text-transform:uppercase; color:#8b6914; margin-bottom:10px; }
          .pixel-heading { font-family:'Georgia',serif; font-size:1.6rem; color:#3a2008; margin-bottom:18px; line-height:1.3; font-style:italic; }
          .book-body { font-family:'Georgia','Times New Roman',serif; font-size:0.9rem; line-height:1.85; color:#3a2c10; }
          .pixel-tag { display:inline-block; font-family:'Press Start 2P',monospace; font-size:0.48rem; padding:3px 8px; border:2px solid currentColor; box-shadow:2px 2px 0 rgba(0,0,0,0.2); letter-spacing:0.1em; vertical-align:middle; }
          .book-single { background:#fdf6e3; border:4px solid #8b6914; box-shadow:8px 8px 0 #5a4208,12px 12px 0 rgba(0,0,0,0.25); padding:40px 52px; margin-bottom:36px; position:relative; overflow:hidden; }
          .book-single::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#d4a830 0%,#f0c84a 50%,#d4a830 100%); }
          .pipeline-arrow { font-family:'Press Start 2P',monospace; font-size:0.6rem; color:#8b6914; padding:0 6px; flex-shrink:0; }
          .glossary-card { background:#fff8e8; border:2px solid #c8a030; box-shadow:3px 3px 0 #8b6914; padding:14px 16px; cursor:pointer; transition:transform 0.15s,box-shadow 0.15s; position:relative; }
          .glossary-card:hover { transform:translate(-2px,-2px); box-shadow:5px 5px 0 #8b6914; }
          .findings-callout { background:#fffbe6; border:3px solid #d4a830; box-shadow:5px 5px 0 #8b6914; padding:22px 28px; position:relative; }
          @media (max-width:680px) { .book-spread { grid-template-columns:1fr; } .book-spread::after { display:none; } .book-page { padding:28px 22px; } }
        `}</style>

        {/* Pages 1-2 */}
        <BookPageAnim direction="left">
          <div className="book-spread">
            <div className="book-page">
              <div className="pixel-chapter">Chapter I</div>
              <div className="pixel-heading">The Wonders of Private AI</div>
              <div className="book-body">
                <p style={{ marginBottom: 16 }}><strong>What if an AI could recognise your handwriting without ever seeing it?</strong></p>
                <p style={{ marginBottom: 16 }}>Every time you use a cloud AI service — from face recognition to medical diagnosis — your raw data travels to a server that can read it. Homomorphic Encryption flips this entirely: the server runs the AI <em>inside the lock</em>, returning only the answer.</p>
                <p style={{ marginBottom: 20 }}>This demo makes that real. You draw a digit. We encrypt it. A neural network runs on the ciphertext. You get a prediction. The server never saw your pixels.</p>
                <div style={{ background:"#f0e4b8", border:"2px solid #c8a030", boxShadow:"3px 3px 0 #8b6914", padding:"14px 18px", marginBottom:20, fontStyle:"italic", fontSize:"0.88rem", color:"#5a3a08" }}>
                  "Computation on data you cannot read — this is the promise of Fully Homomorphic Encryption."
                </div>
                <p style={{ fontSize:"0.82rem", color:"#6a5020" }}>Three libraries are pitted against each other — OpenFHE, Microsoft SEAL, and IBM HElib — on the exact same CNN task. Who is fastest? Who uses the least memory? Read on.</p>
              </div>
              <div style={{ position:"absolute", bottom:16, left:44, fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>1</div>
            </div>
            <div className="book-page book-page-right">
              <div className="pixel-chapter">Chapter II</div>
              <div className="pixel-heading">What You Can Do Here</div>
              <div className="book-body">
                {[
                  { title:"Draw & Predict", desc:"Sketch a digit (0–9) on the canvas. Watch it encrypt and feed through a live CNN — all on the server, all encrypted." },
                  { title:"Compare Libraries", desc:"Run OpenFHE, SEAL, and HElib on the same 5 cryptographic operations and see timing results side by side." },
                  { title:"Explore Parameters", desc:"See what happens when you change polynomial degree, security level, or scale factor — and why some combinations break." },
                  { title:"Read the Findings", desc:"Browse real benchmark data from 10–100 MNIST images across 3 activation degrees and 3 security levels." },
                ].map(({ title, desc }) => (
                  <div key={title} style={{ display:"flex", gap:14, marginBottom:18, paddingBottom:18, borderBottom:"1px dashed #c8a030" }}>
                    <div>
                      <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.52rem", color:"#6b4a10", marginBottom:5, letterSpacing:"0.08em" }}>{title}</div>
                      <div style={{ fontSize:"0.86rem", color:"#4a3010", lineHeight:1.7 }}>{desc}</div>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ position:"absolute", bottom:16, right:44, fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>2</div>
            </div>
          </div>
        </BookPageAnim>

        {/* Page 3 */}
        <BookPageAnim direction="right">
          <div className="book-single">
            <div className="pixel-chapter">Chapter III</div>
            <div className="pixel-heading" style={{ textAlign:"center" }}>How It Works — The Pipeline</div>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center", flexWrap:"wrap", gap:8, margin:"28px 0" }}>
              {[
                { label:"Draw", sub:"28×28 pixels", color:"#0aa35e" },
                { label:"Encrypt", sub:"BFV ciphertext", color:"#0db7c4" },
                { label:"Conv1", sub:"5×5 kernel", color:"#7b3ff2" },
                { label:"Activate", sub:"x² (no ReLU)", color:"#e68a00" },
                { label:"Pool", sub:"avg 2×2", color:"#f4b942" },
                { label:"Conv2", sub:"5×5 kernel", color:"#9b6dff" },
                { label:"Activate x2", sub:"x² again", color:"#ff9f43" },
                { label:"FC Layer", sub:"10 logits", color:"#e03e52" },
                { label:"Decrypt", sub:"secret key", color:"#0db7c4" },
                { label:"Result", sub:"predicted digit", color:"#0aa35e" },
              ].map(({ label, sub, color }, i, arr) => (
                <div key={label} style={{ display:"flex", alignItems:"center", gap:8 }}>
                  <div style={{ background:"#fff8e8", border:`3px solid ${color}`, boxShadow:`3px 3px 0 ${color}66`, padding:"10px 14px", textAlign:"center", minWidth:72 }}>
                    <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.48rem", color, letterSpacing:"0.06em" }}>{label}</div>
                    <div style={{ fontFamily:"system-ui", fontSize:"0.65rem", color:"#8b6020", marginTop:3 }}>{sub}</div>
                  </div>
                  {i < arr.length - 1 && <div className="pipeline-arrow">▶</div>}
                </div>
              ))}
            </div>
            <div style={{ fontFamily:"Georgia,serif", fontSize:"0.88rem", color:"#5a3a08", textAlign:"center", lineHeight:1.7, maxWidth:700, margin:"0 auto" }}>
              Every step from Encrypt to Decrypt runs on the server — but on <strong>locked data</strong>. The secret key never leaves your session. Conv and FC layers account for &gt;90% of inference time; the activation function is just one ciphertext multiply.
            </div>
            <div style={{ position:"absolute", bottom:16, left:"50%", transform:"translateX(-50%)", fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>3</div>
          </div>
        </BookPageAnim>

        {/* Pages 4-5 */}
        <BookPageAnim direction="left">
          <div className="book-spread">
            <div className="book-page">
              <div className="pixel-chapter">Chapter IV</div>
              <div className="pixel-heading">Parameter Glossary</div>
              <div className="book-body" style={{ fontSize:"0.85rem" }}>
                <p style={{ marginBottom:18, color:"#6a5020" }}>Hover each card to see why it matters — no lecture required.</p>
                {[
                  { term:"Polynomial Degree", symbol:"x² / x³ / x⁴", color:"#7b3ff2", tip:"Replaces ReLU in FHE. Degree 2 (x²) is the only one that preserves accuracy end-to-end — higher degrees overflow the plaintext modulus.", verdict:"Use x²" },
                  { term:"Security Level", symbol:"128 / 192 / 256-bit", color:"#0db7c4", tip:"Sets the hardness of the encryption. 128-bit works on a single r6i.large. 192-bit requires >7.6 GB RAM — it OOM'd on every attempt.", verdict:"Use 128-bit" },
                  { term:"Ring Dimension n", symbol:"n = 4096", color:"#e68a00", tip:"The size of the polynomial ring. Larger n = more security + more memory. n=4096 is the minimum for 128-bit BFV with these parameters.", verdict:"Fixed at 4096" },
                ].map(({ term, symbol, color, tip, verdict }) => (
                  <div key={term} className="glossary-card" style={{ marginBottom:14 }} title={tip}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:6 }}>
                      <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.5rem", color, letterSpacing:"0.08em" }}>{term}</div>
                      <span className="pixel-tag" style={{ color, borderColor:color }}>{symbol}</span>
                    </div>
                    <div style={{ fontSize:"0.8rem", color:"#5a3a08", lineHeight:1.6, marginBottom:8 }}>{tip}</div>
                    <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.42rem", color:"#0aa35e" }}>{verdict}</div>
                  </div>
                ))}
              </div>
              <div style={{ position:"absolute", bottom:16, left:44, fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>4</div>
            </div>
            <div className="book-page book-page-right">
              <div className="pixel-chapter">Chapter IV cont.</div>
              <div className="pixel-heading">More Parameters</div>
              <div className="book-body" style={{ fontSize:"0.85rem" }}>
                {[
                  { term:"Plaintext Modulus p", symbol:"100,073,473", color:"#e03e52", tip:"The 'ceiling' for intermediate values during computation. Too small and higher-degree activations overflow, corrupting the result silently.", verdict:"Baseline = safest" },
                  { term:"Scale Factor S", symbol:"S = 1,000", color:"#0aa35e", tip:"Multiplies weights before encoding to preserve decimal precision. S=100 loses too much precision; S=10,000 risks overflow. S=1,000 is the sweet spot.", verdict:"S = 1,000" },
                  { term:"Multiplication Depth", symbol:"depth = 6", color:"#f4b942", tip:"The number of sequential ciphertext multiplications the scheme can handle before noise overwhelms the signal. Our CNN uses depth 6 (conv × act × conv × act × fc × spare).", verdict:"Fixed at 6" },
                  { term:"Noise Budget", symbol:"BFV noise", color:"#9b6dff", tip:"BFV tracks a 'noise budget' that decreases with each operation. When it hits zero, decryption fails. Larger n and p give more budget — at the cost of memory.", verdict:"Managed by library" },
                ].map(({ term, symbol, color, tip, verdict }) => (
                  <div key={term} className="glossary-card" style={{ marginBottom:14 }} title={tip}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:6 }}>
                      <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.5rem", color, letterSpacing:"0.08em" }}>{term}</div>
                      <span className="pixel-tag" style={{ color, borderColor:color }}>{symbol}</span>
                    </div>
                    <div style={{ fontSize:"0.8rem", color:"#5a3a08", lineHeight:1.6, marginBottom:8 }}>{tip}</div>
                    <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.42rem", color:"#0aa35e" }}>{verdict}</div>
                  </div>
                ))}
              </div>
              <div style={{ position:"absolute", bottom:16, right:44, fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>5</div>
            </div>
          </div>
        </BookPageAnim>

        {/* Page 6 */}
        <BookPageAnim direction="right">
          <div className="book-single">
            <div className="pixel-chapter">Chapter V</div>
            <div className="pixel-heading" style={{ textAlign:"center" }}>Experiment Findings</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:20, marginBottom:32 }}>
              {[
                { label:"What Worked", color:"#0aa35e", items:["x² (degree 2) — >80% FHE accuracy","128-bit security on r6i.large","Full gRPC inference pipeline","Scale=1000, p=100,073,473"] },
                { label:"What Failed", color:"#e03e52", items:["x³ — modular overflow → accuracy collapse","192-bit security → OOM (>7.6 GB)","256-bit → never attempted","x⁴ 100-image run → budget ran out"] },
                { label:"Design Decisions", color:"#7b3ff2", items:["Fixed activation to x²","Fixed security to 128-bit","Single-node r6i.large deployment","Future: 4-node EC2 for 192-bit"] },
              ].map(({ label, color, items }) => (
                <div key={label} style={{ background:"#fff8e8", border:`3px solid ${color}`, boxShadow:`4px 4px 0 ${color}55`, padding:"18px 20px" }}>
                  <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.5rem", color, letterSpacing:"0.08em", marginBottom:12 }}>{label}</div>
                  <ul style={{ margin:0, padding:0, listStyle:"none" }}>
                    {items.map((item) => (
                      <li key={item} style={{ fontFamily:"Georgia,serif", fontSize:"0.8rem", color:"#4a3010", lineHeight:1.7, paddingLeft:14, position:"relative" }}>
                        <span style={{ position:"absolute", left:0, color }}>›</span>{item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className="findings-callout">
              <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.55rem", color:"#8b6914", marginBottom:12, letterSpacing:"0.1em" }}>Known Limitations</div>
              <div style={{ fontFamily:"Georgia,serif", fontSize:"0.88rem", color:"#5a3a08", lineHeight:1.8 }}>
                <strong>x⁴ data is incomplete</strong> — the 100-image run was stopped due to EC2 budget constraints. Do not draw accuracy conclusions from degree-4 results.
                The <strong>192-bit and 256-bit OOM failures</strong> are intentional findings, not bugs — they prove the feasibility ceiling of single-node commodity hardware for BFV at n = 4096.
                All results run on a single <strong>r6i.large (16 GB RAM)</strong> on AWS.
              </div>
            </div>
            <div style={{ marginTop:28, textAlign:"center" }}>
              <div style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.52rem", color:"#8b6914", marginBottom:10, letterSpacing:"0.1em" }}>Open Source · Final Year Project 2025–2026</div>
              <a
                href="https://github.com/TiffanyYongNgikChee/Encrypted-Machine-Learning-Benchmark-Framework"
                target="_blank" rel="noreferrer"
                style={{ fontFamily:"'Press Start 2P',monospace", fontSize:"0.52rem", color:"#fff", background:"#3a2008", border:"3px solid #8b6914", boxShadow:"4px 4px 0 #5a4208", padding:"10px 20px", display:"inline-block", textDecoration:"none", letterSpacing:"0.1em", transition:"transform 0.1s,box-shadow 0.1s" }}
                onMouseEnter={e => { e.currentTarget.style.transform="translate(-2px,-2px)"; e.currentTarget.style.boxShadow="6px 6px 0 #5a4208"; }}
                onMouseLeave={e => { e.currentTarget.style.transform=""; e.currentTarget.style.boxShadow="4px 4px 0 #5a4208"; }}
              >View on GitHub</a>
            </div>
            <div style={{ position:"absolute", bottom:16, left:"50%", transform:"translateX(-50%)", fontFamily:"'Press Start 2P',monospace", fontSize:"0.45rem", color:"#c8a030" }}>6</div>
          </div>
        </BookPageAnim>

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
