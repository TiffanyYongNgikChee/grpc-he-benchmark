import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import CnnPipeline, { LAYERS, CATEGORY_COLORS } from "./CnnPipeline";
import MiniCanvas from "./MiniCanvas";
import OutputPanel from "./OutputPanel";
import MetricsStrip from "./MetricsStrip";
import { checkHealth, predictDigit } from "../api/client";

/**
 * Playground — Single-page interactive visualization of the encrypted CNN
 * inference pipeline, inspired by TensorFlow Playground.
 *
 * Layout (top → bottom):
 *   1. Header bar — title + tagline + health dot
 *   2. Top controls — play/run button, library info, epoch counter
 *   3. Main area (3 columns):  Input | CNN Pipeline | Output
 *   4. Bottom metrics — layer timing breakdown + hover tooltip
 */
export default function Playground() {
  /* ─── State ─── */
  const [pixels, setPixels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [healthy, setHealthy] = useState(null);

  /* Animation state */
  const [activeStep, setActiveStep] = useState(-1);    // -1 = idle
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const [runCount, setRunCount] = useState(0);
  const animTimerRef = useRef(null);

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
      const response = await predictDigit(pixels, 1000);
      setResult(response);
      setRunCount((c) => c + 1);

      /* Animate layers sequentially using real timing data */
      let elapsed = 0;
      const timingKeys = LAYERS.map((l) => l.key);
      clearTimeout(animTimerRef.current);

      for (let step = 1; step <= LAYERS.length; step++) {
        const key = timingKeys[step - 1];
        const ms = key && response[key] ? response[key] : 60; // default 60ms visual delay
        elapsed += Math.min(ms * 1.5, 500); // scale for visual effect, cap at 500ms

        animTimerRef.current = setTimeout(() => {
          setActiveStep(step);
        }, elapsed);
      }
    } catch (err) {
      setError(err.message);
      setActiveStep(-1);
    } finally {
      setLoading(false);
    }
  }, [pixels, loading]);

  /* Cleanup timers */
  useEffect(() => {
    return () => clearTimeout(animTimerRef.current);
  }, []);

  /* ─── Hover tooltip data ─── */
  const tooltipLayer = hoveredLayer !== null ? LAYERS[hoveredLayer] : null;
  const tooltipTime =
    tooltipLayer && tooltipLayer.key && result
      ? result[tooltipLayer.key]
      : null;

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      {/* ═══════════ HEADER ═══════════ */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-3">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🔐</span>
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">
                Encrypted ML Playground
              </h1>
              <p className="text-xs text-slate-400 leading-tight">
                Tinker with <b className="text-emerald-400">homomorphic encryption</b> on a real CNN. Don't worry, your data stays encrypted.
              </p>
            </div>
          </div>

          {/* Health dot */}
          <div className="flex items-center gap-2">
            <div
              className={`w-2.5 h-2.5 rounded-full transition-colors ${
                healthy === null
                  ? "bg-slate-500"
                  : healthy
                  ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
                  : "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]"
              }`}
            />
            <span className="text-xs text-slate-500">
              {healthy === null ? "Checking…" : healthy ? "Backend Online" : "Backend Offline"}
            </span>
          </div>
        </div>
      </header>

      {/* ═══════════ TOP CONTROLS ═══════════ */}
      <div className="bg-slate-800/50 border-b border-slate-700/50 px-6 py-3">
        <div className="max-w-[1400px] mx-auto flex items-center gap-6">
          {/* Run button — big and colorful like TF playground's play button */}
          <button
            onClick={handleRun}
            disabled={!pixels || loading}
            className={`
              w-12 h-12 rounded-full flex items-center justify-center
              text-white font-bold text-xl
              transition-all shadow-lg
              ${
                !pixels || loading
                  ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                  : "bg-emerald-500 hover:bg-emerald-400 hover:shadow-emerald-500/30 hover:scale-105 active:scale-95"
              }
            `}
            title={loading ? "Running…" : "Run Encrypted Inference"}
          >
            {loading ? (
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          {/* Run count */}
          <div className="text-sm">
            <span className="text-slate-500">Epoch</span>{" "}
            <span className="text-white font-mono font-semibold text-lg">
              {String(runCount).padStart(3, "0")}
            </span>
          </div>

          {/* Divider */}
          <div className="w-px h-8 bg-slate-700" />

          {/* Library info */}
          <div>
            <label className="text-xs text-slate-500 block">Library</label>
            <span className="text-sm text-cyan-400 font-medium">OpenFHE (BFV)</span>
          </div>

          <div className="w-px h-8 bg-slate-700" />

          {/* Encryption scheme */}
          <div>
            <label className="text-xs text-slate-500 block">Scheme</label>
            <span className="text-sm text-violet-400 font-medium">BFV</span>
          </div>

          <div className="w-px h-8 bg-slate-700" />

          {/* Model */}
          <div>
            <label className="text-xs text-slate-500 block">Model</label>
            <span className="text-sm text-amber-400 font-medium">LeNet-5 CNN</span>
          </div>

          <div className="w-px h-8 bg-slate-700" />

          {/* Scale factor */}
          <div>
            <label className="text-xs text-slate-500 block">Scale factor</label>
            <span className="text-sm text-slate-300 font-mono">1000</span>
          </div>

          {/* Spacer + total time */}
          <div className="ml-auto text-right">
            {result && (
              <>
                <label className="text-xs text-slate-500 block">Total inference</label>
                <span className="text-sm text-emerald-400 font-mono font-semibold">
                  {result.totalMs?.toFixed(1)}ms
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ═══════════ MAIN AREA ═══════════ */}
      <div className="flex-1 px-6 py-4">
        <div className="max-w-[1400px] mx-auto flex gap-4 h-full" style={{ minHeight: 480 }}>

          {/* ── INPUT COLUMN ── */}
          <div className="w-52 flex-shrink-0">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Input
            </h4>
            <p className="text-[11px] text-slate-500 mb-3">
              Draw a digit (0-9) below, then click the play button.
            </p>
            <MiniCanvas onPixelsReady={setPixels} disabled={loading} />

            {/* 28×28 preview */}
            {pixels && (
              <div className="mt-3">
                <p className="text-[10px] text-slate-500 mb-1">28×28 model input</p>
                <canvas
                  width={28}
                  height={28}
                  className="border border-slate-600 rounded"
                  style={{ width: 56, height: 56, imageRendering: "pixelated" }}
                  ref={(el) => {
                    if (!el || !pixels) return;
                    const ctx = el.getContext("2d");
                    const img = ctx.createImageData(28, 28);
                    for (let i = 0; i < 784; i++) {
                      const v = pixels[i];
                      img.data[i * 4] = v;
                      img.data[i * 4 + 1] = v;
                      img.data[i * 4 + 2] = v;
                      img.data[i * 4 + 3] = 255;
                    }
                    ctx.putImageData(img, 0, 0);
                  }}
                />
              </div>
            )}

            {/* Colour legend */}
            <div className="mt-5">
              <p className="text-[10px] text-slate-500 mb-2 font-medium">Layer colours</p>
              {[
                ["Encrypt / Decrypt", "crypto"],
                ["Conv / Bias",       "conv"],
                ["ReLU / Pool",       "act"],
                ["FC Layer",          "fc"],
              ].map(([label, cat]) => (
                <div key={cat} className="flex items-center gap-2 mb-1">
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: CATEGORY_COLORS[cat].active }}
                  />
                  <span className="text-[10px] text-slate-400">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── NETWORK COLUMN (CNN Pipeline) ── */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Column headers like TF playground */}
            <div className="flex items-center mb-2">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Encrypted CNN Pipeline
              </h4>
              <span className="ml-2 text-[10px] text-slate-600">
                ({LAYERS.length} layers)
              </span>
            </div>

            {/* The SVG pipeline */}
            <div className="bg-slate-800/40 rounded-lg border border-slate-700/50 p-3 flex-1 flex items-center">
              <CnnPipeline
                timings={result}
                activeStep={activeStep}
                hovered={hoveredLayer}
                onHover={setHoveredLayer}
              />
            </div>

            {/* Hover tooltip */}
            <AnimatePresence>
              {tooltipLayer && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 5 }}
                  className="mt-2 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-xs"
                >
                  <span className="font-semibold text-white">{tooltipLayer.label}</span>
                  <span className="text-slate-500 ml-2">{tooltipLayer.sub}</span>
                  {tooltipTime !== null && (
                    <span className="ml-3 text-emerald-400 font-mono">
                      {tooltipTime.toFixed(2)}ms
                      <span className="text-slate-500 ml-1">
                        ({((tooltipTime / result.totalMs) * 100).toFixed(1)}%)
                      </span>
                    </span>
                  )}
                  <span className="ml-3 text-slate-600">
                    {getLayerDescription(tooltipLayer.id)}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ── OUTPUT COLUMN ── */}
          <div className="w-64 flex-shrink-0">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Output
            </h4>
            <OutputPanel result={result} error={error} loading={loading} pixels={pixels} />
          </div>
        </div>
      </div>

      {/* ═══════════ BOTTOM METRICS ═══════════ */}
      <div className="border-t border-slate-700/50 bg-slate-800/30">
        <div className="max-w-[1400px] mx-auto px-6 py-3">
          <MetricsStrip result={result} />
        </div>
      </div>
    </div>
  );
}

/* ── Layer descriptions for hover tooltip ── */
function getLayerDescription(id) {
  const map = {
    input:   "Raw 28×28 pixel image in plaintext",
    encrypt: "Encode & encrypt using BFV scheme (OpenFHE)",
    conv1:   "Encrypted 5×5 convolution, 5 output channels",
    bias1:   "Add encrypted bias vector to conv1 output",
    relu1:   "Square activation (x² approximation of ReLU)",
    pool1:   "2×2 average pooling on encrypted data",
    conv2:   "Encrypted 5×5 convolution, 10 output channels",
    bias2:   "Add encrypted bias vector to conv2 output",
    relu2:   "Square activation (x² approximation of ReLU)",
    pool2:   "2×2 average pooling on encrypted data",
    fc:      "Encrypted fully-connected layer → 10 logits",
    biasfc:  "Add encrypted bias vector to FC output",
    decrypt: "Decrypt BFV ciphertext back to plaintext",
    output:  "Softmax + argmax to get predicted digit",
  };
  return map[id] || "";
}
