import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import CnnPipeline, { LAYERS } from "./CnnPipeline";
import MiniCanvas from "./MiniCanvas";
import OutputPanel from "./OutputPanel";
import MetricsStrip from "./MetricsStrip";
import { checkHealth, predictDigit } from "../api/client";

/**
 * Playground — TensorFlow-Playground-inspired single-page layout.
 *
 * Visual hierarchy (top → bottom):
 *   1. Orange header — hero text + tagline
 *   2. Controls bar — play, reset, epoch, dropdowns (scheme, model, scale)
 *   3. Main area — INPUT | CNN PIPELINE (vertical groups) | OUTPUT
 *   4. Bottom metrics — timing stacked bar
 *   5. Info section — "What is HE?" + "About this project"
 */
export default function Playground() {
  /* ─── State ─── */
  const [pixels, setPixels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [healthy, setHealthy] = useState(null);

  /* Animation */
  const [activeStep, setActiveStep] = useState(-1);
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const [runCount, setRunCount] = useState(0);
  const animTimers = useRef([]);

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

      /* Animate layers sequentially */
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
    } finally {
      setLoading(false);
    }
  }, [pixels, loading]);

  const handleReset = () => {
    setResult(null);
    setError(null);
    setActiveStep(-1);
  };

  useEffect(() => () => animTimers.current.forEach(clearTimeout), []);

  /* Tooltip */
  const tl = hoveredLayer !== null ? LAYERS[hoveredLayer] : null;
  const tTime = tl && tl.key && result ? result[tl.key] : null;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#f7f7f7" }}>

      {/* ═══════════ HEADER ═══════════ */}
      <header
        className="text-center py-8 px-4"
        style={{ background: "#1a1a2e" }}
      >
        <h1 className="text-white text-xl md:text-2xl font-light tracking-wide leading-snug">
          Tinker With{" "}
          <span className="font-bold">Encrypted Neural Networks</span>{" "}
          Right Here in Your Browser.
        </h1>
        <p className="text-sm mt-1 font-light" style={{ color: "rgba(255,255,255,0.65)" }}>
          Don't Worry, Your Data Stays <b className="text-white">Encrypted</b>. We Promise.
        </p>
      </header>

      {/* ═══════════ CONTROLS BAR ═══════════ */}
      <div
        className="border-b flex items-center gap-1 md:gap-0 px-4 py-2 flex-wrap"
        style={{ background: "#ebeaea", borderColor: "#d9d9d9" }}
      >
        <div className="flex items-center gap-2 mr-4 md:mr-6">
          {/* Reset */}
          <button
            onClick={handleReset}
            className="w-9 h-9 rounded-full flex items-center justify-center border border-gray-300 bg-white hover:bg-gray-100 text-gray-500 transition-colors"
            title="Reset"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M5.09 15A8 8 0 0119 9M18.91 9A8 8 0 015 15" />
            </svg>
          </button>

          {/* Play */}
          <button
            onClick={handleRun}
            disabled={!pixels || loading}
            className={`w-11 h-11 rounded-full flex items-center justify-center text-white shadow transition-all ${
              !pixels || loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-[#f4743a] hover:bg-[#e05a20] hover:scale-105 active:scale-95"
            }`}
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

          {/* Step */}
          <button
            onClick={handleRun}
            disabled={!pixels || loading}
            className="w-9 h-9 rounded-full flex items-center justify-center border border-gray-300 bg-white hover:bg-gray-100 text-gray-500 disabled:opacity-40 transition-colors"
            title="Step"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4v16l10-8z" /><rect x="16" y="4" width="3" height="16" rx="0.5" />
            </svg>
          </button>
        </div>

        {/* Epoch */}
        <ControlLabel label="Epoch" value={String(runCount).padStart(6, "0")} mono />

        <Divider />

        {/* Dropdowns */}
        <ControlDropdown label="Library" options={["OpenFHE"]} />
        <ControlDropdown label="Encryption" options={["BFV"]} />
        <ControlDropdown label="Activation" options={["x² (Square)"]} />
        <ControlDropdown label="Scale factor" options={["1000"]} />
        <ControlDropdown label="Model" options={["CNN (LeNet-5)"]} />

        {/* Spacer + health + total time */}
        <div className="ml-auto flex items-center gap-3">
          {result && (
            <span className="text-xs font-medium" style={{ color: "#0aa35e" }}>
              {result.totalMs?.toFixed(1)}ms total
            </span>
          )}
          <div className="flex items-center gap-1.5">
            <div
              className="w-2 h-2 rounded-full"
              style={{
                background: healthy === null ? "#aaa" : healthy ? "#0aa35e" : "#e03e52",
                boxShadow: healthy ? "0 0 6px rgba(10,163,94,0.5)" : "none",
              }}
            />
            <span className="text-[11px]" style={{ color: "#999" }}>
              {healthy === null ? "…" : healthy ? "Online" : "Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* ═══════════ MAIN AREA ═══════════ */}
      <div className="flex-1 px-4 md:px-8 py-6">
        <div className="max-w-[1440px] mx-auto flex gap-6" style={{ minHeight: 420 }}>

          {/* ── INPUT COLUMN ── */}
          <div className="flex-shrink-0" style={{ width: 210 }}>
            <SectionTitle>INPUT</SectionTitle>
            <p className="text-xs mb-3" style={{ color: "#999" }}>
              Draw a digit (0-9)
            </p>
            <MiniCanvas onPixelsReady={setPixels} disabled={loading} />

            {/* 28×28 preview */}
            {pixels && (
              <div className="mt-3">
                <p className="text-[10px] mb-1" style={{ color: "#aaa" }}>28×28 model input</p>
                <canvas
                  width={28}
                  height={28}
                  className="rounded"
                  style={{
                    width: 56,
                    height: 56,
                    imageRendering: "pixelated",
                    border: "1px solid #d9d9d9",
                  }}
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

            <div className="mt-5 text-[10px]" style={{ color: "#999" }}>
              <p className="font-medium text-[11px] mb-1" style={{ color: "#666" }}>
                Layer colours
              </p>
              {[
                ["Crypto", "#0db7c4"],
                ["Conv / Bias", "#7b3ff2"],
                ["Activation / Pool", "#e68a00"],
                ["Fully Connected", "#e03e52"],
                ["I/O", "#0aa35e"],
              ].map(([label, color]) => (
                <div key={label} className="flex items-center gap-2 mb-0.5">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── NETWORK COLUMN ── */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center gap-2 mb-2">
              <SectionTitle>ENCRYPTED CNN PIPELINE</SectionTitle>
              <span className="text-[10px]" style={{ color: "#bbb" }}>
                {LAYERS.length} layers
              </span>
            </div>

            {/* Pipeline card */}
            <div
              className="flex-1 rounded-lg p-3 flex items-center"
              style={{ background: "#fff", border: "1px solid #d9d9d9" }}
            >
              <CnnPipeline
                timings={result}
                activeStep={activeStep}
                hovered={hoveredLayer}
                onHover={setHoveredLayer}
              />
            </div>

            {/* Hover tooltip */}
            <AnimatePresence>
              {tl && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mt-2 rounded-lg px-4 py-2 text-xs"
                  style={{ background: "#fff", border: "1px solid #d9d9d9" }}
                >
                  <span className="font-medium" style={{ color: "#333" }}>{tl.label}</span>
                  <span className="ml-2" style={{ color: "#999" }}>{tl.sub}</span>
                  {tTime !== null && (
                    <span className="ml-3 font-mono" style={{ color: "#0aa35e" }}>
                      {tTime.toFixed(2)}ms
                      <span className="ml-1" style={{ color: "#bbb" }}>
                        ({((tTime / result.totalMs) * 100).toFixed(1)}%)
                      </span>
                    </span>
                  )}
                  <span className="ml-3" style={{ color: "#999" }}>
                    — {getLayerDescription(tl.id)}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ── OUTPUT COLUMN ── */}
          <div className="flex-shrink-0" style={{ width: 240 }}>
            <SectionTitle>OUTPUT</SectionTitle>
            <OutputPanel result={result} error={error} loading={loading} pixels={pixels} />
          </div>
        </div>
      </div>

      {/* ═══════════ METRICS BAR ═══════════ */}
      <div style={{ background: "#ebeaea", borderTop: "1px solid #d9d9d9" }}>
        <div className="max-w-[1440px] mx-auto px-4 md:px-8 py-3">
          <MetricsStrip result={result} />
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
        <div className="max-w-2xl mx-auto space-y-10 text-[15px] leading-relaxed" style={{ color: "#555" }}>
          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              What Is Homomorphic Encryption?
            </h2>
            <p>
              Homomorphic encryption (HE) lets you <b>compute on encrypted data</b> without
              ever decrypting it. The server never sees your raw input — it performs the
              entire CNN inference on ciphertext. Only you, the data owner, can decrypt
              the result. This project uses the{" "}
              <b style={{ color: "#0db7c4" }}>BFV scheme</b> from{" "}
              <a href="https://www.openfhe.org/" target="_blank" rel="noreferrer" className="underline" style={{ color: "#f4743a" }}>
                OpenFHE
              </a>, a leading open-source HE library.
            </p>
          </div>

          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              How Does This Demo Work?
            </h2>
            <p className="mb-2">
              You draw a digit on the canvas. The pixel values are sent to our backend, which:
            </p>
            <ol className="list-decimal ml-5 space-y-1">
              <li><b>Encrypts</b> the 784 pixels into BFV ciphertext.</li>
              <li>Runs the full <b>CNN pipeline</b> (Conv → x² → AvgPool → FC) entirely on encrypted data.</li>
              <li><b>Decrypts</b> the 10 output logits and returns the predicted digit.</li>
            </ol>
            <p className="mt-2">
              The model uses <b>x² activation</b> (instead of ReLU) because HE only supports
              addition and multiplication — no comparisons. It was trained on MNIST with
              this constraint from the start.
            </p>
          </div>

          <div>
            <h2 className="text-xl font-medium mb-3" style={{ color: "#333" }}>
              This Is Cool, Can I Repurpose It?
            </h2>
            <p>
              Absolutely! This is an open-source FYP (Final Year Project) — an{" "}
              <b>Encrypted Machine Learning Benchmark Framework</b> comparing three HE
              libraries: <b>OpenFHE</b>, <b>Microsoft SEAL</b>, and <b>HElib</b>.
              Check out the source on{" "}
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
        Built by Tiffany Yong · FYP 2025-2026 · Powered by OpenFHE, Spring Boot &amp; React
      </footer>
    </div>
  );
}

/* ─── Reusable sub-components ─── */

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

function ControlDropdown({ label, options }) {
  return (
    <div className="px-2 md:px-3">
      <div className="text-[10px] uppercase tracking-wide mb-0.5" style={{ color: "#999" }}>{label}</div>
      <select
        className="text-sm font-medium bg-white border rounded px-2 py-0.5 cursor-pointer appearance-none pr-6"
        style={{ color: "#333", borderColor: "#ccc", backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23999'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 6px center" }}
      >
        {options.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
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
    relu1:   "Square activation (x² approximation of ReLU)",
    pool1:   "2×2 average pooling on encrypted data",
    conv2:   "Encrypted 5×5 convolution, single channel",
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
