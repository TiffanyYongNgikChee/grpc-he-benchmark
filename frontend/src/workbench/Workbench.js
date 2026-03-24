import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import CnnPipeline, { LAYERS } from "./CnnPipeline";
import MiniCanvas from "./MiniCanvas";
import OutputPanel from "./OutputPanel";
import MetricsStrip from "./MetricsStrip";
import LibraryComparison from "./LibraryComparison";
import MnistBatchBenchmark from "./MnistBatchBenchmark";
import ParameterComparison from "./ParameterComparison";
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

  /* Parameter panel state */
  const [securityLevel, setSecurityLevel] = useState(0);       // 0=128-bit, 1=192-bit, 2=256-bit
  const [activationDegree, setActivationDegree] = useState(2); // 2=x², 3=cubic, 4=quartic

  /* Animation — layer-by-layer progress */
  const progress = useInferenceProgress();
  const [activeStep, setActiveStep] = useState(-1);
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const [runCount, setRunCount] = useState(0);
  const animTimers = useRef([]);

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
        <ControlDropdown label="Library" options={["OpenFHE", "SEAL", "HElib"]} />
        <ControlDropdown label="Encryption" options={["BFV"]} />
        <ControlDropdownStateful
          label="Security"
          options={[
            { label: "128-bit", value: 0 },
            { label: "192-bit", value: 1 },
            { label: "256-bit", value: 2 },
          ]}
          value={securityLevel}
          onChange={setSecurityLevel}
          disabled={loading}
        />
        <ControlDropdownStateful
          label="Activation"
          options={[
            { label: "x² (degree 2)", value: 2 },
            { label: "degree 3", value: 3 },
            { label: "degree 4", value: 4 },
          ]}
          value={activationDegree}
          onChange={setActivationDegree}
          disabled={loading}
        />
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
                layerStatus={progress.running || result ? progress.layerStatus : null}
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

      {/* ═══════════ METRICS BAR ═══════════ */}
      <div style={{ background: "#ebeaea", borderTop: "1px solid #d9d9d9" }}>
        <div className="max-w-[1440px] mx-auto px-4 md:px-8 py-3">
          <MetricsStrip result={result} />
        </div>
      </div>

      {/* ═══════════ LIBRARY COMPARISON ═══════════ */}
      <div className="px-4 md:px-8 py-8" style={{ background: "#f7f7f7", borderTop: "1px solid #e5e5e5" }}>
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

      {/* ═══════════ MNIST BATCH BENCHMARK ═══════════ */}
      <div className="px-4 md:px-8 py-8" style={{ background: "#ebeaea", borderTop: "1px solid #d9d9d9" }}>
        <div className="max-w-[1100px] mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <h3
              className="text-xs font-medium uppercase tracking-widest"
              style={{ color: "#888", letterSpacing: "0.12em" }}
            >
              MNIST BATCH BENCHMARK
            </h3>
            <span className="text-[10px]" style={{ color: "#bbb" }}>
              10 test images • Encrypted CNN Inference
            </span>
          </div>
          <div
            className="rounded-lg p-5"
            style={{ background: "#fafafa", border: "1px solid #e5e5e5" }}
          >
            <MnistBatchBenchmark />
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
              Security Level & Activation Degree Comparison
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
              max(0, x). You can choose between x² (degree 2), degree 3, and degree 4
              polynomials using the controls above. The x² function gives the best accuracy
              as it matches the training configuration.
            </p>
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

function ControlDropdownStateful({ label, options, value, onChange, disabled }) {
  return (
    <div className="px-2 md:px-3">
      <div className="text-[10px] uppercase tracking-wide mb-0.5" style={{ color: "#999" }}>{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="text-sm font-medium bg-white border rounded px-2 py-0.5 cursor-pointer appearance-none pr-6 disabled:opacity-50"
        style={{ color: "#333", borderColor: "#ccc", backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23999'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 6px center" }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
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
