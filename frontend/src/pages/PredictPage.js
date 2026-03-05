import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import DrawingCanvas from "../components/DrawingCanvas";
import LogitsChart from "../components/LogitsChart";
import TimingChart from "../components/TimingChart";
import CountUp from "../components/CountUp";
import { predictDigit } from "../api/client";

/* Framer Motion variants */
const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.12 } },
};

/**
 * PredictPage - Draw a digit and get an encrypted prediction.
 * Left column: drawing canvas + 28×28 preview
 * Right column: predict button + result display + timing breakdown
 */
export default function PredictPage() {
  const [pixels, setPixels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handlePredict() {
    if (!pixels) return;
    setLoading(true);
    setError(null);
    try {
      const response = await predictDigit(pixels, 1000);
      setResult(response);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-white mb-2">
        Encrypted Digit Prediction
      </h1>
      <p className="text-slate-400 mb-8">
        Draw a digit (0–9) and run inference on encrypted data using homomorphic encryption.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left column: Drawing canvas */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Draw a Digit</h2>
          <DrawingCanvas onPixelsReady={setPixels} disabled={loading} />

          {/* Predict button */}
          <button
            onClick={handlePredict}
            disabled={!pixels || loading}
            className="mt-4 w-full py-3 rounded-lg font-semibold text-lg transition-all
                       bg-emerald-600 hover:bg-emerald-500 text-white
                       disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running Encrypted Inference...
              </span>
            ) : (
              " Predict with HE"
            )}
          </button>

          {/* Pixel preview — shows the 28×28 downscaled version */}
          {pixels && (
            <div className="mt-4">
              <p className="text-xs text-slate-500 mb-2">
                28×28 preview (what the model sees):
              </p>
              <canvas
                width={28}
                height={28}
                className="border border-slate-600 rounded"
                style={{ width: 112, height: 112, imageRendering: "pixelated" }}
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
        </div>

        {/* Right column: Prediction results */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Prediction Result</h2>

          {/* Error state */}
          {error && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-4">
              <p className="text-red-400 text-sm font-medium">Prediction failed</p>
              <p className="text-red-300 text-xs mt-1">{error}</p>
              <p className="text-red-400/60 text-xs mt-2">
                Make sure Docker and Spring Boot are running.
              </p>
            </div>
          )}

          {/* Empty state */}
          {!result && !error && !loading && (
            <div className="text-center text-slate-500 py-12">
              {pixels
                ? 'Click "Predict with HE" to run encrypted inference'
                : "Draw a digit on the left to get started"}
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="text-center py-12">
              <div className="inline-flex items-center gap-3 text-emerald-400">
                <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm">Encrypting and running CNN...</span>
              </div>
            </div>
          )}

          {/* Result display */}
          <AnimatePresence mode="wait">
          {result && !loading && (
            <motion.div
              key={result.predictedDigit + "-" + result.totalMs}
              className="space-y-6"
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={stagger}
            >
              {/* Big predicted digit */}
              <motion.div className="text-center" variants={fadeIn}>
                <p className="text-sm text-slate-400 mb-1">Predicted Digit</p>
                <p className="text-7xl font-bold text-emerald-400">
                  <CountUp end={result.predictedDigit} duration={400} decimals={0} />
                </p>
              </motion.div>

              {/* Confidence and total time */}
              <motion.div className="grid grid-cols-2 gap-4" variants={fadeIn}>
                <div className="bg-slate-900 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500 mb-1">Confidence</p>
                  <p className="text-xl font-semibold text-white">
                    <CountUp end={result.confidence * 100} duration={900} decimals={1} suffix="%" />
                  </p>
                </div>
                <div className="bg-slate-900 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500 mb-1">Total Time</p>
                  <p className="text-xl font-semibold text-white">
                    <CountUp end={result.totalMs} duration={900} decimals={1} suffix="ms" />
                  </p>
                </div>
              </motion.div>

              {/* Status badge */}
              <motion.div className="flex items-center justify-center gap-2" variants={fadeIn}>
                <span
                  className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${
                    result.status === "success"
                      ? "bg-emerald-900/40 text-emerald-400 border border-emerald-700"
                      : "bg-red-900/40 text-red-400 border border-red-700"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      result.status === "success" ? "bg-emerald-400" : "bg-red-400"
                    }`}
                  />
                  {result.status}
                </span>
                <span className="text-xs text-slate-500">
                  Float model accuracy: {result.floatModelAccuracy}%
                </span>
              </motion.div>

              {/* Per-layer timing breakdown (Chart.js horizontal bar) */}
              <motion.div variants={fadeIn}>
                <h3 className="text-sm font-medium text-slate-300 mb-3">
                  Layer Timing Breakdown
                </h3>
                <TimingChart result={result} />
              </motion.div>

              {/* Logits bar chart (Chart.js) */}
              <motion.div variants={fadeIn}>
                <h3 className="text-sm font-medium text-slate-300 mb-3">
                  Output Logits (per digit)
                </h3>
                <LogitsChart
                  logits={result.logits}
                  predictedDigit={result.predictedDigit}
                />
              </motion.div>
            </motion.div>
          )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
