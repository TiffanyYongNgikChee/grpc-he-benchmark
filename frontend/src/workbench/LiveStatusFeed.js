import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

/**
 * LiveStatusFeed — Shows what the server is doing right now, one step at a time.
 *
 * Each step has:
 *   - A timestamp (elapsed seconds)
 *   - A short label  ("Conv1 complete")
 *   - A description explaining what is happening cryptographically
 *
 * When a new step becomes active, the previous step's description fades out
 * and the new one slides in. Completed steps are shown as a compact log line.
 */

const STEP_INFO = {
  encrypt: {
    label:  "Encrypting pixel data",
    color:  "#6aabf7",
    icon:   "🔒",
    desc:   "Your 28×28 pixel values are being packed into a BFV ciphertext using the public key. After this step, the server can compute on the data — but can never read it. The plaintext image is gone.",
  },
  conv1: {
    label:  "Conv1 — First convolutional layer",
    color:  "#c890f0",
    icon:   "⊛",
    desc:   "A 5×5 weight kernel is applied to the encrypted feature map using polynomial multiplication in ciphertext space. The server performs thousands of homomorphic additions and multiplications — without ever seeing a pixel.",
  },
  bias1: {
    label:  "Bias1 — Adding bias to Conv1",
    color:  "#c890f0",
    icon:   "+",
    desc:   "Encrypted bias constants are added to each position in the Conv1 output. Bias addition in BFV is just plaintext-ciphertext addition — very fast, but still fully encrypted.",
  },
  relu1: {
    label:  "Activation — Polynomial x² (layer 1)",
    color:  "#f0c030",
    icon:   "∿",
    desc:   "Because ReLU cannot be computed on ciphertext, we use x² as a polynomial approximation. This is a single homomorphic multiplication — it consumes one level of the multiplicative depth budget.",
  },
  pool1: {
    label:  "Pool1 — Average pooling (layer 1)",
    color:  "#f0c030",
    icon:   "↓",
    desc:   "A 2×2 average pool is computed using homomorphic additions and a plaintext division constant. Pooling halves the spatial size, reducing the ciphertext noise budget consumed by later layers.",
  },
  conv2: {
    label:  "Conv2 — Second convolutional layer",
    color:  "#c890f0",
    icon:   "⊛",
    desc:   "The second 5×5 convolution runs on the still-encrypted feature map from Pool1. The server is learning deeper spatial patterns — shape edges, curves — entirely in ciphertext arithmetic.",
  },
  bias2: {
    label:  "Bias2 — Adding bias to Conv2",
    color:  "#c890f0",
    icon:   "+",
    desc:   "Encrypted bias terms for Conv2 are summed into the ciphertext output. The noise in the ciphertext has grown from two convolutions — decryption is still far away.",
  },
  relu2: {
    label:  "Activation — Polynomial x² (layer 2)",
    color:  "#f0c030",
    icon:   "∿",
    desc:   "Another x² activation is applied. This is the deepest multiplicative operation in the circuit — it brings the ciphertext close to its noise budget limit. The parameter choice of 128-bit security was calibrated to survive exactly this.",
  },
  pool2: {
    label:  "Pool2 — Average pooling (layer 2)",
    color:  "#f0c030",
    icon:   "↓",
    desc:   "Second average pool. After this, the feature map is small enough to be flattened and passed to the fully-connected layer. The ciphertext is still intact and unreadable.",
  },
  fc: {
    label:  "FC — Fully connected layer",
    color:  "#e05a5a",
    icon:   "Σ",
    desc:   "A matrix–vector multiply over the encrypted 16-dimensional feature vector. Each of the 10 output neurons (one per digit) accumulates a weighted sum — all in homomorphic arithmetic. This is typically the most expensive layer.",
  },
  biasfc: {
    label:  "BiasFc — Adding final bias",
    color:  "#e05a5a",
    icon:   "+",
    desc:   "The last bias vector is added to the encrypted logit outputs. After this step, the 10 ciphertexts each encode one raw class score — still fully encrypted.",
  },
  decrypt: {
    label:  "Decrypting results",
    color:  "#58c896",
    icon:   "🔓",
    desc:   "The 10 encrypted logit values are decrypted using your private key. The server learns nothing — it only sees ciphertexts. The plaintext scores are revealed only here, locally, to determine the predicted digit.",
  },
};

const ORDER = [
  "encrypt","conv1","bias1","relu1","pool1",
  "conv2","bias2","relu2","pool2",
  "fc","biasfc","decrypt",
];

export default function LiveStatusFeed({ layerStatus, elapsedMs, running, result, onLogComplete }) {
  /* Track the currently active (processing) step */
  const [activeStep, setActiveStep] = useState(null);

  /* Compact log of completed steps */
  const [log, setLog] = useState([]);

  /* Timestamp when each step started */
  const stepStartRef = useRef({});
  const prevStatusRef = useRef({});

  /* Elapsed seconds for the active step */
  const [stepElapsed, setStepElapsed] = useState(0);
  const stepClockRef = useRef(null);

  useEffect(() => {
    if (!layerStatus) return;

    const prev = prevStatusRef.current;

    for (const id of ORDER) {
      const cur  = layerStatus[id];
      const was  = prev[id];

      /* Step just started processing */
      if (cur === "processing" && was !== "processing") {
        stepStartRef.current[id] = elapsedMs;
        setActiveStep(id);
        setStepElapsed(0);
      }

      /* Step just finished — add to compact log */
      if (cur === "done" && was === "processing") {
        const startMs  = stepStartRef.current[id] ?? elapsedMs;
        const durationMs = elapsedMs - startMs;
        setLog(l => [
          ...l,
          {
            id,
            label:      STEP_INFO[id]?.label ?? id,
            color:      STEP_INFO[id]?.color ?? "#888",
            icon:       STEP_INFO[id]?.icon  ?? "·",
            desc:       STEP_INFO[id]?.desc  ?? "",
            atMs:       elapsedMs,
            durationMs,
          },
        ]);
        /* Clear active step only if no newer step is about to replace it */
        setActiveStep(a => (a === id ? null : a));
      }
    }

    prevStatusRef.current = { ...layerStatus };
  }, [layerStatus, elapsedMs]);

  /* Fire onLogComplete once when the run finishes with a non-empty log */
  const firedRef = useRef(false);
  useEffect(() => {
    if (result && !running && log.length > 0 && !firedRef.current) {
      firedRef.current = true;
      if (onLogComplete) onLogComplete(log, result);
    }
    /* Reset fired flag when a new run begins */
    if (running && Object.values(layerStatus || {}).every(s => s === "idle")) {
      firedRef.current = false;
    }
  }, [result, running, log, layerStatus, onLogComplete]);

  /* Tick the per-step elapsed counter */
  useEffect(() => {
    if (activeStep && running) {
      const t = setInterval(() => {
        const start = stepStartRef.current[activeStep] ?? elapsedMs;
        setStepElapsed(elapsedMs - start);
      }, 80);
      stepClockRef.current = t;
      return () => clearInterval(t);
    }
  }, [activeStep, running, elapsedMs]);

  /* Reset when inference restarts */
  useEffect(() => {
    if (running && Object.values(layerStatus || {}).every(s => s === "idle")) {
      setLog([]);
      setActiveStep(null);
      stepStartRef.current = {};
      prevStatusRef.current = {};
    }
  }, [running, layerStatus]);

  /* Nothing to show — idle state */
  if (!running && !result && log.length === 0) return null;

  const info = activeStep ? STEP_INFO[activeStep] : null;
  const totalSec = (elapsedMs / 1000).toFixed(1);

  return (
    <div style={{
      borderRadius: 8,
      background: "rgba(0,0,0,0.06)",
      border: "1.5px solid rgba(0,0,0,0.18)",
      overflow: "hidden",
    }}>

      {/* ── Header ── */}
      <div style={{
        padding: "8px 14px",
        borderBottom: "1px solid rgba(0,0,0,0.12)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{
          fontFamily: "'Press Start 2P', monospace",
          fontSize: "0.55rem", letterSpacing: "0.2em", textTransform: "uppercase",
          color: "#4a3800",
        }}>Server Log</span>
        {running && (
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.5rem", letterSpacing: "0.08em",
            color: "#58c896",
            textShadow: "0 0 8px rgba(88,200,150,0.5)",
          }}>{totalSec}s</span>
        )}
        {result && !running && (
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.45rem", letterSpacing: "0.1em",
            color: "#58c896",
          }}>done · {(result.totalMs / 1000).toFixed(2)}s</span>
        )}
      </div>

      <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 6 }}>

        {/* ── Compact completed log ── */}
        {log.map((entry) => (
          <div key={entry.id} style={{
            display: "flex", alignItems: "center", gap: 8,
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
              color: "#7a6a44", flexShrink: 0, width: 38,
            }}>{(entry.atMs / 1000).toFixed(1)}s</span>
            {/* Label */}
            <span style={{
              fontFamily: "system-ui, sans-serif",
              fontSize: "0.72rem", color: "#2a2a2a",
              flex: 1,
            }}>{entry.label}</span>
            {/* Duration */}
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.38rem", letterSpacing: "0.06em",
              color: entry.color, opacity: 1, flexShrink: 0,
            }}>{(entry.durationMs / 1000).toFixed(1)}s</span>
            {/* Tick */}
            <span style={{ color: "#58c896", fontSize: "0.75rem", flexShrink: 0 }}>✓</span>
          </div>
        ))}

        {/* ── Active step — full description, animates in/out ── */}
        <AnimatePresence mode="wait">
          {info && running && (
            <motion.div
              key={activeStep}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              style={{
                marginTop: log.length > 0 ? 8 : 0,
                borderRadius: 7,
                background: `${info.color}0d`,
                border: `1.5px solid ${info.color}30`,
                padding: "12px 14px",
                boxShadow: `0 0 18px ${info.color}10`,
              }}
            >
              {/* Step header row */}
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                marginBottom: 8,
              }}>
                {/* Pulsing dot */}
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                  style={{
                    display: "inline-block",
                    width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
                    background: info.color,
                    boxShadow: `0 0 8px ${info.color}`,
                  }}
                />
                <span style={{
                  fontFamily: "'Press Start 2P', monospace",
                  fontSize: "0.52rem", letterSpacing: "0.1em",
                  color: info.color,
                  flex: 1,
                }}>{info.label}</span>
                {/* Step elapsed */}
                <span style={{
                  fontFamily: "'Press Start 2P', monospace",
                  fontSize: "0.48rem", letterSpacing: "0.06em",
                  color: `${info.color}99`,
                  flexShrink: 0,
                }}>{(stepElapsed / 1000).toFixed(1)}s</span>
              </div>

              {/* Description */}
              <p style={{
                fontFamily: "Georgia, 'Times New Roman', serif",
                fontSize: "0.82rem",
                color: "#2a2218",
                lineHeight: 1.8,
                margin: 0,
                letterSpacing: "0.01em",
              }}>
                {info.desc}
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── All done — summary line ── */}
        {result && !running && log.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            style={{
              marginTop: 6,
              padding: "8px 12px",
              borderRadius: 6,
              background: "rgba(20,100,60,0.1)",
              border: "1px solid rgba(20,100,60,0.3)",
              display: "flex", alignItems: "center", gap: 8,
            }}
          >
            <span style={{ color: "#117744", fontSize: "0.85rem" }}>★</span>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.42rem", letterSpacing: "0.1em",
              color: "#117744",
            }}>
              Inference complete — {log.length} layers · {(result.totalMs / 1000).toFixed(2)}s total
            </span>
          </motion.div>
        )}
      </div>
    </div>
  );
}
