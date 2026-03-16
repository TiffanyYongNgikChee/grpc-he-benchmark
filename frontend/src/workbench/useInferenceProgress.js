import { useState, useCallback, useRef, useEffect } from "react";

/**
 * useInferenceProgress — Manages layer-by-layer progress animation
 * during encrypted CNN inference.
 *
 * Two modes:
 *   1. SSE mode   — listens to real server-sent events from /api/predict/stream
 *   2. Timer mode  — simulates progress based on known average layer timings
 *
 * The hook exposes `activeLayer` (index into LAYERS), `layerStatus` map,
 * and elapsed-time helpers.
 */

/* Average layer timings from benchmarks (ms) — used for timer-mode estimation */
const ESTIMATED_TIMINGS = {
  encrypt:  260,
  conv1:    6950,
  bias1:    12,
  relu1:    695,
  pool1:    1150,
  conv2:    6540,
  bias2:    24,
  relu2:    695,
  pool2:    1025,
  fc:       12200,
  biasfc:   22,
  decrypt:  70,
};

/* Layer IDs in pipeline order (matches CnnPipeline LAYERS, skipping input/output) */
const PIPELINE_LAYER_IDS = [
  "encrypt", "conv1", "bias1", "relu1", "pool1",
  "conv2", "bias2", "relu2", "pool2",
  "fc", "biasfc", "decrypt",
];

/**
 * Status for each layer:
 *   "idle"       — not started yet (grey)
 *   "processing" — currently running (pulsing orange animation)
 *   "done"       — completed (solid green / category colour)
 */

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8080/api";

export default function useInferenceProgress() {
  /* Layer status map:  { encrypt: "idle", conv1: "idle", ... } */
  const [layerStatus, setLayerStatus] = useState(() =>
    Object.fromEntries(PIPELINE_LAYER_IDS.map((id) => [id, "idle"]))
  );

  /* Index of the layer currently being processed (-1 = not started, 12 = all done) */
  const [activeLayerIdx, setActiveLayerIdx] = useState(-1);

  /* Elapsed time since inference started (ms) */
  const [elapsedMs, setElapsedMs] = useState(0);

  /* Is inference in progress? */
  const [running, setRunning] = useState(false);

  /* Internal refs for timers */
  const timersRef = useRef([]);
  const clockRef = useRef(null);
  const startTimeRef = useRef(null);
  const eventSourceRef = useRef(null);

  /* ── Cleanup helper ── */
  const cleanup = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    if (clockRef.current) {
      cancelAnimationFrame(clockRef.current);
      clockRef.current = null;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  useEffect(() => cleanup, [cleanup]);

  /* ── Reset to idle state ── */
  const reset = useCallback(() => {
    cleanup();
    setLayerStatus(Object.fromEntries(PIPELINE_LAYER_IDS.map((id) => [id, "idle"])));
    setActiveLayerIdx(-1);
    setElapsedMs(0);
    setRunning(false);
  }, [cleanup]);

  /* ── Elapsed-time clock (requestAnimationFrame) ── */
  const startClock = useCallback(() => {
    startTimeRef.current = performance.now();
    const tick = () => {
      setElapsedMs(performance.now() - startTimeRef.current);
      clockRef.current = requestAnimationFrame(tick);
    };
    clockRef.current = requestAnimationFrame(tick);
  }, []);

  const stopClock = useCallback(() => {
    if (clockRef.current) {
      cancelAnimationFrame(clockRef.current);
      clockRef.current = null;
    }
  }, []);

  /* ── Advance to next layer ── */
  const advanceToLayer = useCallback((layerIdx) => {
    setActiveLayerIdx(layerIdx);
    setLayerStatus((prev) => {
      const next = { ...prev };
      // Mark all previous layers as done
      for (let i = 0; i < layerIdx && i < PIPELINE_LAYER_IDS.length; i++) {
        next[PIPELINE_LAYER_IDS[i]] = "done";
      }
      // Mark current layer as processing
      if (layerIdx < PIPELINE_LAYER_IDS.length) {
        next[PIPELINE_LAYER_IDS[layerIdx]] = "processing";
      }
      return next;
    });
  }, []);

  /* ── Mark all layers done ── */
  const markAllDone = useCallback(() => {
    setLayerStatus(Object.fromEntries(PIPELINE_LAYER_IDS.map((id) => [id, "done"])));
    setActiveLayerIdx(PIPELINE_LAYER_IDS.length);
    stopClock();
    setRunning(false);
  }, [stopClock]);

  /* ── Start simulated (timer-based) progress ── */
  const startSimulated = useCallback(() => {
    cleanup();
    setRunning(true);
    startClock();
    advanceToLayer(0);

    let cumulative = 0;
    PIPELINE_LAYER_IDS.forEach((id, idx) => {
      const duration = ESTIMATED_TIMINGS[id] || 100;
      cumulative += duration;

      const t = setTimeout(() => {
        if (idx + 1 < PIPELINE_LAYER_IDS.length) {
          advanceToLayer(idx + 1);
        }
      }, cumulative);
      timersRef.current.push(t);
    });
  }, [cleanup, startClock, advanceToLayer]);

  /* ── Start SSE-based progress (future backend support) ── */
  const startSSE = useCallback((requestBody) => {
    cleanup();
    setRunning(true);
    startClock();
    advanceToLayer(0);

    return new Promise((resolve, reject) => {
      // Use fetch + ReadableStream for POST-based SSE
      // (EventSource only supports GET, so we use fetch streaming)
      fetch(`${API_BASE}/predict/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(300_000),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          function read() {
            reader.read().then(({ done, value }) => {
              if (done) return;
              buffer += decoder.decode(value, { stream: true });

              // Parse SSE lines
              const lines = buffer.split("\n");
              buffer = lines.pop(); // Keep incomplete line in buffer

              for (const line of lines) {
                if (line.startsWith("data:")) {
                  try {
                    const event = JSON.parse(line.slice(5).trim());
                    handleSSEEvent(event, resolve);
                  } catch {
                    // Skip malformed events
                  }
                }
              }
              read();
            }).catch(reject);
          }
          read();
        })
        .catch((err) => {
          // SSE not available — caller should fall back to simulated
          reject(err);
        });
    });

    function handleSSEEvent(event, resolve) {
      if (event.type === "layer_start") {
        const idx = PIPELINE_LAYER_IDS.indexOf(event.layer);
        if (idx >= 0) advanceToLayer(idx);
      } else if (event.type === "layer_done") {
        const idx = PIPELINE_LAYER_IDS.indexOf(event.layer);
        if (idx >= 0) {
          setLayerStatus((prev) => ({ ...prev, [event.layer]: "done" }));
        }
      } else if (event.type === "complete") {
        markAllDone();
        resolve(event.result);
      }
    }
  }, [cleanup, startClock, advanceToLayer, markAllDone]);

  return {
    layerStatus,
    activeLayerIdx,
    elapsedMs,
    running,
    reset,
    startSimulated,
    startSSE,
    markAllDone,
    PIPELINE_LAYER_IDS,
  };
}
