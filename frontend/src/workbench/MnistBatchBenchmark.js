import { useState, useCallback, useRef, useEffect } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import { TEST_IMAGES, TEST_LABELS } from "./mnistTestData";
import { predictDigit, checkHealth } from "../api/client";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/**
 * MnistBatchBenchmark — Transparent pipeline view for batch MNIST inference.
 *
 * Each of the 10 test images is shown as a card that visually progresses
 * through the encrypted CNN pipeline stages:
 *   queued → encrypting → conv1 → … → decrypting → done ✓ / ✗
 *
 * The user can see exactly where their data is at all times.
 */

/* Pipeline stages (simplified from CnnPipeline LAYERS) */
const STAGES = [
  { id: "encrypt", label: "Encrypt", color: "#0db7c4" },
  { id: "conv1",   label: "Conv 1",  color: "#7b3ff2" },
  { id: "relu1",   label: "x²",      color: "#e68a00" },
  { id: "pool1",   label: "Pool 1",  color: "#e68a00" },
  { id: "conv2",   label: "Conv 2",  color: "#7b3ff2" },
  { id: "relu2",   label: "x²",      color: "#e68a00" },
  { id: "pool2",   label: "Pool 2",  color: "#e68a00" },
  { id: "fc",      label: "FC",      color: "#e03e52" },
  { id: "decrypt", label: "Decrypt", color: "#0db7c4" },
];

/* Estimated ms per stage (for simulated progress) */
const STAGE_DURATIONS = [260, 6950, 695, 1150, 6540, 695, 1025, 12200, 70];

/* ── Tiny 28×28 canvas thumbnail ── */
function DigitThumbnail({ pixels, size = 28 }) {
  const drawnRef = useRef(false);
  const setRef = useCallback(
    (node) => {
      if (node && !drawnRef.current) {
        const ctx = node.getContext("2d");
        const imgData = ctx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
          const v = pixels[i];
          imgData.data[i * 4] = v;
          imgData.data[i * 4 + 1] = v;
          imgData.data[i * 4 + 2] = v;
          imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        drawnRef.current = true;
      }
    },
    [pixels]
  );

  return (
    <canvas
      ref={setRef}
      width={28}
      height={28}
      style={{
        width: size,
        height: size,
        imageRendering: "pixelated",
        borderRadius: 2,
        border: "2px solid #333",
        borderTop: "2px solid #666",
        background: "#000",
      }}
    />
  );
}

/* ── Stage pipeline mini-bar for one image ── */
function StagePipeline({ activeStage, done, error }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
      {STAGES.map((stage, i) => {
        let bg, opacity;
        if (error) {
          bg = "#cc2222"; opacity = 0.5;
        } else if (done) {
          bg = stage.color; opacity = 1;
        } else if (activeStage > i) {
          bg = stage.color; opacity = 1;
        } else if (activeStage === i) {
          bg = stage.color; opacity = 1;
        } else {
          bg = "#aaaaaa"; opacity = 0.35;
        }
        const isActive = activeStage === i && !done && !error;
        return (
          <div
            key={stage.id}
            className="relative group"
            style={{
              width: isActive ? 22 : 14,
              height: 7,
              borderRadius: 2,
              background: bg,
              opacity,
              transition: "all 0.4s ease",
              border: isActive ? `1px solid ${stage.color}` : "1px solid rgba(0,0,0,0.2)",
            }}
          >
            <span
              style={{
                position: "absolute", bottom: "calc(100% + 4px)", left: "50%",
                transform: "translateX(-50%)",
                padding: "2px 6px", borderRadius: 2,
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.32rem", whiteSpace: "nowrap",
                background: "#1a2a3a", color: "#e8d8a0",
                opacity: 0, pointerEvents: "none",
                transition: "opacity 0.15s", zIndex: 10,
              }}
              className="group-hover:opacity-100"
            >
              {stage.label}
            </span>
            {isActive && (
              <span
                className="absolute inset-0 animate-pulse"
                style={{ borderRadius: 2, background: stage.color, opacity: 0.5 }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Per-image card — Habbo 2007 style ── */
function ImageCard({ index, pixels, label, result, status, activeStage, isCurrent }) {
  const isDone   = status === "done";
  const isError  = status === "error";
  const isQueued = status === "queued";
  const isIdle   = status === "idle";
  const dimmed   = isQueued || isIdle;

  /* Card background & border based on state */
  let cardBg     = "#ddd8c8";   // default cream
  let borderT    = "#c8c0a8";
  let borderL    = "#bbb4a0";
  let borderB    = "#888";
  let borderR    = "#888";

  if (isCurrent) {
    cardBg  = "#e8dfa0";   // warm amber highlight
    borderT = "#f0c030";
    borderL = "#d4a800";
    borderB = "#7a5e00";
    borderR = "#7a5e00";
  } else if (isDone && result?.correct) {
    cardBg  = "#d0e8d0";
    borderT = "#44aa66";
    borderL = "#338855";
    borderB = "#226644";
    borderR = "#226644";
  } else if (isDone && !result?.correct) {
    cardBg  = "#e8d0d0";
    borderT = "#cc5555";
    borderL = "#aa3333";
    borderB = "#882222";
    borderR = "#882222";
  } else if (isError) {
    cardBg  = "#e8d0d0";
    borderT = "#cc5555";
    borderL = "#aa3333";
    borderB = "#882222";
    borderR = "#882222";
  }

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "6px 10px",
      background: cardBg,
      borderTop: `2px solid ${borderT}`,
      borderLeft: `2px solid ${borderL}`,
      borderBottom: `2px solid ${borderB}`,
      borderRight: `2px solid ${borderR}`,
      borderRadius: 2,
      opacity: dimmed ? 0.5 : 1,
      transition: "all 0.3s ease",
    }}>
      {/* Index badge — Habbo pill style */}
      <span style={{
        fontFamily: "'Press Start 2P', monospace",
        fontSize: "0.42rem", letterSpacing: "0.06em",
        color: "#5a4a22",
        width: 18, textAlign: "center", flexShrink: 0,
      }}>{index + 1}</span>

      {/* Thumbnail */}
      <DigitThumbnail pixels={pixels} size={30} />

      {/* True label */}
      <span style={{
        fontFamily: "'Press Start 2P', monospace",
        fontSize: "0.75rem", fontWeight: "bold",
        color: "#2a1a00",
        width: 16, textAlign: "center", flexShrink: 0,
      }}>{label}</span>

      {/* Arrow */}
      <span style={{ color: "#8a7a5a", fontSize: "0.8rem", flexShrink: 0 }}>›</span>

      {/* Pipeline stages */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {dimmed ? (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ display: "flex", gap: 3 }}>
              {STAGES.map((s) => (
                <div key={s.id} style={{
                  width: 14, height: 7, borderRadius: 2,
                  background: "#aaa", opacity: 0.25,
                  border: "1px solid rgba(0,0,0,0.15)",
                }} />
              ))}
            </div>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.32rem", color: "#8a7a5a",
            }}>{isQueued ? "Queued" : "Waiting"}</span>
          </div>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <StagePipeline activeStage={activeStage} done={isDone} error={isError} />
            {isCurrent && !isDone && (
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span className="relative flex" style={{ width: 10, height: 10 }}>
                  <span
                    className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                    style={{ background: STAGES[Math.max(0, activeStage)]?.color }}
                  />
                  <span
                    className="relative inline-flex rounded-full"
                    style={{ width: 10, height: 10, background: STAGES[Math.max(0, activeStage)]?.color }}
                  />
                </span>
                <span style={{
                  fontFamily: "'Press Start 2P', monospace",
                  fontSize: "0.32rem", whiteSpace: "nowrap",
                  color: STAGES[Math.max(0, activeStage)]?.color,
                }}>
                  {STAGES[Math.max(0, activeStage)]?.label}…
                </span>
              </div>
            )}
            {isDone && !isError && (
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.35rem", color: "#226644",
              }}>✓ Done</span>
            )}
            {isError && (
              <span style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.35rem", color: "#882222",
              }}>✗ Failed</span>
            )}
          </div>
        )}
      </div>

      {/* Arrow */}
      <span style={{ color: "#8a7a5a", fontSize: "0.8rem", flexShrink: 0 }}>›</span>

      {/* Prediction result */}
      <div style={{ width: 64, textAlign: "right", flexShrink: 0 }}>
        {isDone && result && !isError ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 5 }}>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.85rem",
              color: result.correct ? "#226644" : "#882222",
            }}>{result.predicted}</span>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.38rem",
              padding: "2px 5px", borderRadius: 2,
              background: result.correct ? "#b0ddb0" : "#ddb0b0",
              color: result.correct ? "#155533" : "#771111",
              border: `1px solid ${result.correct ? "#338855" : "#aa3333"}`,
            }}>{result.correct ? "✓" : "✗"}</span>
          </div>
        ) : isCurrent ? (
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.32rem", color: "#8a6a00",
          }}>Running…</span>
        ) : isError ? (
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.32rem", color: "#882222",
          }}>Error</span>
        ) : (
          <span style={{ color: "#9a8a6a", fontSize: "0.8rem" }}>—</span>
        )}
      </div>

      {/* Time */}
      <div style={{ width: 52, textAlign: "right", flexShrink: 0 }}>
        {isDone && result ? (
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.38rem", letterSpacing: "0.04em",
            color: "#4a5a7a",
          }}>{(result.totalMs / 1000).toFixed(1)}s</span>
        ) : (
          <span style={{ color: "#9a8a6a", fontSize: "0.8rem" }}>—</span>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Main component
   ══════════════════════════════════════════════════════════════════ */

export default function MnistBatchBenchmark() {
  const [results, setResults] = useState([]);
  const [imageStates, setImageStates] = useState(
    () => TEST_IMAGES.map(() => ({ status: "idle", activeStage: -1, result: null }))
  );
  const [running, setRunning] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(-1);
  const [batchError, setBatchError] = useState(null);
  const abortRef = useRef(false);
  const abortControllerRef = useRef(null);
  const stageTimerRef = useRef(null);

  /* Cleanup timers on unmount */
  const stopStageTimers = useCallback(() => {
    if (stageTimerRef.current) {
      stageTimerRef.current.forEach(clearTimeout);
      stageTimerRef.current = null;
    }
  }, []);
  useEffect(() => () => {
    stopStageTimers();
    if (abortControllerRef.current) abortControllerRef.current.abort();
  }, [stopStageTimers]);

  /* ── Main run loop ── */
  const handleRun = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setResults([]);
    setBatchError(null);
    abortRef.current = false;

    /* Pre-flight health check — fail fast if backend is down */
    const healthy = await checkHealth();
    if (!healthy) {
      setBatchError(
        "Cannot reach the backend server. Make sure EC2 is running and Docker containers are up. " +
        "(SSH in → sudo systemctl start docker → docker compose up -d)"
      );
      setRunning(false);
      return;
    }

    /* Mark all images as queued */
    setImageStates(TEST_IMAGES.map(() => ({ status: "queued", activeStage: -1, result: null })));

    for (let i = 0; i < TEST_IMAGES.length; i++) {
      if (abortRef.current) break;
      setCurrentIdx(i);

      /* Mark current as processing at stage 0 */
      setImageStates((prev) => {
        const next = [...prev];
        next[i] = { status: "processing", activeStage: 0, result: null };
        return next;
      });

      /* Start simulated stage progress (advances the mini pipeline bar).
         This gives the user real-time visual feedback while the single
         /api/predict call is in-flight (~30s). */
      {
        let elapsed = 0;
        const timers = [];
        STAGES.forEach((_, si) => {
          elapsed += STAGE_DURATIONS[si];
          const t = setTimeout(() => {
            setImageStates((prev) => {
              const next = [...prev];
              if (i < next.length && next[i].status === "processing") {
                next[i] = { ...next[i], activeStage: si };
              }
              return next;
            });
          }, elapsed);
          timers.push(t);
        });
        stageTimerRef.current = timers;
      }

      try {
        /* Create an AbortController so the Stop button can cancel immediately */
        const ac = new AbortController();
        abortControllerRef.current = ac;

        const res = await predictDigit(TEST_IMAGES[i], 1000, ac.signal);
        stopStageTimers();

        if (abortRef.current) break;

        const entry = {
          label: TEST_LABELS[i],
          predicted: res.predictedDigit ?? res.predicted_digit,
          correct: (res.predictedDigit ?? res.predicted_digit) === TEST_LABELS[i],
          totalMs: res.totalMs ?? res.total_ms ?? 0,
          confidence: res.confidence ?? 0,
        };

        setResults((prev) => [...prev, entry]);
        setImageStates((prev) => {
          const next = [...prev];
          next[i] = { status: "done", activeStage: STAGES.length, result: entry };
          return next;
        });
      } catch (err) {
        stopStageTimers();

        /* If the user aborted, don't record an error */
        if (abortRef.current || err.name === "AbortError") break;

        const entry = {
          label: TEST_LABELS[i],
          predicted: -1,
          correct: false,
          totalMs: 0,
          confidence: 0,
          error: err.message,
        };
        setResults((prev) => [...prev, entry]);
        setImageStates((prev) => {
          const next = [...prev];
          next[i] = { status: "error", activeStage: -1, result: entry };
          return next;
        });

        /* If first image fails, likely the server is down — stop the whole batch */
        if (i === 0 && (err.message.includes("fetch") || err.message.includes("network") || err.message.includes("Failed"))) {
          setBatchError("Connection to backend lost. Check that the server is running.");
          break;
        }
      }
    }

    abortControllerRef.current = null;
    setCurrentIdx(-1);
    setRunning(false);
  }, [running, stopStageTimers]);

  const handleStop = () => {
    abortRef.current = true;
    stopStageTimers();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  /* ─── Computed stats ─── */
  const doneResults = results.filter((r) => !r.error);
  const correctCount = doneResults.filter((r) => r.correct).length;
  const accuracy = doneResults.length > 0 ? correctCount / doneResults.length : 0;
  const avgTime = doneResults.length > 0
    ? doneResults.reduce((s, r) => s + r.totalMs, 0) / doneResults.length
    : 0;
  const totalTime = doneResults.reduce((s, r) => s + r.totalMs, 0);

  /* ─── Stage legend ─── */
  const StageLegend = (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      {STAGES.map((s) => (
        <div key={s.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{
            width: 12, height: 7, borderRadius: 2,
            background: s.color,
            border: "1px solid rgba(0,0,0,0.2)",
          }} />
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.32rem", letterSpacing: "0.08em",
            color: "#5a4a22",
          }}>{s.label}</span>
        </div>
      ))}
    </div>
  );

  /* ━━━ Empty state ━━━ */
  if (!running && results.length === 0) {
    return (
      <div>
        {/* Info bar */}
        <div style={{
          background: "linear-gradient(180deg, #e8a030 0%, #c07800 100%)",
          border: "2px solid #7a4a00",
          borderTop: "2px solid #f0b840",
          borderRadius: 2,
          padding: "6px 12px",
          marginBottom: 12,
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.42rem", letterSpacing: "0.1em",
            color: "#1a0e00",
          }}>Each image ~30s · Total ~5 min</span>
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.38rem", color: "rgba(26,14,0,0.65)",
          }}>Sequential · BFV 128-bit</span>
        </div>

        {/* Legend */}
        <div style={{
          background: "#c8c0b0",
          border: "2px solid #999",
          borderTop: "2px solid #bbb",
          borderLeft: "2px solid #b0b0b0",
          borderRadius: 2,
          padding: "7px 12px",
          marginBottom: 12,
        }}>
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.36rem", letterSpacing: "0.14em",
            color: "#5a4a22", textTransform: "uppercase",
            display: "block", marginBottom: 6,
          }}>Pipeline stages</span>
          {StageLegend}
        </div>

        {/* Preview cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 14 }}>
          {TEST_IMAGES.map((img, i) => (
            <ImageCard key={i} index={i} pixels={img} label={TEST_LABELS[i]}
              result={null} status="idle" activeStage={-1} isCurrent={false} />
          ))}
        </div>

        {/* Run button */}
        <div style={{ textAlign: "center" }}>
          <button
            onClick={handleRun}
            style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.6rem", letterSpacing: "0.14em",
              padding: "10px 28px",
              background: "linear-gradient(180deg, #f0c030 0%, #c89800 100%)",
              border: "2px solid #7a5e00",
              borderTop: "2px solid #f8e060",
              borderLeft: "2px solid #e8b820",
              borderRadius: 3, cursor: "pointer",
              color: "#1a0e00",
              boxShadow: "0 3px 8px rgba(0,0,0,0.4)",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "linear-gradient(180deg, #f8d040 0%, #d4a400 100%)"}
            onMouseLeave={e => e.currentTarget.style.background = "linear-gradient(180deg, #f0c030 0%, #c89800 100%)"}
          >
            ▶ Run MNIST Batch
          </button>
        </div>

        {batchError && (
          <div style={{
            marginTop: 10,
            background: "#e8c0c0", color: "#771111",
            border: "2px solid #aa3333",
            borderTop: "2px solid #cc5555",
            borderRadius: 2, padding: "8px 12px",
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.38rem", letterSpacing: "0.08em",
            textAlign: "center",
          }}>⚠ {batchError}</div>
        )}
      </div>
    );
  }

  /* ━━━ Running / Results view ━━━ */

  /* Bar chart data */
  const chartData = {
    labels: doneResults.map((r) => `Digit ${r.label}`),
    datasets: [
      {
        label: "Inference Time",
        data: doneResults.map((r) => r.totalMs),
        backgroundColor: doneResults.map((r) =>
          r.correct ? "rgba(34,102,68,0.75)" : "rgba(170,51,51,0.75)"
        ),
        borderColor: doneResults.map((r) =>
          r.correct ? "#226644" : "#aa3333"
        ),
        borderWidth: 2,
        borderRadius: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#1a2a3a",
        titleColor: "#e8d8a0",
        bodyColor: "#c8c0a0",
        callbacks: {
          label: (ctx) => {
            const r = doneResults[ctx.dataIndex];
            return [
              `Time: ${(r.totalMs / 1000).toFixed(1)}s`,
              `Predicted: ${r.predicted} ${r.correct ? "✓" : "✗"}`,
              `Confidence: ${(r.confidence * 100).toFixed(1)}%`,
            ];
          },
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#5a4a22", font: { family: "'Press Start 2P'", size: 7 } },
      },
      y: {
        grid: { color: "rgba(0,0,0,0.08)" },
        ticks: { color: "#5a4a22", font: { family: "'Press Start 2P'", size: 7 }, callback: (v) => `${(v / 1000).toFixed(0)}s` },
        title: { display: true, text: "Time (s)", color: "#7a6a42", font: { size: 9 } },
      },
    },
  };

  return (
    <div>
      {/* Stop button — shown while running */}
      {running && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
          <button
            onClick={handleStop}
            style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.38rem", letterSpacing: "0.1em",
              padding: "5px 12px",
              background: "linear-gradient(180deg, #cc4444, #882222)",
              border: "2px solid #551111",
              borderTop: "2px solid #ee6666",
              borderRadius: 2, cursor: "pointer",
              color: "#ffe8e8",
            }}
          >■ Stop</button>
        </div>
      )}

      {/* Progress bar — while running */}
      {running && (
        <div style={{
          background: "#c8c0b0",
          border: "2px solid #999",
          borderTop: "2px solid #bbb",
          borderLeft: "2px solid #b0b0b0",
          borderRadius: 2,
          padding: "8px 12px",
          marginBottom: 10,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.38rem", color: "#5a4a22",
            }}>Image {Math.min(results.length + 1, TEST_IMAGES.length)} of {TEST_IMAGES.length}</span>
            <span style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.38rem", color: "#7a6a42",
            }}>~{Math.round((TEST_IMAGES.length - results.length) * 30)}s remaining</span>
          </div>
          <div style={{
            width: "100%", height: 10, borderRadius: 2, overflow: "hidden",
            background: "#aaa8a0",
            border: "2px solid #888",
            borderTop: "2px solid #bbb",
          }}>
            <div style={{
              height: "100%", borderRadius: 1,
              width: `${(results.length / TEST_IMAGES.length) * 100}%`,
              background: "linear-gradient(90deg, #2a8a5a, #2a5a8a, #7a3af0)",
              transition: "width 1s ease",
            }} />
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{
        background: "#c8c0b0",
        border: "2px solid #999",
        borderTop: "2px solid #bbb",
        borderLeft: "2px solid #b0b0b0",
        borderRadius: 2,
        padding: "6px 12px",
        marginBottom: 8,
      }}>
        {StageLegend}
      </div>

      {/* Image pipeline cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 10 }}>
        {TEST_IMAGES.map((img, i) => (
          <ImageCard key={i} index={i} pixels={img} label={TEST_LABELS[i]}
            result={imageStates[i].result} status={imageStates[i].status}
            activeStage={imageStates[i].activeStage} isCurrent={currentIdx === i && running} />
        ))}
      </div>

      {/* Summary stat cards */}
      {doneResults.length > 0 && (
        <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
          {[
            {
              title: "Accuracy",
              value: `${correctCount}/${doneResults.length}`,
              sub: `${(accuracy * 100).toFixed(0)}%`,
              borderT: accuracy >= 0.8 ? "#44aa66" : accuracy >= 0.5 ? "#d4a800" : "#cc4444",
              borderB: accuracy >= 0.8 ? "#226644" : accuracy >= 0.5 ? "#7a5e00" : "#882222",
              color: accuracy >= 0.8 ? "#155533" : accuracy >= 0.5 ? "#5a3a00" : "#771111",
              bg: accuracy >= 0.8 ? "#c8e8c8" : accuracy >= 0.5 ? "#e8d8a0" : "#e8c0c0",
            },
            {
              title: "Avg Time",
              value: `${(avgTime / 1000).toFixed(1)}s`,
              sub: "per image",
              borderT: "#4a7acc", borderB: "#1a3a8a",
              color: "#1a3a8a", bg: "#c8d4e8",
            },
            {
              title: "Total",
              value: `${(totalTime / 1000).toFixed(1)}s`,
              sub: `${doneResults.length} images`,
              borderT: "#8a5acc", borderB: "#4a1a8a",
              color: "#4a1a8a", bg: "#d8c8e8",
            },
            {
              title: "Library",
              value: "OpenFHE",
              sub: "BFV scheme",
              borderT: "#2a9aaa", borderB: "#0a5a6a",
              color: "#0a5a6a", bg: "#c0dce0",
            },
          ].map(({ title, value, sub, borderT, borderB, color, bg }) => (
            <div key={title} style={{
              flex: 1, minWidth: 100,
              background: bg,
              borderTop: `3px solid ${borderT}`,
              borderLeft: `2px solid ${borderT}`,
              borderBottom: `3px solid ${borderB}`,
              borderRight: `2px solid ${borderB}`,
              borderRadius: 2,
              padding: "10px 10px 8px",
              textAlign: "center",
            }}>
              <div style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.36rem", letterSpacing: "0.14em",
                color: color, textTransform: "uppercase",
                marginBottom: 5, opacity: 0.7,
              }}>{title}</div>
              <div style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "1.1rem", letterSpacing: "0.04em",
                color,
              }}>{value}</div>
              <div style={{
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "0.32rem", color, opacity: 0.6,
                marginTop: 3,
              }}>{sub}</div>
            </div>
          ))}
        </div>
      )}

      {/* Bar chart */}
      {doneResults.length > 1 && (
        <div style={{
          background: "#c8c0b0",
          border: "2px solid #999",
          borderTop: "2px solid #bbb",
          borderLeft: "2px solid #b0b0b0",
          borderRadius: 2,
          padding: "10px 12px",
          marginBottom: 10,
        }}>
          <div style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "0.4rem", letterSpacing: "0.14em",
            color: "#5a4a22", textTransform: "uppercase",
            marginBottom: 8,
          }}>
            Per-Image Inference Time
            <span style={{ color: "#7a6a42", marginLeft: 8, fontFamily: "system-ui,sans-serif", fontSize: "0.72rem", textTransform: "none", letterSpacing: 0 }}>
              (green = correct, red = wrong)
            </span>
          </div>
          <div style={{ height: 200 }}>
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Error banner */}
      {batchError && (
        <div style={{
          background: "#e8c0c0", color: "#771111",
          border: "2px solid #aa3333",
          borderTop: "2px solid #cc5555",
          borderRadius: 2, padding: "8px 12px",
          fontFamily: "'Press Start 2P', monospace",
          fontSize: "0.38rem", letterSpacing: "0.08em",
          textAlign: "center", marginBottom: 8,
        }}>⚠ {batchError}</div>
      )}

      {/* Run again */}
      {!running && results.length > 0 && (
        <div style={{ textAlign: "center" }}>
          <button
            onClick={handleRun}
            style={{
              fontFamily: "'Press Start 2P', monospace",
              fontSize: "0.48rem", letterSpacing: "0.1em",
              padding: "7px 20px",
              background: "linear-gradient(180deg, #5a5a5a, #3a3a3a)",
              border: "2px solid #222",
              borderTop: "2px solid #888",
              borderLeft: "2px solid #777",
              borderRadius: 2, cursor: "pointer",
              color: "#ddd",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "linear-gradient(180deg, #6a6a6a, #4a4a4a)"}
            onMouseLeave={e => e.currentTarget.style.background = "linear-gradient(180deg, #5a5a5a, #3a3a3a)"}
          >▶ Run Again</button>
        </div>
      )}
    </div>
  );
}
