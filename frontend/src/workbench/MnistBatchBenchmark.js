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
        borderRadius: 4,
        border: "1px solid #e0e0e0",
        background: "#000",
      }}
    />
  );
}

/* ── Stage pipeline mini-bar for one image ── */
function StagePipeline({ activeStage, done, error }) {
  return (
    <div className="flex items-center gap-[3px]">
      {STAGES.map((stage, i) => {
        let bg, opacity;
        if (error) {
          bg = "#ef4444";
          opacity = 0.3;
        } else if (done) {
          bg = stage.color;
          opacity = 1;
        } else if (activeStage > i) {
          /* completed stage */
          bg = stage.color;
          opacity = 1;
        } else if (activeStage === i) {
          /* active stage – wider */
          bg = stage.color;
          opacity = 1;
        } else {
          bg = "#e0e0e0";
          opacity = 0.6;
        }

        const isActive = activeStage === i && !done && !error;

        return (
          <div
            key={stage.id}
            className="relative group"
            style={{
              width: isActive ? 22 : 14,
              height: 7,
              borderRadius: 3,
              background: bg,
              opacity,
              transition: "all 0.4s ease",
            }}
          >
            {/* Tooltip on hover */}
            <span
              className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-1.5 py-0.5 rounded text-[8px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10"
              style={{ background: "#333", color: "#fff" }}
            >
              {stage.label}
            </span>
            {/* Pulsing glow on active stage */}
            {isActive && (
              <span
                className="absolute inset-0 rounded-sm animate-pulse"
                style={{ background: stage.color, opacity: 0.45 }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Per-image card ── */
function ImageCard({ index, pixels, label, result, status, activeStage, isCurrent }) {
  const isDone = status === "done";
  const isError = status === "error";
  const isQueued = status === "queued";
  const isIdle = status === "idle";

  let borderColor = "#e5e5e5";
  let bg = "#fff";
  if (isCurrent) {
    borderColor = STAGES[Math.max(0, activeStage)]?.color || "#6366f1";
    bg = "#fffbeb";
  } else if (isDone && result?.correct) {
    borderColor = "#10b981";
    bg = "#f0fdf4";
  } else if (isDone && !result?.correct) {
    borderColor = "#ef4444";
    bg = "#fef2f2";
  } else if (isError) {
    borderColor = "#ef4444";
    bg = "#fef2f2";
  }

  const dimmed = isQueued || isIdle;

  return (
    <div
      className="flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300"
      style={{
        background: bg,
        border: `1.5px solid ${borderColor}`,
        opacity: dimmed ? 0.45 : 1,
        boxShadow: isCurrent ? `0 0 8px ${borderColor}44` : "none",
      }}
    >
      {/* Index badge */}
      <span
        className="text-[10px] font-mono font-bold w-4 text-center shrink-0"
        style={{ color: "#999" }}
      >
        {index + 1}
      </span>

      {/* Thumbnail */}
      <DigitThumbnail pixels={pixels} size={28} />

      {/* True label */}
      <span
        className="font-mono font-bold text-sm w-5 text-center shrink-0"
        style={{ color: "#555" }}
      >
        {label}
      </span>

      {/* Arrow → */}
      <svg className="w-4 h-4 shrink-0" style={{ color: "#ccc" }} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>

      {/* Pipeline stages — the heart of the "transparency" */}
      <div className="flex-1 min-w-0">
        {dimmed ? (
          /* Idle / queued – grey placeholder bar */
          <div className="flex items-center gap-1.5">
            <div className="flex gap-[3px]">
              {STAGES.map((s) => (
                <div
                  key={s.id}
                  style={{ width: 14, height: 7, borderRadius: 3, background: "#e5e5e5" }}
                />
              ))}
            </div>
            <span className="text-[9px]" style={{ color: "#bbb" }}>
              {isQueued ? "Queued" : "Waiting"}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <StagePipeline activeStage={activeStage} done={isDone} error={isError} />
            {isCurrent && !isDone && (
              <div className="flex items-center gap-1">
                {/* Pulsing dot */}
                <span className="relative flex h-2.5 w-2.5">
                  <span
                    className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                    style={{ background: STAGES[Math.max(0, activeStage)]?.color }}
                  />
                  <span
                    className="relative inline-flex rounded-full h-2.5 w-2.5"
                    style={{ background: STAGES[Math.max(0, activeStage)]?.color }}
                  />
                </span>
                <span
                  className="text-[9px] font-medium whitespace-nowrap"
                  style={{ color: STAGES[Math.max(0, activeStage)]?.color }}
                >
                  {STAGES[Math.max(0, activeStage)]?.label}…
                </span>
              </div>
            )}
            {isDone && !isError && (
              <span className="text-[9px] font-medium" style={{ color: "#10b981" }}>✓ Done</span>
            )}
            {isError && (
              <span className="text-[9px] font-medium" style={{ color: "#ef4444" }}>✗ Failed</span>
            )}
          </div>
        )}
      </div>

      {/* Arrow → */}
      <svg className="w-4 h-4 shrink-0" style={{ color: "#ccc" }} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>

      {/* Prediction result */}
      <div className="w-20 text-right shrink-0">
        {isDone && result && !isError ? (
          <div className="flex items-center justify-end gap-1.5">
            <span
              className="font-mono font-bold text-sm"
              style={{ color: result.correct ? "#10b981" : "#ef4444" }}
            >
              {result.predicted}
            </span>
            <span
              className="text-[9px] px-1 py-0.5 rounded font-bold"
              style={{
                background: result.correct ? "#d1fae5" : "#fee2e2",
                color: result.correct ? "#059669" : "#dc2626",
              }}
            >
              {result.correct ? "✓" : "✗"}
            </span>
          </div>
        ) : isCurrent ? (
          <span className="text-[9px]" style={{ color: "#d97706" }}>Running…</span>
        ) : isError ? (
          <span className="text-[9px]" style={{ color: "#ef4444" }}>Error</span>
        ) : (
          <span className="text-[9px]" style={{ color: "#ccc" }}>—</span>
        )}
      </div>

      {/* Time */}
      <div className="w-14 text-right shrink-0">
        {isDone && result ? (
          <span className="text-[10px] font-mono" style={{ color: "#888" }}>
            {(result.totalMs / 1000).toFixed(1)}s
          </span>
        ) : (
          <span className="text-[10px]" style={{ color: "#ddd" }}>—</span>
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

  /* ─── Stage legend (reused in both views) ─── */
  const StageLegend = (
    <div className="flex items-center justify-center gap-1 flex-wrap">
      {STAGES.map((s) => (
        <div key={s.id} className="flex items-center gap-1 px-1.5">
          <div style={{ width: 10, height: 7, borderRadius: 3, background: s.color }} />
          <span className="text-[9px]" style={{ color: "#999" }}>{s.label}</span>
        </div>
      ))}
    </div>
  );

  /* ━━━ Empty state ━━━ */
  if (!running && results.length === 0) {
    return (
      <div className="py-6">
        <div className="text-center mb-5">
          <p className="text-sm mb-1" style={{ color: "#888" }}>
            Run all 10 MNIST test digits through encrypted CNN inference
          </p>
          <p className="text-[10px]" style={{ color: "#bbb" }}>
            Each image takes ~30s • Total: ~5 minutes • OpenFHE BFV scheme
          </p>
        </div>

        {/* Stage legend */}
        <div className="mb-5">{StageLegend}</div>

        {/* Preview: 10 digit cards in idle state */}
        <div className="space-y-1.5 mb-6">
          {TEST_IMAGES.map((img, i) => (
            <ImageCard
              key={i}
              index={i}
              pixels={img}
              label={TEST_LABELS[i]}
              result={null}
              status="idle"
              activeStage={-1}
              isCurrent={false}
            />
          ))}
        </div>

        <div className="text-center">
          <button
            onClick={handleRun}
            className="px-5 py-2.5 rounded-lg text-sm font-medium text-white shadow-sm hover:shadow transition-all hover:scale-[1.02] active:scale-[0.98]"
            style={{ background: "#1a1a2e" }}
          >
            ▶ Run MNIST Batch Benchmark
          </button>
        </div>

        {/* Error banner */}
        {batchError && (
          <div
            className="mt-4 rounded-lg p-3 text-center text-xs"
            style={{ background: "#fef2f2", border: "1px solid #fecaca", color: "#dc2626" }}
          >
            ⚠ {batchError}
          </div>
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
          r.correct ? "rgba(16,185,129,0.7)" : "rgba(239,68,68,0.7)"
        ),
        borderColor: doneResults.map((r) =>
          r.correct ? "#10b981" : "#ef4444"
        ),
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#333",
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
      x: { grid: { display: false }, ticks: { color: "#888", font: { size: 11 } } },
      y: {
        grid: { color: "rgba(0,0,0,0.06)" },
        ticks: { color: "#999", font: { size: 10 }, callback: (v) => `${(v / 1000).toFixed(0)}s` },
        title: { display: true, text: "Inference Time", color: "#aaa", font: { size: 10 } },
      },
    },
  };

  return (
    <div>
      {/* ── Header: stage legend + stop button ── */}
      <div className="flex items-center justify-between mb-3">
        {StageLegend}
        {running && (
          <button
            onClick={handleStop}
            className="text-[10px] px-2.5 py-1 rounded border hover:bg-red-50 transition-colors"
            style={{ borderColor: "#fca5a5", color: "#dc2626" }}
          >
            ■ Stop
          </button>
        )}
      </div>

      {/* ── Overall progress bar (only while running) ── */}
      {running && (
        <div className="mb-3">
          <div className="flex items-center justify-between text-[10px] mb-1" style={{ color: "#888" }}>
            <span>
              Image <strong>{Math.min(results.length + 1, TEST_IMAGES.length)}</strong> of {TEST_IMAGES.length}
            </span>
            <span>
              ~{Math.round((TEST_IMAGES.length - results.length) * 30)}s remaining
            </span>
          </div>
          <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "#e5e5e5" }}>
            <div
              className="h-full rounded-full transition-all duration-1000"
              style={{
                width: `${(results.length / TEST_IMAGES.length) * 100}%`,
                background: "linear-gradient(90deg, #0db7c4, #7b3ff2, #e03e52)",
              }}
            />
          </div>
        </div>
      )}

      {/* ── Image pipeline cards (the main transparent view) ── */}
      <div className="space-y-1.5 mb-5">
        {TEST_IMAGES.map((img, i) => (
          <ImageCard
            key={i}
            index={i}
            pixels={img}
            label={TEST_LABELS[i]}
            result={imageStates[i].result}
            status={imageStates[i].status}
            activeStage={imageStates[i].activeStage}
            isCurrent={currentIdx === i && running}
          />
        ))}
      </div>

      {/* ── Summary cards (appear after first result) ── */}
      {doneResults.length > 0 && (
        <div className="flex gap-3 mb-5 flex-wrap">
          {/* Accuracy */}
          <div
            className="flex-1 min-w-[110px] rounded-lg p-3 text-center"
            style={{
              background: "#fff",
              border: `2px solid ${accuracy >= 0.8 ? "#10b981" : accuracy >= 0.5 ? "#f59e0b" : "#ef4444"}`,
            }}
          >
            <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>Accuracy</p>
            <p
              className="text-2xl font-bold font-mono"
              style={{ color: accuracy >= 0.8 ? "#10b981" : accuracy >= 0.5 ? "#f59e0b" : "#ef4444" }}
            >
              {correctCount}/{doneResults.length}
            </p>
            <p className="text-[10px] mt-0.5" style={{ color: "#bbb" }}>
              {(accuracy * 100).toFixed(0)}%
            </p>
          </div>

          {/* Avg Time */}
          <div className="flex-1 min-w-[110px] rounded-lg p-3 text-center" style={{ background: "#fff", border: "2px solid #3b82f6" }}>
            <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>Avg Time</p>
            <p className="text-2xl font-bold font-mono" style={{ color: "#3b82f6" }}>
              {(avgTime / 1000).toFixed(1)}
              <span className="text-xs font-normal ml-0.5" style={{ color: "#bbb" }}>s</span>
            </p>
            <p className="text-[10px] mt-0.5" style={{ color: "#bbb" }}>per image</p>
          </div>

          {/* Total Time */}
          <div className="flex-1 min-w-[110px] rounded-lg p-3 text-center" style={{ background: "#fff", border: "2px solid #8b5cf6" }}>
            <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>Total</p>
            <p className="text-2xl font-bold font-mono" style={{ color: "#8b5cf6" }}>
              {(totalTime / 1000).toFixed(1)}
              <span className="text-xs font-normal ml-0.5" style={{ color: "#bbb" }}>s</span>
            </p>
            <p className="text-[10px] mt-0.5" style={{ color: "#bbb" }}>
              {doneResults.length} image{doneResults.length !== 1 ? "s" : ""}
            </p>
          </div>

          {/* Library */}
          <div className="flex-1 min-w-[110px] rounded-lg p-3 text-center" style={{ background: "#fff", border: "2px solid #0db7c4" }}>
            <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#999" }}>Library</p>
            <p className="text-lg font-bold" style={{ color: "#0db7c4" }}>OpenFHE</p>
            <p className="text-[10px] mt-0.5" style={{ color: "#bbb" }}>BFV scheme</p>
          </div>
        </div>
      )}

      {/* ── Bar chart ── */}
      {doneResults.length > 1 && (
        <div className="rounded-lg p-4 mb-4" style={{ background: "#fff", border: "1px solid #e5e5e5" }}>
          <p className="text-[10px] uppercase tracking-wider mb-3 font-medium" style={{ color: "#888" }}>
            Per-Image Inference Time
            <span className="font-normal ml-2" style={{ color: "#bbb" }}>(green = correct, red = wrong)</span>
          </p>
          <div style={{ height: 200 }}>
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {/* ── Error banner ── */}
      {batchError && (
        <div
          className="rounded-lg p-3 mb-4 text-center text-xs"
          style={{ background: "#fef2f2", border: "1px solid #fecaca", color: "#dc2626" }}
        >
          ⚠ {batchError}
        </div>
      )}

      {/* ── Run again ── */}
      {!running && results.length > 0 && (
        <div className="text-center">
          <button
            onClick={handleRun}
            className="px-4 py-1.5 rounded-lg text-xs font-medium border hover:bg-gray-50 transition-colors"
            style={{ borderColor: "#ccc", color: "#666" }}
          >
            ▶ Run Again
          </button>
        </div>
      )}
    </div>
  );
}
