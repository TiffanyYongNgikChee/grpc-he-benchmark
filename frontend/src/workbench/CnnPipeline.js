import { useEffect, useRef, useState } from "react";

/**
 * CnnPipeline — Neural network visualisation for the encrypted CNN.
 *
 * Layers are grouped into vertical columns with circle "neuron" nodes
 * connected by weighted lines.
 *
 * Supports three node states:
 *   "idle"       → muted fill, grey stroke
 *   "processing" → pulsing glow animation, category colour
 *   "done"       → solid category colour with timing
 *
 * Groups (columns):
 *   Input | Encrypt | Conv Block 1 | Conv Block 2 | FC | Decrypt | Output
 */

/* ── Layer definitions ── */
const LAYERS = [
  { id: "input",   label: "Input",   sub: "28x28 pixels",  category: "io",     key: null },
  { id: "encrypt", label: "Encrypt", sub: "BFV scheme",    category: "crypto", key: "encryptionMs" },
  { id: "conv1",   label: "Conv1",   sub: "5x5 kernel",    category: "conv",   key: "conv1Ms" },
  { id: "bias1",   label: "Bias1",   sub: "+bias",         category: "conv",   key: "bias1Ms" },
  { id: "relu1",   label: "x\u00B2", sub: "activation",    category: "act",    key: "act1Ms" },
  { id: "pool1",   label: "Pool1",   sub: "2x2 avg",       category: "pool",   key: "pool1Ms" },
  { id: "conv2",   label: "Conv2",   sub: "5x5 kernel",    category: "conv",   key: "conv2Ms" },
  { id: "bias2",   label: "Bias2",   sub: "+bias",         category: "conv",   key: "bias2Ms" },
  { id: "relu2",   label: "x\u00B2", sub: "activation",    category: "act",    key: "act2Ms" },
  { id: "pool2",   label: "Pool2",   sub: "2x2 avg",       category: "pool",   key: "pool2Ms" },
  { id: "fc",      label: "FC",      sub: "16 to 10",      category: "fc",     key: "fcMs" },
  { id: "biasfc",  label: "BiasFc",  sub: "+bias",         category: "fc",     key: "biasFcMs" },
  { id: "decrypt", label: "Decrypt", sub: "BFV scheme",    category: "crypto", key: "decryptionMs" },
  { id: "output",  label: "Output",  sub: "argmax",        category: "io",     key: null },
];

/* ── Category colours ── */
const CATEGORY_COLORS = {
  io:     { stroke: "#0aa35e", fill: "#e6f7ef", active: "#0aa35e", glow: "rgba(10,163,94,0.3)",  pulse: "#0aa35e" },
  crypto: { stroke: "#0db7c4", fill: "#e6f8fa", active: "#0db7c4", glow: "rgba(13,183,196,0.3)",  pulse: "#0db7c4" },
  conv:   { stroke: "#7b3ff2", fill: "#f0ebfe", active: "#7b3ff2", glow: "rgba(123,63,242,0.3)",  pulse: "#7b3ff2" },
  act:    { stroke: "#e68a00", fill: "#fef3e2", active: "#e68a00", glow: "rgba(230,138,0,0.3)",   pulse: "#e68a00" },
  pool:   { stroke: "#e68a00", fill: "#fef3e2", active: "#e68a00", glow: "rgba(230,138,0,0.3)",   pulse: "#e68a00" },
  fc:     { stroke: "#e03e52", fill: "#fde8eb", active: "#e03e52", glow: "rgba(224,62,82,0.3)",   pulse: "#e03e52" },
};

/* ── Group layers into vertical columns ── */
const GROUPS = [
  { title: "INPUT",        layers: [0] },
  { title: "ENCRYPT",      layers: [1] },
  { title: "CONV BLOCK 1", layers: [2, 3, 4, 5] },
  { title: "CONV BLOCK 2", layers: [6, 7, 8, 9] },
  { title: "FC",           layers: [10, 11] },
  { title: "DECRYPT",      layers: [12] },
  { title: "OUTPUT",       layers: [13] },
];

/* ── Geometry ── */
const R         = 24;          // bubble radius
const BUBBLE_GAP = 14;         // vertical gap between bubbles in a group
const GROUP_GAP  = 80;         // horizontal gap between groups
const PAD_X      = 30;
const PAD_TOP    = 40;         // top padding for group titles
const MAX_STACK  = 4;          // max bubbles in one group

const GROUP_W  = R * 2;
const SVG_W    = PAD_X * 2 + GROUPS.length * GROUP_W + (GROUPS.length - 1) * GROUP_GAP;
const COL_H    = MAX_STACK * (R * 2 + BUBBLE_GAP) - BUBBLE_GAP;
const SVG_H    = PAD_TOP + COL_H + 60;

/* Position helpers */
function groupCenterX(gi) {
  return PAD_X + R + gi * (GROUP_W + GROUP_GAP);
}
function bubbleY(indexInGroup, totalInGroup) {
  const stackH = totalInGroup * (R * 2 + BUBBLE_GAP) - BUBBLE_GAP;
  const startY = PAD_TOP + (COL_H - stackH) / 2 + R;
  return startY + indexInGroup * (R * 2 + BUBBLE_GAP);
}

/* Build a lookup: layer index → {cx, cy} */
function buildPositions() {
  const pos = {};
  GROUPS.forEach((g, gi) => {
    const cx = groupCenterX(gi);
    g.layers.forEach((li, bi) => {
      pos[li] = { cx, cy: bubbleY(bi, g.layers.length) };
    });
  });
  return pos;
}
const POS = buildPositions();

/**
 * Resolve per-layer status.
 * Uses layerStatus map if provided, else falls back to legacy activeStep.
 */
function getNodeStatus(layerIndex, layerStatus, activeStep) {
  const layer = LAYERS[layerIndex];

  // Input is always "done" once anything starts
  if (layer.id === "input") return activeStep >= 0 ? "done" : "idle";

  // Output is "done" only when everything is done
  if (layer.id === "output") {
    if (layerStatus && Object.values(layerStatus).every((s) => s === "done")) return "done";
    if (activeStep >= LAYERS.length) return "done";
    return "idle";
  }

  // Check layerStatus map first (from useInferenceProgress)
  if (layerStatus && layerStatus[layer.id]) return layerStatus[layer.id];

  // Fallback to old activeStep logic
  if (activeStep >= LAYERS.length) return "done";
  if (layerIndex > 0 && layerIndex <= activeStep) return "done";
  return "idle";
}

export default function CnnPipeline({
  timings,
  activeStep = -1,
  hovered,
  onHover,
  layerStatus = null,
}) {
  const svgRef = useRef(null);

  /* Animated dash offset for flowing connections */
  const [dashOff, setDashOff] = useState(0);
  useEffect(() => {
    let raf;
    const tick = () => { setDashOff(p => (p - 0.8) % 24); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  /* Pulse phase for processing nodes (0→1→0 cycle) */
  const [pulsePhase, setPulsePhase] = useState(0);
  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = () => {
      const t = ((performance.now() - start) % 1200) / 1200;
      setPulsePhase(Math.sin(t * Math.PI * 2) * 0.5 + 0.5);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  /* Build all inter-group connections (every bubble in group i → every bubble in group i+1) */
  const connections = [];
  for (let gi = 0; gi < GROUPS.length - 1; gi++) {
    const srcLayers = GROUPS[gi].layers;
    const dstLayers = GROUPS[gi + 1].layers;
    for (const si of srcLayers) {
      for (const di of dstLayers) {
        connections.push({ from: si, to: di });
      }
    }
  }

  return (
    <div className="w-full overflow-x-auto">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full"
        style={{ minWidth: 860 }}
      >
        <defs>
          {/* Glow filters */}
          {Object.entries(CATEGORY_COLORS).map(([cat, c]) => (
            <filter key={cat} id={`glow-${cat}`} x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor={c.glow} floodOpacity="1" />
            </filter>
          ))}
          {/* Processing pulse filters — larger double-glow */}
          {Object.entries(CATEGORY_COLORS).map(([cat, c]) => (
            <filter key={`pulse-${cat}`} id={`pulse-${cat}`} x="-80%" y="-80%" width="260%" height="260%">
              <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor={c.pulse} floodOpacity="0.8" />
              <feDropShadow dx="0" dy="0" stdDeviation="14" floodColor={c.pulse} floodOpacity="0.3" />
            </filter>
          ))}
        </defs>

        {/* ── Group titles ── */}
        {GROUPS.map((g, gi) => (
          <text
            key={`title-${gi}`}
            x={groupCenterX(gi)}
            y={14}
            textAnchor="middle"
            fill="#999"
            fontSize={9}
            fontWeight={500}
            fontFamily="'Roboto', sans-serif"
            letterSpacing="0.05em"
          >
            {g.title}
          </text>
        ))}

        {/* ── Encrypted domain bracket ── */}
        {(() => {
          const x1 = groupCenterX(1) - R - 10;
          const x2 = groupCenterX(5) + R + 10;
          const y  = 26;
          return (
            <g opacity={0.45}>
              <line x1={x1} y1={y} x2={x2} y2={y} stroke="#0db7c4" strokeWidth={1.5} />
              <line x1={x1} y1={y - 4} x2={x1} y2={y + 4} stroke="#0db7c4" strokeWidth={1.5} />
              <line x1={x2} y1={y - 4} x2={x2} y2={y + 4} stroke="#0db7c4" strokeWidth={1.5} />
              <text x={(x1 + x2) / 2} y={y - 6} textAnchor="middle" fill="#0db7c4" fontSize={8} fontWeight={600}>
                ENCRYPTED DOMAIN
              </text>
            </g>
          );
        })()}

        {/* ── Connections (weighted lines between groups) ── */}
        {connections.map(({ from, to }, ci) => {
          const p1 = POS[from];
          const p2 = POS[to];
          const srcStatus = getNodeStatus(from, layerStatus, activeStep);
          const dstStatus = getNodeStatus(to, layerStatus, activeStep);
          const srcDone = srcStatus === "done";
          const dstProcessing = dstStatus === "processing";
          const dstDone = dstStatus === "done";
          const allDone = srcDone && dstDone;
          const flowing = srcDone && dstProcessing;

          const weight = allDone || flowing ? 2.2 : 1;
          const color  = allDone ? "#f4743a" : flowing ? "#f4743a" : "#ddd";

          return (
            <g key={`conn-${ci}`}>
              <line
                x1={p1.cx + R} y1={p1.cy}
                x2={p2.cx - R} y2={p2.cy}
                stroke={color}
                strokeWidth={weight}
                strokeDasharray={flowing ? "6 4" : "none"}
                strokeDashoffset={flowing ? dashOff : 0}
                opacity={allDone || flowing ? 0.85 : 0.4}
                style={{ transition: "stroke 0.3s, opacity 0.3s" }}
              />
              {/* Data particle flowing along active connection */}
              {flowing && (
                <DataParticle
                  x1={p1.cx + R} y1={p1.cy}
                  x2={p2.cx - R} y2={p2.cy}
                  phase={pulsePhase}
                />
              )}
            </g>
          );
        })}

        {/* ── Neuron bubbles ── */}
        {LAYERS.map((layer, i) => {
          const { cx, cy } = POS[i];
          const cat = CATEGORY_COLORS[layer.category];
          const status = getNodeStatus(i, layerStatus, activeStep);
          const isHov = hovered === i;
          const timeMs = timings && layer.key ? timings[layer.key] : null;

          const isProcessing = status === "processing";
          const isDone = status === "done";

          // Slightly larger radius during processing pulse
          const dynamicR = isProcessing ? R + pulsePhase * 2 : R;

          let bubbleFill, bubbleStroke, filterAttr, labelColor;
          if (isProcessing) {
            bubbleFill   = cat.active;
            bubbleStroke  = cat.stroke;
            filterAttr    = `url(#pulse-${layer.category})`;
            labelColor    = "#fff";
          } else if (isDone) {
            bubbleFill   = cat.active;
            bubbleStroke  = isHov ? "#333" : cat.stroke;
            filterAttr    = `url(#glow-${layer.category})`;
            labelColor    = "#fff";
          } else {
            bubbleFill   = cat.fill;
            bubbleStroke  = isHov ? "#333" : cat.stroke;
            filterAttr    = isHov ? `url(#glow-${layer.category})` : "none";
            labelColor    = cat.stroke;
          }

          return (
            <g
              key={layer.id}
              onMouseEnter={() => onHover?.(i)}
              onMouseLeave={() => onHover?.(null)}
              style={{ cursor: "pointer" }}
            >
              {/* Processing ring animation — expanding ring around active node */}
              {isProcessing && (
                <circle
                  cx={cx} cy={cy}
                  r={R + 6 + pulsePhase * 4}
                  fill="none"
                  stroke={cat.pulse}
                  strokeWidth={1.5}
                  opacity={0.25 + pulsePhase * 0.25}
                />
              )}

              {/* Circle node */}
              <circle
                cx={cx} cy={cy} r={dynamicR}
                fill={bubbleFill}
                stroke={bubbleStroke}
                strokeWidth={isHov ? 2.5 : isProcessing ? 2 : 1.5}
                filter={filterAttr}
                opacity={(isDone || isProcessing || i === 0 || i === LAYERS.length - 1) ? 1 : 0.9}
                style={{ transition: "all 0.3s ease" }}
              />

              {/* Label inside bubble */}
              <text
                x={cx} y={cy + (timeMs !== null && isDone ? -4 : 1)}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={labelColor}
                fontSize={10}
                fontWeight={600}
                fontFamily="'Roboto', sans-serif"
                style={{ pointerEvents: "none", transition: "fill 0.3s" }}
              >
                {layer.label}
              </text>

              {/* Timing inside bubble (shown when done with results) */}
              {timeMs !== null && isDone && (
                <text
                  x={cx} y={cy + 10}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="rgba(255,255,255,0.85)"
                  fontSize={8}
                  fontFamily="'Roboto Mono', monospace"
                  style={{ pointerEvents: "none" }}
                >
                  {timeMs.toFixed(1)}ms
                </text>
              )}

              {/* "running…" text for processing nodes */}
              {isProcessing && (
                <text
                  x={cx} y={cy + 10}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="rgba(255,255,255,0.65)"
                  fontSize={7}
                  fontFamily="'Roboto', sans-serif"
                  style={{ pointerEvents: "none" }}
                >
                  running…
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * DataParticle — a small glowing dot that travels along a connection line
 * to visually represent encrypted data flowing between layers.
 */
function DataParticle({ x1, y1, x2, y2, phase }) {
  const px = x1 + (x2 - x1) * phase;
  const py = y1 + (y2 - y1) * phase;
  return (
    <circle cx={px} cy={py} r={3} fill="#f4743a" opacity={0.85}>
      <animate attributeName="r" values="2;4;2" dur="0.8s" repeatCount="indefinite" />
    </circle>
  );
}

export { LAYERS, CATEGORY_COLORS };
