import { useEffect, useRef, useState } from "react";

/**
 * CnnPipeline — Interactive SVG visualization of the CNN architecture,
 * inspired by TensorFlow Playground's network diagram.
 *
 * Shows the full encrypted inference pipeline:
 *   Input → 🔒 Encrypt → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → FC → Decrypt 🔓 → Output
 *
 * Layers light up sequentially when timing data is provided, with animated
 * connections showing data flow between layers.
 *
 * Props:
 *   timings    — prediction result object with *Ms fields (null if no result yet)
 *   activeStep — index of the currently-animating layer (-1 = idle, 12 = done)
 *   hovered    — index of the currently hovered layer (or null)
 *   onHover    — callback(index | null)
 */

/* ── Layer definitions ── */
const LAYERS = [
  { id: "input",      label: "Input",      sub: "28×28",        category: "io",      key: null },
  { id: "encrypt",    label: "Encrypt",    sub: "🔒 BFV",       category: "crypto",  key: "encryptionMs" },
  { id: "conv1",      label: "Conv1",      sub: "5×5, 5 filters", category: "conv",  key: "conv1Ms" },
  { id: "bias1",      label: "Bias1",      sub: "+bias",        category: "conv",    key: "bias1Ms" },
  { id: "relu1",      label: "ReLU1",      sub: "x²",           category: "act",     key: "act1Ms" },
  { id: "pool1",      label: "Pool1",      sub: "2×2 avg",      category: "pool",    key: "pool1Ms" },
  { id: "conv2",      label: "Conv2",      sub: "5×5, 10 filters", category: "conv", key: "conv2Ms" },
  { id: "bias2",      label: "Bias2",      sub: "+bias",        category: "conv",    key: "bias2Ms" },
  { id: "relu2",      label: "ReLU2",      sub: "x²",           category: "act",     key: "act2Ms" },
  { id: "pool2",      label: "Pool2",      sub: "2×2 avg",      category: "pool",    key: "pool2Ms" },
  { id: "fc",         label: "FC",         sub: "→ 10 classes",  category: "fc",      key: "fcMs" },
  { id: "biasfc",     label: "BiasFc",     sub: "+bias",        category: "fc",      key: "biasFcMs" },
  { id: "decrypt",    label: "Decrypt",    sub: "🔓 BFV",       category: "crypto",  key: "decryptionMs" },
  { id: "output",     label: "Output",     sub: "logits",       category: "io",      key: null },
];

/* ── Category colour map (matches TF playground colour language) ── */
const CATEGORY_COLORS = {
  io:     { fill: "#334155", stroke: "#64748b", active: "#10b981", glow: "rgba(16,185,129,0.5)" },
  crypto: { fill: "#1e293b", stroke: "#06b6d4", active: "#22d3ee", glow: "rgba(34,211,238,0.5)" },
  conv:   { fill: "#1e293b", stroke: "#8b5cf6", active: "#a78bfa", glow: "rgba(139,92,246,0.5)" },
  act:    { fill: "#1e293b", stroke: "#f59e0b", active: "#fbbf24", glow: "rgba(245,158,11,0.5)" },
  pool:   { fill: "#1e293b", stroke: "#f59e0b", active: "#fbbf24", glow: "rgba(251,191,36,0.5)" },
  fc:     { fill: "#1e293b", stroke: "#f43f5e", active: "#fb7185", glow: "rgba(244,63,94,0.5)" },
};

/* ── Geometry constants ── */
const NODE_W = 72;
const NODE_H = 52;
const GAP_X  = 14;
const PAD_Y  = 30;
const PAD_X  = 20;

/* Total SVG dimensions */
const COLS = LAYERS.length;
const SVG_W = PAD_X * 2 + COLS * NODE_W + (COLS - 1) * GAP_X;
const SVG_H = PAD_Y * 2 + NODE_H + 40; // extra for sub-labels

/* Node position helper */
function nodeX(i) { return PAD_X + i * (NODE_W + GAP_X); }
function nodeY()  { return PAD_Y; }

export default function CnnPipeline({ timings, activeStep = -1, hovered, onHover }) {
  const svgRef = useRef(null);

  /* Dash-offset animation for active links */
  const [dashOffset, setDashOffset] = useState(0);
  useEffect(() => {
    let raf;
    const animate = () => {
      setDashOffset((prev) => (prev - 0.6) % 20);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full"
        style={{ minWidth: 900 }}
      >
        <defs>
          {/* Arrow marker */}
          <marker
            id="arrowHead"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M0,0 L7,3.5 L0,7 z" fill="#475569" />
          </marker>

          {/* Glow filters per category */}
          {Object.entries(CATEGORY_COLORS).map(([cat, c]) => (
            <filter key={cat} id={`glow-${cat}`} x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={c.glow} floodOpacity="0.8" />
            </filter>
          ))}
        </defs>

        {/* ── Connection lines ── */}
        {LAYERS.slice(0, -1).map((layer, i) => {
          const x1 = nodeX(i) + NODE_W;
          const y1 = nodeY() + NODE_H / 2;
          const x2 = nodeX(i + 1);
          const y2 = y1;
          const isActive = i < activeStep;
          const isFlowing = i === activeStep - 1;

          return (
            <line
              key={`link-${i}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={isActive ? "#10b981" : "#334155"}
              strokeWidth={isActive ? 2.5 : 1.5}
              strokeDasharray={isFlowing ? "6 3" : "none"}
              strokeDashoffset={isFlowing ? dashOffset : 0}
              markerEnd="url(#arrowHead)"
              style={{
                transition: "stroke 0.3s, stroke-width 0.3s",
              }}
            />
          );
        })}

        {/* ── Encrypt / Decrypt boundary markers ── */}
        {[1, 12].map((idx) => {
          const x = nodeX(idx) - GAP_X / 2;
          return (
            <g key={`boundary-${idx}`}>
              <line
                x1={x}
                y1={PAD_Y - 12}
                x2={x}
                y2={PAD_Y + NODE_H + 12}
                stroke={idx === 1 ? "#06b6d4" : "#2dd4bf"}
                strokeWidth={1.5}
                strokeDasharray="4 3"
                opacity={0.5}
              />
              <text
                x={x}
                y={PAD_Y - 16}
                textAnchor="middle"
                fill={idx === 1 ? "#06b6d4" : "#2dd4bf"}
                fontSize={9}
                fontWeight={600}
              >
                {idx === 1 ? "🔒 ENCRYPTED DOMAIN" : "🔓 PLAINTEXT"}
              </text>
            </g>
          );
        })}

        {/* ── Layer nodes ── */}
        {LAYERS.map((layer, i) => {
          const x = nodeX(i);
          const y = nodeY();
          const cat = CATEGORY_COLORS[layer.category];
          const isActive = i > 0 && i <= activeStep;
          const isDone = activeStep >= LAYERS.length;
          const isHovered = hovered === i;
          const timeMs = timings && layer.key ? timings[layer.key] : null;

          const fillColor = (isActive || isDone) ? cat.active : cat.fill;
          const strokeColor = isHovered
            ? "#fff"
            : (isActive || isDone)
            ? cat.active
            : cat.stroke;
          const filterAttr = (isActive || isHovered) ? `url(#glow-${layer.category})` : "none";

          return (
            <g
              key={layer.id}
              onMouseEnter={() => onHover?.(i)}
              onMouseLeave={() => onHover?.(null)}
              style={{ cursor: "pointer" }}
            >
              {/* Node rectangle */}
              <rect
                x={x}
                y={y}
                width={NODE_W}
                height={NODE_H}
                rx={6}
                ry={6}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={isHovered ? 2 : 1.5}
                filter={filterAttr}
                opacity={(isActive || isDone || i === 0) ? 1 : 0.6}
                style={{ transition: "all 0.3s ease" }}
              />

              {/* Layer name */}
              <text
                x={x + NODE_W / 2}
                y={y + (timeMs !== null ? 16 : NODE_H / 2)}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={(isActive || isDone) ? "#0f172a" : "#e2e8f0"}
                fontSize={11}
                fontWeight={600}
                style={{ pointerEvents: "none", transition: "fill 0.3s" }}
              >
                {layer.label}
              </text>

              {/* Timing value (when available) */}
              {timeMs !== null && (
                <text
                  x={x + NODE_W / 2}
                  y={y + 33}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={(isActive || isDone) ? "#1e293b" : "#94a3b8"}
                  fontSize={10}
                  fontWeight={400}
                  fontFamily="'JetBrains Mono', monospace"
                  style={{ pointerEvents: "none" }}
                >
                  {timeMs.toFixed(1)}ms
                </text>
              )}

              {/* Sub-label below node */}
              <text
                x={x + NODE_W / 2}
                y={y + NODE_H + 14}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#64748b"
                fontSize={9}
                style={{ pointerEvents: "none" }}
              >
                {layer.sub}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export { LAYERS, CATEGORY_COLORS };
