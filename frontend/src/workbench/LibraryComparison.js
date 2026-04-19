import { useState } from "react";
import OwlGuide from "./OwlGuide";

/* ═══════════════════════════════════════════════════════
   HABBO-HOTEL PIXEL STYLE  —  Library Comparison Panel
   ═══════════════════════════════════════════════════════ */

const PIXEL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
  @keyframes pixelPop  { 0%{transform:scale(0.88);opacity:0} 100%{transform:scale(1);opacity:1} }
  @keyframes pixelSlideUp { 0%{transform:translateY(18px);opacity:0} 100%{transform:translateY(0);opacity:1} }
  @keyframes pixelBlink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} 99%{opacity:0} }
  @keyframes pixelSpin  { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
  @keyframes pixelMarch { 0%{background-position:0 0} 100%{background-position:28px 0} }
  .px-pop  { animation: pixelPop 0.22s steps(4,end) both; }
  .px-slide{ animation: pixelSlideUp 0.3s steps(6,end) both; }
  .px-blink{ animation: pixelBlink 1s step-end infinite; }
  .px-spin { animation: pixelSpin 1.1s steps(8,end) infinite; }
  .px-march{ animation: pixelMarch 0.6s steps(4,end) infinite; }
`;

/* ── Habbo dark-wood / neon palette ── */
const P = {
  bg:         "#12090200",   // transparent outer
  panelBg:    "#1a0f06",
  panel:      "#241409",
  panelMid:   "#2e1a0c",
  border:     "#5a3510",
  borderHi:   "#c07820",
  borderLo:   "#0e0702",
  floor:      "#3a2210",
  floorAlt:   "#2e1a0c",
  gold:       "#f0c030",
  goldDim:    "#a07820",
  cream:      "#ffe8a0",
  dim:        "#7a5030",
  scanline:   "rgba(0,0,0,0.18)",
  seal:       "#38bdf8",
  sealDark:   "#0369a1",
  sealBg:     "#082032",
  helib:      "#a78bfa",
  helibDark:  "#5b21b6",
  helibBg:    "#1a0d30",
  openfhe:    "#34d399",
  openfheDark:"#065f46",
  openfheBg:  "#052018",
};

const LIB = {
  SEAL:    { color: P.seal,    dark: P.sealDark,    bg: P.sealBg,    scheme: "BFV", madeBy: "Microsoft" },
  HELib:   { color: P.helib,   dark: P.helibDark,   bg: P.helibBg,   scheme: "BGV", madeBy: "IBM" },
  OpenFHE: { color: P.openfhe, dark: P.openfheDark, bg: P.openfheBg, scheme: "BFV", madeBy: "Community" },
};

const OPS = [
  {
    key:   "keyGenTimeMs",
    short: "Key Gen",
    param: "pub+sec+relin",
    color: P.gold,
    what:  "Generates the public/secret key pair and relinearisation keys.",
    how:   "Samples random polynomials in ring Rq. Public key = (-a*s+e, a). Relin keys encode s2 to keep ciphertexts degree-1 after multiply.",
    why:   "Bigger n = bigger key matrices = slower sampling. Done once per session.",
  },
  {
    key:   "encryptionTimeMs",
    short: "Encrypt",
    param: "784 pixels -> ct",
    color: P.seal,
    what:  "Encodes 784 MNIST pixel values into a single BFV/BGV ciphertext.",
    how:   "NTT packs pixels into slots. ct = (pk0*u+e1+delta*m, pk1*u+e2) where delta = floor(q/t) lifts message into high bits.",
    why:   "BFV/BGV batch 4096 slots. NTT is O(n log n) — fast but dominated by random polynomial sampling.",
  },
  {
    key:   "additionTimeMs",
    short: "HE Add",
    param: "ct + ct",
    color: P.openfhe,
    what:  "Adds two ciphertexts, equivalent to adding the underlying plaintexts.",
    how:   "Component-wise polynomial addition mod q: (c0+c0', c1+c1') mod q. No extra noise beyond additive.",
    why:   "Pure polynomial addition is O(n). Cheapest HE op — used for bias addition in FC layers.",
  },
  {
    key:   "multiplicationTimeMs",
    short: "HE Multiply",
    param: "ct x ct + relin",
    color: P.helib,
    what:  "Multiplies two ciphertexts — most expensive HE primitive.",
    how:   "Tensor product of degree-1 ciphertexts -> degree-2, then relinearise via key-switching back to degree-1. Needs NTT + modulus switching.",
    why:   "Key-switching against large gadget matrices. Noise grows quadratically. Limits circuit depth. Used for Conv/FC weights.",
  },
  {
    key:   "decryptionTimeMs",
    short: "Decrypt",
    param: "ct -> 10 logits",
    color: "#ff6080",
    what:  "Recovers the 10 prediction logits from the output ciphertext.",
    how:   "Inner product with secret key: m~ = c0 + c1*s mod q, scale back: m = round(m~*t/q) mod t. Argmax -> digit 0-9.",
    why:   "Single polynomial evaluation. Very fast. Secret key never leaves client in a real deployment.",
  },
];

/* ══════════════════════════════════════════════════════
   PIXEL SVG ICONS  (no emojis — hand-drawn pixel art)
   ══════════════════════════════════════════════════════ */

function IconKeyGen({ size = 18, color = P.gold }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" style={{ imageRendering: "pixelated", display: "block" }}>
      {/* Ring of key */}
      <rect x="2" y="1" width="7" height="2" fill={color}/>
      <rect x="1" y="3" width="2" height="5" fill={color}/>
      <rect x="8" y="3" width="2" height="5" fill={color}/>
      <rect x="2" y="7" width="7" height="2" fill={color}/>
      <rect x="3" y="4" width="5" height="4" fill={P.panelBg}/>
      {/* Shaft */}
      <rect x="9" y="8" width="6" height="2" fill={color}/>
      {/* Teeth */}
      <rect x="11" y="10" width="2" height="2" fill={color}/>
      <rect x="14" y="10" width="2" height="2" fill={color}/>
    </svg>
  );
}

function IconEncrypt({ size = 18, color = P.seal }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" style={{ imageRendering: "pixelated", display: "block" }}>
      {/* Shackle */}
      <rect x="4" y="0" width="8" height="2" fill={color}/>
      <rect x="2" y="2" width="2" height="5" fill={color}/>
      <rect x="12" y="2" width="2" height="5" fill={color}/>
      {/* Body */}
      <rect x="1" y="7" width="14" height="9" fill={color}/>
      <rect x="3" y="9" width="10" height="5" fill={P.panelBg}/>
      {/* Keyhole */}
      <rect x="6" y="10" width="4" height="2" fill={color}/>
      <rect x="7" y="12" width="2" height="2" fill={color}/>
    </svg>
  );
}

function IconAdd({ size = 18, color = P.openfhe }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" style={{ imageRendering: "pixelated", display: "block" }}>
      <rect x="6" y="1" width="4" height="14" fill={color}/>
      <rect x="1" y="6" width="14" height="4" fill={color}/>
    </svg>
  );
}

function IconMultiply({ size = 18, color = P.helib }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" style={{ imageRendering: "pixelated", display: "block" }}>
      {/* Lightning bolt shape */}
      <rect x="9" y="0"  width="4" height="2" fill={color}/>
      <rect x="7" y="2"  width="4" height="2" fill={color}/>
      <rect x="5" y="4"  width="6" height="2" fill={color}/>
      <rect x="4" y="6"  width="8" height="2" fill={color}/>
      <rect x="6" y="8"  width="6" height="2" fill={color}/>
      <rect x="4" y="10" width="6" height="2" fill={color}/>
      <rect x="3" y="12" width="6" height="2" fill={color}/>
      <rect x="2" y="14" width="4" height="2" fill={color}/>
    </svg>
  );
}

function IconDecrypt({ size = 18, color = "#ff6080" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" style={{ imageRendering: "pixelated", display: "block" }}>
      {/* Open shackle — left arm raised */}
      <rect x="4" y="0"  width="8" height="2" fill={color}/>
      <rect x="2" y="2"  width="2" height="7" fill={color}/>
      {/* right arm stops short */}
      <rect x="12" y="2" width="2" height="2" fill={color}/>
      {/* Body */}
      <rect x="1" y="7"  width="14" height="9" fill={color}/>
      <rect x="3" y="9"  width="10" height="5" fill={P.panelBg}/>
      <rect x="6" y="10" width="4" height="2" fill={color}/>
      <rect x="7" y="12" width="2" height="2" fill={color}/>
    </svg>
  );
}

function IconTrophy({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" style={{ imageRendering: "pixelated", display: "block" }}>
      <rect x="2"  y="0"  width="10" height="6" fill={P.gold}/>
      <rect x="0"  y="0"  width="2"  height="4" fill={P.gold}/>
      <rect x="12" y="0"  width="2"  height="4" fill={P.gold}/>
      <rect x="1"  y="1"  width="2"  height="2" fill={P.panelBg} opacity="0.4"/>
      <rect x="5"  y="6"  width="4"  height="4" fill={P.gold}/>
      <rect x="3"  y="10" width="8"  height="2" fill={P.gold}/>
      <rect x="2"  y="12" width="10" height="2" fill={P.goldDim}/>
    </svg>
  );
}

function IconChart({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" style={{ imageRendering: "pixelated", display: "block" }}>
      <rect x="0"  y="8"  width="3" height="6" fill={P.seal}/>
      <rect x="4"  y="4"  width="3" height="10" fill={P.helib}/>
      <rect x="8"  y="1"  width="3" height="13" fill={P.openfhe}/>
      <rect x="11" y="6"  width="3" height="8" fill={P.gold}/>
      <rect x="0"  y="13" width="14" height="1" fill={P.dim}/>
    </svg>
  );
}

function IconOwl({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 16" style={{ imageRendering: "pixelated", display: "block" }}>
      <rect x="2"  y="0"  width="2" height="3" fill={P.goldDim}/>
      <rect x="10" y="0"  width="2" height="3" fill={P.goldDim}/>
      <rect x="2"  y="3"  width="10" height="6" fill={P.gold}/>
      <rect x="1"  y="4"  width="12" height="4" fill={P.gold}/>
      <rect x="3"  y="4"  width="3" height="3" fill="white"/>
      <rect x="8"  y="4"  width="3" height="3" fill="white"/>
      <rect x="4"  y="5"  width="2" height="2" fill={P.panelBg}/>
      <rect x="9"  y="5"  width="2" height="2" fill={P.panelBg}/>
      <rect x="6"  y="7"  width="2" height="2" fill="#f07010"/>
      <rect x="2"  y="9"  width="10" height="5" fill="#e09018"/>
      <rect x="4"  y="10" width="6" height="4" fill="#fce090"/>
      <rect x="0"  y="9"  width="2" height="4" fill="#c07808"/>
      <rect x="12" y="9"  width="2" height="4" fill="#c07808"/>
      <rect x="3"  y="14" width="3" height="2" fill="#f07010"/>
      <rect x="8"  y="14" width="3" height="2" fill="#f07010"/>
    </svg>
  );
}

function IconRefresh({ size = 12, color = P.dim }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" style={{ imageRendering: "pixelated", display: "block" }}>
      <rect x="4" y="0" width="4" height="2" fill={color}/>
      <rect x="8" y="2" width="2" height="2" fill={color}/>
      <rect x="10" y="4" width="2" height="4" fill={color}/>
      <rect x="2" y="8" width="2" height="2" fill={color}/>
      <rect x="4" y="10" width="4" height="2" fill={color}/>
      <rect x="0" y="4" width="2" height="4" fill={color}/>
      <rect x="2" y="2" width="2" height="2" fill={color}/>
      {/* Arrow head on top right */}
      <rect x="8" y="0" width="4" height="2" fill={color}/>
      <rect x="10" y="2" width="2" height="2" fill={color}/>
    </svg>
  );
}

const OP_ICONS = {
  keyGenTimeMs:         (c) => <IconKeyGen color={c}/>,
  encryptionTimeMs:     (c) => <IconEncrypt color={c}/>,
  additionTimeMs:       (c) => <IconAdd color={c}/>,
  multiplicationTimeMs: (c) => <IconMultiply color={c}/>,
  decryptionTimeMs:     (c) => <IconDecrypt color={c}/>,
};

/* ══════════════════════════════════════════════════════
   REUSABLE PIXEL UI PRIMITIVES
   ══════════════════════════════════════════════════════ */

/** Dark wood panel with inset border + optional title bar */
function PixelPanel({ children, style = {}, accent = P.borderHi, title = null, titleIcon = null }) {
  return (
    <div style={{
      background: P.panel,
      border: `3px solid ${P.borderLo}`,
      boxShadow: `inset 0 0 0 1px ${accent}44, 3px 3px 0 ${P.borderLo}`,
      outline: `1px solid ${accent}`,
      outlineOffset: -4,
      position: "relative",
      ...style,
    }}>
      {title && (
        <div style={{
          background: `linear-gradient(90deg, ${accent}22, ${P.panelMid})`,
          borderBottom: `2px solid ${accent}55`,
          padding: "5px 10px 4px",
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}>
          {/* Pixel corner square */}
          <div style={{ width: 6, height: 6, background: accent, flexShrink: 0,
                        boxShadow: `0 0 0 1px ${P.borderLo}` }}/>
          {titleIcon && titleIcon}
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontSize: 7,
            color: accent,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}>{title}</span>
        </div>
      )}
      {children}
    </div>
  );
}

/** Habbo-style 3-D pixel button */
function PixelButton({ children, onClick, color = P.gold, bg = P.panelMid, small = false, disabled = false }) {
  const [pressed, setPressed] = useState(false);
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      onMouseLeave={() => setPressed(false)}
      style={{
        fontFamily: "'Press Start 2P', monospace",
        fontSize: small ? 7 : 8,
        color: disabled ? P.dim : (bg === P.gold ? P.panelBg : color),
        background: bg,
        border: `2px solid ${disabled ? P.dim : color}`,
        borderBottom: pressed ? `2px solid ${color}` : `4px solid ${P.borderLo}`,
        borderRight:  pressed ? `2px solid ${color}` : `4px solid ${P.borderLo}`,
        padding: small ? "5px 10px" : "7px 16px",
        cursor: disabled ? "not-allowed" : "pointer",
        transform: pressed ? "translate(2px,2px)" : "none",
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        imageRendering: "pixelated",
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        transition: "transform 0.05s, border-bottom 0.05s, border-right 0.05s",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {children}
    </button>
  );
}

/** Pixel floor-tile strip divider */
function PixelDivider() {
  return (
    <div style={{ display: "flex", height: 6, overflow: "hidden", flexShrink: 0 }}>
      {Array.from({ length: 60 }).map((_, i) => (
        <div key={i} style={{
          flex: 1,
          background: i % 2 === 0 ? P.floor : P.floorAlt,
          borderRight: `1px solid ${P.borderLo}`,
          borderTop: `1px solid ${P.borderHi}18`,
        }}/>
      ))}
    </div>
  );
}

/** Small inline pixel badge */
function PixelBadge({ children, color = P.gold }) {
  return (
    <span style={{
      fontFamily: "'Press Start 2P', monospace",
      fontSize: 6,
      color,
      background: `${color}18`,
      border: `1px solid ${color}55`,
      padding: "2px 5px",
      letterSpacing: "0.05em",
      display: "inline-block",
      lineHeight: 1.4,
      flexShrink: 0,
    }}>
      {children}
    </span>
  );
}

/** Pixel scanline overlay (purely decorative) */
function Scanlines() {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1,
      backgroundImage: `repeating-linear-gradient(0deg, ${P.scanline} 0px, ${P.scanline} 1px, transparent 1px, transparent 3px)`,
    }}/>
  );
}

/* ══════════════════════════════════════════════════════
   PIXEL RADAR CHART
   ══════════════════════════════════════════════════════ */
function PixelRadar({ results }) {
  const W = 200, H = 190, CX = 100, CY = 92, R = 64;
  const n = OPS.length;
  const angles  = OPS.map((_, i) => (i / n) * 2 * Math.PI - Math.PI / 2);
  const pt      = (a, r) => [CX + r * Math.cos(a), CY + r * Math.sin(a)];
  const opMaxes = OPS.map(op => Math.max(...results.map(r => r[op.key] || 0), 1));
  const score   = (lib, i) => 1 - (lib[OPS[i].key] || 0) / opMaxes[i];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {/* Concentric pixel rings */}
      {[0.25, 0.5, 0.75, 1.0].map((lvl) => (
        <polygon key={lvl}
          points={angles.map(a => pt(a, R * lvl).join(",")).join(" ")}
          fill="none" stroke={P.border} strokeWidth={1} strokeDasharray="3 2"/>
      ))}
      {/* Spokes */}
      {angles.map((a, i) => {
        const [x, y] = pt(a, R);
        return <line key={i} x1={CX} y1={CY} x2={x} y2={y} stroke={P.border} strokeWidth={1}/>;
      })}
      {/* Library polygons */}
      {results.map(lib => {
        const lc  = LIB[lib.library] || LIB.SEAL;
        const pts = angles.map((a, i) => pt(a, R * Math.max(0.06, score(lib, i))));
        return (
          <g key={lib.library}>
            <polygon points={pts.map(p => p.join(",")).join(" ")}
              fill={lc.color} fillOpacity={0.13} stroke={lc.color} strokeWidth={2} strokeLinejoin="miter"/>
            {pts.map(([x, y], i) => (
              <rect key={i} x={x-3} y={y-3} width={6} height={6}
                fill={lc.color} stroke={P.borderLo} strokeWidth={1}/>
            ))}
          </g>
        );
      })}
      {/* Labels */}
      {angles.map((a, i) => {
        const [x, y] = pt(a, R + 15);
        return (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fontSize={6} fontFamily="'Press Start 2P', monospace" fill={P.goldDim}>
            {OPS[i].short}
          </text>
        );
      })}
      <text x={CX} y={CY} textAnchor="middle" dominantBaseline="middle"
        fontSize={6} fontFamily="'Press Start 2P', monospace" fill={P.dim}>SPEED</text>
    </svg>
  );
}

/* ══════════════════════════════════════════════════════
   FULL-WIDTH TIME BARS  (readable, big)
   ══════════════════════════════════════════════════════ */
function BigTimeBars({ results }) {
  const sorted   = [...results].sort((a, b) => a.totalTimeMs - b.totalTimeMs);
  const maxTotal = Math.max(...results.map(r => r.totalTimeMs || 0), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {sorted.map((lib, rank) => {
        const lc  = LIB[lib.library] || LIB.SEAL;
        const pct = Math.max((lib.totalTimeMs / maxTotal) * 100, 3);
        const segs = Math.max(1, Math.floor(pct / 5));
        return (
          <div key={lib.library}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 5 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 10, height: 10, background: lc.color, flexShrink: 0, boxShadow: `0 0 0 2px ${P.borderLo}` }}/>
                <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: lc.color }}>{lib.library}</span>
                <PixelBadge color={lc.color}>{lc.scheme}</PixelBadge>
                {rank === 0 && <IconTrophy size={13}/>}
              </div>
              <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 20, fontWeight: 700, color: lc.color, letterSpacing: "-0.02em" }}>
                {lib.totalTimeMs.toFixed(1)}<span style={{ fontSize: 12, fontWeight: 400, marginLeft: 3, opacity: 0.7 }}>ms</span>
              </span>
            </div>
            <div style={{ height: 20, background: P.panelBg, border: `2px solid ${P.border}`, boxShadow: `inset 2px 2px 0 ${P.borderLo}`, position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${pct}%`, background: lc.color, boxShadow: `inset 0 5px 0 ${lc.color}80, inset 0 -3px 0 ${lc.dark}` }}>
                {Array.from({ length: segs }).map((_, i) => (
                  <div key={i} style={{ position: "absolute", left: `${((i + 1) / segs) * 100}%`, top: 0, bottom: 0, width: 2, background: `${P.borderLo}55` }}/>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   COMPACT TIME BARS  (smaller, single-screen friendly)
   ══════════════════════════════════════════════════════ */
function CompactTimeBars({ results }) {
  const sorted   = [...results].sort((a, b) => a.totalTimeMs - b.totalTimeMs);
  const maxTotal = Math.max(...results.map(r => r.totalTimeMs || 0), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {sorted.map((lib, rank) => {
        const lc  = LIB[lib.library] || LIB.SEAL;
        const pct = Math.max((lib.totalTimeMs / maxTotal) * 100, 3);
        const segs = Math.max(1, Math.floor(pct / 5));
        return (
          <div key={lib.library}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 5 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 10, height: 10, background: lc.color, flexShrink: 0, boxShadow: `0 0 0 2px ${P.borderLo}` }}/>
                <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: lc.color }}>{lib.library}</span>
                <PixelBadge color={lc.color}>{lc.scheme}</PixelBadge>
                {rank === 0 && <IconTrophy size={13}/>}
              </div>
              <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 20, fontWeight: 700, color: lc.color, letterSpacing: "-0.02em" }}>
                {lib.totalTimeMs.toFixed(1)}<span style={{ fontSize: 12, fontWeight: 400, marginLeft: 3, opacity: 0.7 }}>ms</span>
              </span>
            </div>
            <div style={{ height: 20, background: P.panelBg, border: `2px solid ${P.border}`, boxShadow: `inset 2px 2px 0 ${P.borderLo}`, position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${pct}%`, background: lc.color, boxShadow: `inset 0 5px 0 ${lc.color}80, inset 0 -3px 0 ${lc.dark}` }}>
                {Array.from({ length: segs }).map((_, i) => (
                  <div key={i} style={{ position: "absolute", left: `${((i + 1) / segs) * 100}%`, top: 0, bottom: 0, width: 2, background: `${P.borderLo}55` }}/>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   FULL-WIDTH OP ROWS  (readable text, big bars)
   ══════════════════════════════════════════════════════ */
function BigOpRow({ op, results, isActive, onToggle }) {
  const maxVal   = Math.max(...results.map(r => r[op.key] || 0), 1);
  const opSorted = [...results].sort((a, b) => (a[op.key] || 0) - (b[op.key] || 0));

  return (
    <div style={{ background: isActive ? `${op.color}0c` : "transparent", border: `2px solid ${isActive ? op.color + "55" : "transparent"}`, marginBottom: 6 }}>
      {/* Header */}
      <div onClick={onToggle} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", cursor: "pointer", userSelect: "none" }}>
        <div style={{ width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          {OP_ICONS[op.key]?.(op.color)}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 11, color: op.color }}>{op.short}</span>
            <PixelBadge color={op.color}>{op.param}</PixelBadge>
          </div>
        </div>
        <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.dim, flexShrink: 0 }}>
          {isActive ? "[HIDE]" : "[INFO]"}
        </span>
      </div>

      {/* Bars */}
      <div style={{ padding: "0 16px 14px" }}>
        {results.map(lib => {
          const lc  = LIB[lib.library] || LIB.SEAL;
          const val = lib[op.key] || 0;
          const pct = Math.max(2, (val / maxVal) * 100);
          return (
            <div key={lib.library} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: lc.color, width: 72, textAlign: "right", flexShrink: 0 }}>{lib.library}</span>
              <div style={{ flex: 1, height: 18, background: P.panelBg, border: `2px solid ${P.border}`, boxShadow: `inset 2px 2px 0 ${P.borderLo}`, overflow: "hidden", position: "relative" }}>
                <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${pct}%`, background: lc.color, boxShadow: `inset 0 4px 0 ${lc.color}80` }}/>
              </div>
              <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 16, fontWeight: 700, color: lc.color, width: 80, textAlign: "right", flexShrink: 0 }}>{val.toFixed(1)} ms</span>
            </div>
          );
        })}
      </div>

      {/* Expanded */}
      {isActive && (
        <div className="px-slide" style={{ margin: "0 16px 14px", background: P.panelBg, border: `2px solid ${op.color}44`, boxShadow: `2px 2px 0 ${P.borderLo}`, padding: "16px 18px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
            {[{ label: "WHAT", body: op.what }, { label: "HOW", body: op.how }, { label: "WHY", body: op.why }].map(card => (
              <div key={card.label} style={{ background: P.panel, border: `2px solid ${op.color}33`, padding: "12px 14px" }}>
                <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: op.color, marginBottom: 8, letterSpacing: "0.08em" }}>{card.label}</div>
                <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 13, color: P.cream, lineHeight: 1.7, opacity: 0.9 }}>{card.body}</div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            {opSorted.map((lib, idx) => {
              const lc  = LIB[lib.library] || LIB.SEAL;
              const val = lib[op.key] || 0;
              return (
                <div key={lib.library} style={{ flex: 1, background: P.panel, border: `2px solid ${idx === 0 ? lc.color : lc.color + "44"}`, boxShadow: idx === 0 ? `4px 4px 0 ${P.borderLo}` : `2px 2px 0 ${P.borderLo}`, padding: "12px 14px", textAlign: "center" }}>
                  {idx === 0 && (
                    <div style={{ display: "flex", justifyContent: "center", marginBottom: 6, gap: 5 }}>
                      <IconTrophy size={14}/>
                      <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: P.gold }}>FASTEST</span>
                    </div>
                  )}
                  <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: lc.color, marginBottom: 6 }}>{lib.library}</div>
                  <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 32, fontWeight: 700, color: lc.color, lineHeight: 1, letterSpacing: "-0.02em" }}>{val.toFixed(1)}</div>
                  <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 13, color: P.dim, marginTop: 3 }}>ms</div>
                  <div style={{ height: 5, background: P.panelBg, border: `1px solid ${P.border}`, marginTop: 10, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(val / maxVal) * 100}%`, background: lc.color }}/>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      <PixelDivider/>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   BIG LIBRARY SUMMARY CARD
   ══════════════════════════════════════════════════════ */
function BigLibCard({ lib, rank }) {
  const lc = LIB[lib.library] || LIB.SEAL;
  return (
    <div style={{ background: P.panel, border: `3px solid ${rank === 0 ? lc.color : P.border}`, boxShadow: rank === 0 ? `5px 5px 0 ${P.borderLo}` : `3px 3px 0 ${P.borderLo}`, position: "relative", overflow: "hidden" }}>
      <Scanlines/>
      {/* Header */}
      <div style={{ background: `linear-gradient(90deg, ${lc.color}22, ${P.panelMid})`, borderBottom: `2px solid ${lc.color}55`, padding: "10px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "relative", zIndex: 2 }}>
        <div>
          <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 12, color: lc.color, marginBottom: 4 }}>{lib.library}</div>
          <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 12, color: P.dim }}>{lc.madeBy} · {lc.scheme}</div>
        </div>
        <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: rank === 0 ? P.gold : P.dim, background: P.panelBg, border: `2px solid ${rank === 0 ? P.gold : P.border}`, padding: "4px 9px", display: "flex", alignItems: "center", gap: 5 }}>
          {rank === 0 && <IconTrophy size={12}/>}#{rank + 1}
        </div>
      </div>
      {/* Big total */}
      <div style={{ textAlign: "center", padding: "20px 14px 14px", borderBottom: `1px solid ${P.border}`, position: "relative", zIndex: 2 }}>
        <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 48, fontWeight: 800, color: lc.color, textShadow: `3px 3px 0 ${P.borderLo}`, lineHeight: 1, letterSpacing: "-0.03em" }}>{lib.totalTimeMs.toFixed(0)}</div>
        <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 16, color: P.dim, marginTop: 4, fontWeight: 500 }}>ms total</div>
      </div>
      {/* Per-op */}
      <div style={{ padding: "10px 14px", position: "relative", zIndex: 2 }}>
        {OPS.map(op => (
          <div key={op.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "5px 0", borderBottom: `1px solid ${P.border}44` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <div style={{ opacity: 0.7 }}>{OP_ICONS[op.key]?.(op.color)}</div>
              <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 13, color: P.cream, opacity: 0.7 }}>{op.short}</span>
            </div>
            <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 15, fontWeight: 700, color: lc.color }}>{(lib[op.key] || 0).toFixed(1)} ms</span>
          </div>
        ))}
      </div>
      <div style={{ padding: "8px 14px 10px", position: "relative", zIndex: 2 }}>
        <PixelBadge color={lib.success ? P.openfhe : "#ff6080"}>{lib.success ? "ALL OK" : "FAILED"}</PixelBadge>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   MAIN EXPORT
   ══════════════════════════════════════════════════════ */
export default function LibraryComparison({ data, loading, error, onRun }) {
  const [activeOp, setActiveOp] = useState(null);
  const [showOwl,  setShowOwl]  = useState(false);
  const [owlDone,  setOwlDone]  = useState(false);

  const handleRun = () => { setOwlDone(false); setShowOwl(false); onRun(null); };

  /* Shared full-width wrapper */
  const FullBleed = ({ children, bg = P.panelBg }) => (
    <div style={{ width: "100%", background: bg, overflow: "hidden" }}>
      {children}
    </div>
  );

  /* ── Idle / empty state ── */
  if (!data && !loading && !error) {
    return (
      <FullBleed>
        <style>{PIXEL_CSS}</style>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 48px" }}>
          {/* Banner */}
          <PixelPanel accent={P.gold} titleIcon={<IconChart size={14}/>} title="MNIST HE PRIMITIVE BENCHMARKS" style={{ marginBottom: 20 }}>
            <div style={{ padding: "10px 16px 12px", fontFamily: "system-ui, sans-serif", fontSize: 14, color: P.cream, opacity: 0.8, lineHeight: 1.6, display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
              <span>3 libraries · 5 operations · 10× averaged · n = 4096 · 128-bit security</span>
              <PixelBadge color={P.seal}>SEAL = BFV</PixelBadge>
              <PixelBadge color={P.helib}>HELib = BGV</PixelBadge>
              <PixelBadge color={P.openfhe}>OpenFHE = BFV</PixelBadge>
            </div>
          </PixelPanel>

          {/* Library tiles */}
          <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 24 }}>
            {Object.entries(LIB).map(([name, lc]) => (
              <div key={name} style={{ background: P.panel, border: `3px solid ${lc.color}`, boxShadow: `4px 4px 0 ${P.borderLo}`, padding: "18px 32px", textAlign: "center", minWidth: 140, position: "relative", overflow: "hidden" }}>
                <Scanlines/>
                <div style={{ position: "relative", zIndex: 2 }}>
                  <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 12, color: lc.color, marginBottom: 6 }}>{name}</div>
                  <PixelBadge color={lc.color}>{lc.scheme}</PixelBadge>
                  <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 13, color: P.cream, opacity: 0.5, marginTop: 6 }}>{lc.madeBy}</div>
                </div>
              </div>
            ))}
          </div>

          <PixelDivider/>
          <div style={{ display: "flex", justifyContent: "center", paddingTop: 20 }}>
            <PixelButton onClick={handleRun} color={P.gold} bg={P.panelMid}>
              <IconChart size={16}/>
              RUN LIBRARY COMPARISON
            </PixelButton>
          </div>
        </div>
      </FullBleed>
    );
  }

  /* ── Loading ── */
  if (loading) {
    return (
      <FullBleed>
        <style>{PIXEL_CSS}</style>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 48px" }}>
          <PixelPanel accent={P.gold} titleIcon={<IconChart size={14}/>} title="MNIST HE PRIMITIVE BENCHMARKS" style={{ marginBottom: 20 }}>
            <div style={{ padding: "10px 16px 12px", fontFamily: "system-ui, sans-serif", fontSize: 14, color: P.cream, opacity: 0.7 }}>
              3 libraries · 5 operations · 10× averaged · n = 4096 · 128-bit security
            </div>
          </PixelPanel>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px 0", gap: 20 }}>
            <div className="px-spin" style={{ width: 48, height: 48, border: `5px solid ${P.border}`, borderTop: `5px solid ${P.gold}`, boxShadow: `4px 4px 0 ${P.borderLo}` }}/>
            <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 13, color: P.gold, letterSpacing: "0.1em" }}>RUNNING BENCHMARKS</div>
            <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 15, color: P.cream, opacity: 0.6, textAlign: "center", lineHeight: 2 }}>
              SEAL <span className="px-blink" style={{ color: P.gold, fontFamily: "'Press Start 2P', monospace" }}>...</span>
              {"  "}HELib{"  "}OpenFHE
              <br/>Estimated 30 – 90 seconds total
            </div>
            <div style={{ width: 320, height: 18, background: P.panel, border: `2px solid ${P.border}`, boxShadow: `inset 2px 2px 0 ${P.borderLo}`, overflow: "hidden", position: "relative" }}>
              <div className="px-march" style={{ position: "absolute", top: 0, bottom: 0, left: 0, width: "40%", backgroundImage: `repeating-linear-gradient(90deg, ${P.gold} 0px, ${P.gold} 12px, ${P.goldDim} 12px, ${P.goldDim} 14px)`, backgroundSize: "28px 100%" }}/>
            </div>
          </div>
        </div>
      </FullBleed>
    );
  }

  /* ── Error ── */
  if (error) {
    return (
      <FullBleed>
        <style>{PIXEL_CSS}</style>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 48px" }}>
          <PixelPanel accent="#ff6080" title="BENCHMARK FAILED">
            <div style={{ padding: "20px 16px", textAlign: "center" }}>
              <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 15, color: "#ff6080", marginBottom: 16, lineHeight: 1.7 }}>{error}</div>
              <PixelButton onClick={handleRun} color="#ff6080" bg={P.panelMid}>RETRY</PixelButton>
            </div>
          </PixelPanel>
        </div>
      </FullBleed>
    );
  }

  /* ── Results ── */
  const results = data?.results || [];
  if (!results.length) return (
    <FullBleed>
      <div style={{ padding: 24, textAlign: "center", fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: P.dim }}>NO RESULTS RETURNED</div>
    </FullBleed>
  );

  const sortedResults = [...results].sort((a, b) => a.totalTimeMs - b.totalTimeMs);
  const fastest       = sortedResults[0];
  const slowest       = sortedResults[sortedResults.length - 1];
  const speedup       = (slowest.totalTimeMs / fastest.totalTimeMs).toFixed(1);
  const owlVisible    = showOwl || (data && !owlDone);

  return (
    <>
      <style>{PIXEL_CSS}</style>

      {/* ── Owl guide: fixed fullscreen overlay ── */}
      {owlVisible && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9999,
          background: "rgba(10,5,2,0.92)",
          display: "flex", alignItems: "center", justifyContent: "center",
          padding: 0,
        }}>
          <div style={{ width: "100%", height: "100%", maxWidth: 1100, display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
            <OwlGuide onDone={() => { setShowOwl(false); setOwlDone(true); }}/>
          </div>
        </div>
      )}

      {/* ── Results: full-bleed panel ── */}
      <FullBleed>
        <PixelDivider/>

        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "14px 28px 18px" }}>

          {/* ── Header bar ── */}
          <PixelPanel accent={P.gold} titleIcon={<IconChart size={14}/>} title="MNIST HE PRIMITIVE BENCHMARKS" style={{ marginBottom: 12 }}>
            <div style={{ padding: "6px 14px 7px", fontFamily: "system-ui, sans-serif", fontSize: 13, color: P.cream, opacity: 0.8, lineHeight: 1.5, display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
              <span>3 libraries · 5 operations · 10× averaged · n = 4096 · 128-bit security</span>
              <PixelBadge color={P.seal}>SEAL = BFV</PixelBadge>
              <PixelBadge color={P.helib}>HELib = BGV</PixelBadge>
              <PixelBadge color={P.openfhe}>OpenFHE = BFV</PixelBadge>
            </div>
          </PixelPanel>

          {/* ── Two-column: bars (left) + radar + buttons (right) ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 210px", gap: 12, marginBottom: 12, alignItems: "start" }}>

            <PixelPanel accent={P.gold} title="TOTAL PIPELINE TIME">
              <div style={{ padding: "12px 18px 10px" }}>
                <CompactTimeBars results={results}/>
                {results.length >= 2 && (
                  <div style={{ marginTop: 10, background: `${LIB[fastest.library]?.color || P.gold}12`, border: `1px solid ${LIB[fastest.library]?.color || P.gold}44`, padding: "6px 12px", display: "flex", alignItems: "center", gap: 8 }}>
                    <IconTrophy size={12}/>
                    <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 13, fontWeight: 600, color: LIB[fastest.library]?.color || P.gold }}>
                      <strong>{fastest.library}</strong> is <strong>{speedup}×</strong> faster than <strong>{slowest.library}</strong>
                    </span>
                  </div>
                )}
              </div>
            </PixelPanel>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <PixelPanel accent={P.goldDim} title="SPEED RADAR">
                <div style={{ padding: "4px 2px 0" }}>
                  <PixelRadar results={results}/>
                  <div style={{ padding: "2px 10px 7px", display: "flex", flexDirection: "column", gap: 4 }}>
                    {results.map(r => {
                      const lc = LIB[r.library] || LIB.SEAL;
                      return (
                        <div key={r.library} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 8, height: 8, background: lc.color, flexShrink: 0 }}/>
                          <span style={{ fontFamily: "system-ui, sans-serif", fontSize: 12, color: lc.color, fontWeight: 600 }}>{r.library}</span>
                        </div>
                      );
                    })}
                    <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 11, color: P.dim, marginTop: 2 }}>Larger = faster</div>
                  </div>
                </div>
              </PixelPanel>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <PixelButton onClick={() => { setShowOwl(true); setOwlDone(false); }} color={P.gold} bg={P.panelMid} small>
                  <IconOwl size={13}/>
                  GUIDE
                </PixelButton>
                <PixelButton onClick={handleRun} color={P.dim} bg={P.panelMid} small>
                  <IconRefresh size={12}/>
                  RUN AGAIN
                </PixelButton>
              </div>
            </div>
          </div>

          {/* ── Operation breakdown ── */}
          <PixelPanel accent={P.borderHi} title="OPERATION BREAKDOWN — CLICK ANY ROW FOR DETAILS">
            <div style={{ display: "flex", gap: 16, padding: "5px 14px", borderBottom: `1px solid ${P.border}`, alignItems: "center" }}>
              {results.map(r => {
                const lc = LIB[r.library] || LIB.SEAL;
                return (
                  <div key={r.library} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 16, height: 8, background: lc.color, boxShadow: `0 0 0 1px ${P.borderLo}` }}/>
                    <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: lc.color }}>{r.library}</span>
                  </div>
                );
              })}
            </div>
            <div style={{ padding: "4px 0" }}>
              {OPS.map(op => (
                <BigOpRow key={op.key} op={op} results={results} isActive={activeOp === op.key} onToggle={() => setActiveOp(activeOp === op.key ? null : op.key)}/>
              ))}
            </div>
            <div style={{ fontFamily: "system-ui, sans-serif", fontSize: 11, color: P.dim, fontStyle: "italic", padding: "0 14px 10px", lineHeight: 1.5 }}>
              Fig. 1 — HE primitive latencies at n=4096, t=65537, 128-bit security, averaged over 10 repetitions.
            </div>
          </PixelPanel>

        </div>
        <PixelDivider/>
      </FullBleed>
    </>
  );
}

