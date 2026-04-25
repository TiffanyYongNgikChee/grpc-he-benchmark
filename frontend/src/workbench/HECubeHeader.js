import { useEffect, useState, useRef } from "react";

// ─── CSS injected once ────────────────────────────────────────────────────────
const CUBE_STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

  /* Cube float + spin */
  @keyframes cubeFloat  { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
  @keyframes cubeSpin   { from{transform:rotateY(0deg) rotateX(15deg)} to{transform:rotateY(360deg) rotateX(15deg)} }
  @keyframes cubeSpinR  { from{transform:rotateY(360deg) rotateX(15deg)} to{transform:rotateY(0deg) rotateX(15deg)} }

  /* Compute box pulse */
  @keyframes boxPulse   { 0%,100%{box-shadow:0 0 18px 4px #4285f455} 50%{box-shadow:0 0 36px 10px #9333ea88} }
  @keyframes boxGlow    { 0%,100%{opacity:0.5} 50%{opacity:1} }

  /* Flying cube through box */
  @keyframes flyIn  {
    0%   { transform: translateX(0)   scale(1)   opacity(1); }
    40%  { transform: translateX(82px) scale(0.7) opacity(0.6); }
    50%  { transform: translateX(92px) scale(0.5) opacity(0); }
    51%  { transform: translateX(92px) scale(0.5) opacity(0); }
    60%  { transform: translateX(104px) scale(0.5) opacity(0); }
    70%  { transform: translateX(116px) scale(0.7) opacity(0.6); }
    100% { transform: translateX(196px) scale(1)  opacity(1); }
  }

  /* Number scramble shimmer inside cube */
  @keyframes numShimmer { 0%{opacity:0.15} 50%{opacity:0.55} 100%{opacity:0.15} }

  /* Lock shake */
  @keyframes lockShake  {
    0%,100%{transform:rotate(0deg)}
    20%{transform:rotate(-6deg)}
    40%{transform:rotate(6deg)}
    60%{transform:rotate(-4deg)}
    80%{transform:rotate(4deg)}
  }

  /* Particle burst */
  @keyframes particleFly {
    0%   { transform:translate(0,0) scale(1); opacity:1; }
    100% { transform:translate(var(--px),var(--py)) scale(0); opacity:0; }
  }

  /* Scan line on compute box */
  @keyframes scanLine { from{top:0%} to{top:100%} }

  .cube-spin  { animation: cubeSpin  6s linear infinite; }
  .cube-spin-r{ animation: cubeSpinR 8s linear infinite; }
  .cube-float { animation: cubeFloat 2.8s ease-in-out infinite; }
  .num-shimmer{ animation: numShimmer 1.4s ease-in-out infinite; }
  .lock-shake { animation: lockShake 2.5s ease-in-out infinite; }
`;

// ─── One CSS 3D cube face ─────────────────────────────────────────────────────
function CubeFace({ transform, bg, border, children, style = {} }) {
  return (
    <div style={{
      position: "absolute",
      width: "100%",
      height: "100%",
      background: bg,
      border: `2px solid ${border}`,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      backfaceVisibility: "hidden",
      transform,
      boxSizing: "border-box",
      imageRendering: "pixelated",
      ...style,
    }}>
      {children}
    </div>
  );
}

// ─── Scrambled number texture (hidden data inside) ────────────────────────────
function DataNoise({ color }) {
  const nums = useRef(
    Array.from({ length: 12 }, () => Math.floor(Math.random() * 10))
  );
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(4,1fr)",
      gap: 1,
      padding: 3,
      width: "100%",
      height: "100%",
      overflow: "hidden",
    }}>
      {nums.current.map((n, i) => (
        <span key={i} className="num-shimmer" style={{
          fontFamily: "monospace",
          fontSize: 8,
          color,
          opacity: 0.2 + (i % 3) * 0.15,
          animationDelay: `${i * 0.12}s`,
          userSelect: "none",
        }}>{n}</span>
      ))}
    </div>
  );
}

// ─── Lock icon SVG (pixel-style) ─────────────────────────────────────────────
function PixelLock({ color = "#f0c030", size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16"
      style={{ imageRendering: "pixelated", display: "block", flexShrink: 0 }}>
      {/* Shackle */}
      <rect x="4" y="1" width="8" height="2" fill={color} />
      <rect x="3" y="3" width="2" height="4" fill={color} />
      <rect x="11" y="3" width="2" height="4" fill={color} />
      {/* Body */}
      <rect x="2" y="7" width="12" height="8" fill={color} />
      {/* Keyhole */}
      <rect x="6" y="9" width="4" height="4" fill="#0a0502" />
      <rect x="7" y="10" width="2" height="3" fill="#0a0502" />
    </svg>
  );
}

// ─── One spinning cube ────────────────────────────────────────────────────────
function EncryptedCube({ size = 54, colors, spin = "spin", floatDelay = 0, label, locked = true }) {
  const s = size;
  const half = s / 2;

  const [face, side, top] = colors;

  return (
    <div style={{
      width: s, height: s,
      position: "relative",
      perspective: 400,
      animationDelay: `${floatDelay}s`,
    }}>
      <div className={`cube-float ${spin === "spin" ? "cube-spin" : "cube-spin-r"}`}
        style={{
          width: s, height: s,
          position: "relative",
          transformStyle: "preserve-3d",
          animationDelay: `${floatDelay}s`,
        }}>
        {/* Front */}
        <CubeFace transform={`translateZ(${half}px)`} bg={face} border={top}>
          {locked
            ? <div className="lock-shake"><PixelLock color={top} size={s * 0.32} /></div>
            : <DataNoise color={top} />}
        </CubeFace>
        {/* Back */}
        <CubeFace transform={`rotateY(180deg) translateZ(${half}px)`} bg={face} border={top}>
          <DataNoise color={top} />
        </CubeFace>
        {/* Left */}
        <CubeFace transform={`rotateY(-90deg) translateZ(${half}px)`} bg={side} border={top}>
          <DataNoise color={top} />
        </CubeFace>
        {/* Right */}
        <CubeFace transform={`rotateY(90deg) translateZ(${half}px)`} bg={side} border={top}>
          <DataNoise color={top} />
        </CubeFace>
        {/* Top */}
        <CubeFace transform={`rotateX(90deg) translateZ(${half}px)`} bg={top} border={top}
          style={{ opacity: 0.6 }} />
        {/* Bottom */}
        <CubeFace transform={`rotateX(-90deg) translateZ(${half}px)`} bg={side} border={top}
          style={{ opacity: 0.3 }} />
      </div>
      {label && (
        <div style={{
          position: "absolute",
          bottom: -20,
          left: "50%",
          transform: "translateX(-50%)",
          fontFamily: "'Press Start 2P',monospace",
          fontSize: "clamp(7px, 1vw, 9px)",
          color: top,
          whiteSpace: "nowrap",
        }}>{label}</div>
      )}
    </div>
  );
}

// ─── Compute box (middle) ─────────────────────────────────────────────────────
function ComputeBox({ active }) {
  return (
    <div style={{
      width: 72,
      height: 72,
      background: "linear-gradient(135deg,#0d1b2a,#1a0f04)",
      border: "3px solid #f0c030",
      boxShadow: active
        ? "0 0 32px 8px #9333ea99, inset 0 0 20px #4285f444"
        : "0 0 18px 4px #4285f433",
      borderRadius: 6,
      position: "relative",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 4,
      flexShrink: 0,
      transition: "box-shadow 0.4s",
      overflow: "hidden",
      imageRendering: "pixelated",
    }}>
      {/* Scan line */}
      <div className="scan-line" style={{
        position: "absolute",
        left: 0, right: 0,
        height: 2,
        background: "linear-gradient(90deg,transparent,#f0c03088,transparent)",
        animation: "scanLine 1.8s linear infinite",
        pointerEvents: "none",
      }} />
      {/* CPU grid icon (pixel) */}
      <svg width="32" height="32" viewBox="0 0 16 16"
        style={{ imageRendering: "pixelated" }}>
        <rect x="4" y="4" width="8" height="8" fill="none" stroke="#f0c030" strokeWidth="1.5" />
        <rect x="6" y="6" width="4" height="4" fill="#f0c03044" stroke="#f0c030" strokeWidth="0.5" />
        <rect x="2" y="6" width="2" height="1" fill="#4285f4" />
        <rect x="12" y="6" width="2" height="1" fill="#4285f4" />
        <rect x="2" y="9" width="2" height="1" fill="#9333ea" />
        <rect x="12" y="9" width="2" height="1" fill="#9333ea" />
        <rect x="6" y="2" width="1" height="2" fill="#10b981" />
        <rect x="9" y="2" width="1" height="2" fill="#10b981" />
        <rect x="6" y="12" width="1" height="2" fill="#f0c030" />
        <rect x="9" y="12" width="1" height="2" fill="#f0c030" />
      </svg>
      <div style={{
        fontFamily: "'Press Start 2P',monospace",
        fontSize: 5,
        color: "#f0c030",
        textAlign: "center",
        lineHeight: 1.6,
      }}>
        HE<br/>COMPUTE
      </div>
    </div>
  );
}

// ─── Arrow (pixel style) ──────────────────────────────────────────────────────
function PixelArrow({ color = "#f0c030", flipped = false }) {
  return (
    <svg width="28" height="14" viewBox="0 0 28 14"
      style={{ imageRendering: "pixelated", transform: flipped ? "scaleX(-1)" : "none", flexShrink: 0 }}>
      <rect x="0"  y="6" width="20" height="2" fill={color} />
      <rect x="16" y="3" width="2"  height="8" fill={color} />
      <rect x="18" y="4" width="2"  height="6" fill={color} />
      <rect x="20" y="5" width="2"  height="4" fill={color} />
      <rect x="22" y="6" width="2"  height="2" fill={color} />
    </svg>
  );
}

// ─── Particle burst (on "compute done") ──────────────────────────────────────
function Particles({ active }) {
  const parts = [
    { px: "-40px", py: "-30px", c: "#4285f4" },
    { px: "40px",  py: "-28px", c: "#9333ea" },
    { px: "-36px", py: "32px",  c: "#10b981" },
    { px: "38px",  py: "30px",  c: "#f0c030" },
    { px: "-8px",  py: "-42px", c: "#f0c030" },
    { px: "8px",   py: "42px",  c: "#4285f4" },
  ];
  if (!active) return null;
  return (
    <>
      {parts.map((p, i) => (
        <div key={i} style={{
          position: "absolute",
          width: 6, height: 6,
          background: p.c,
          top: "50%", left: "50%",
          "--px": p.px, "--py": p.py,
          animation: `particleFly 0.7s ease-out ${i * 0.05}s both`,
          imageRendering: "pixelated",
          pointerEvents: "none",
        }} />
      ))}
    </>
  );
}

// ─── Main HECubeHeader ────────────────────────────────────────────────────────
export default function HECubeHeader() {
  const [phase, setPhase] = useState(0);
  // phase: 0=idle cubes floating, 1=cubes entering box, 2=computing, 3=output cubes emerging
  const [burst, setBurst] = useState(false);

  useEffect(() => {
    // Loop the animation: idle 2s → enter 1.4s → compute 1.8s → emerge 1.4s → idle …
    const cycle = () => {
      setPhase(0);
      setBurst(false);
      const t1 = setTimeout(() => setPhase(1), 2000);
      const t2 = setTimeout(() => setPhase(2), 3400);
      const t3 = setTimeout(() => { setBurst(true); }, 5000);
      const t4 = setTimeout(() => setPhase(3), 5200);
      const t5 = setTimeout(() => setPhase(0), 7000);
      return [t1, t2, t3, t4, t5];
    };
    let timers = cycle();
    const loopId = setInterval(() => {
      timers.forEach(clearTimeout);
      timers = cycle();
    }, 7400);
    return () => { timers.forEach(clearTimeout); clearInterval(loopId); };
  }, []);

  // Input cubes (SEAL, HELib, OpenFHE colours)
  const inputCubes = [
    { colors: ["#0d2040", "#0a1830", "#4285f4"], spin: "spin",   floatDelay: 0,   label: "SEAL" },
    { colors: ["#1a0830", "#120620", "#9333ea"], spin: "spin-r", floatDelay: 0.4, label: "HELib" },
    { colors: ["#042018", "#031410", "#10b981"], spin: "spin",   floatDelay: 0.8, label: "OpenFHE" },
  ];

  // Caption for each phase
  const captions = [
    "Encrypted data cubes — locked & floating",
    "Cubes entering the HE compute engine...",
    "Computing on encrypted data... 🔐",
    "Results emerge — still encrypted! ✨",
  ];

  return (
    <div style={{
      background: "linear-gradient(135deg,#0a0f1e,#1a0f04,#0a1a0e)",
      borderRadius: 12,
      border: "2px solid #f0c03044",
      marginBottom: 20,
      overflow: "hidden",
      position: "relative",
    }}>
      <style>{CUBE_STYLE}</style>

      {/* Pixel grid background */}
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "linear-gradient(#f0c03008 1px,transparent 1px),linear-gradient(90deg,#f0c03008 1px,transparent 1px)",
        backgroundSize: "16px 16px",
        pointerEvents: "none",
      }} />

      {/* Top title bar */}
      <div style={{
        padding: "10px 16px 6px",
        borderBottom: "1px solid #f0c03022",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}>
        <span style={{ fontFamily: "'Press Start 2P',monospace", fontSize: 8, color: "#f0c030" }}>
          ▸ HE BENCHMARK
        </span>
        <span style={{ fontFamily: "'Press Start 2P',monospace", fontSize: 6, color: "#4285f4" }}>
          SEAL
        </span>
        <span style={{ fontFamily: "'Press Start 2P',monospace", fontSize: 6, color: "#9333ea" }}>
          HELib
        </span>
        <span style={{ fontFamily: "'Press Start 2P',monospace", fontSize: 6, color: "#10b981" }}>
          OpenFHE
        </span>
        <div style={{ flex: 1 }} />
        <span style={{
          fontFamily: "system-ui,sans-serif", fontSize: 11,
          color: "#94a3b8",
        }}>
          n = 4096 · 128-bit · 5 ops × 10 reps
        </span>
      </div>

      {/* Animation stage */}
      <div className="he-cube-stage">

        {/* Left: input cubes */}
        <div style={{
          display: "flex",
          gap: "clamp(6px,1.5vw,14px)",
          alignItems: "center",
          flexWrap: "wrap",
          justifyContent: "center",
          opacity: phase === 3 ? 0.3 : 1,
          transform: phase === 1 || phase === 2 ? "translateX(20px)" : "translateX(0)",
          transition: "transform 0.9s cubic-bezier(.22,1,.36,1), opacity 0.6s",
        }}>
          {inputCubes.map((c, i) => (
            <EncryptedCube key={i} size={Math.min(50, Math.max(34, window.innerWidth * 0.04))} colors={c.colors}
              spin={c.spin} floatDelay={c.floatDelay} label={c.label} locked={true} />
          ))}
        </div>

        {/* Arrow in */}
        <div style={{ opacity: phase >= 1 ? 1 : 0.3, transition: "opacity 0.4s" }}>
          <PixelArrow color="#f0c030" />
        </div>

        {/* Compute box */}
        <div style={{ position: "relative" }}>
          <ComputeBox active={phase === 2} />
          <Particles active={burst} />
        </div>

        {/* Arrow out */}
        <div style={{ opacity: phase >= 3 ? 1 : 0.3, transition: "opacity 0.4s" }}>
          <PixelArrow color="#f0c030" />
        </div>

        {/* Right: output cubes (still locked!) */}
        <div style={{
          display: "flex",
          gap: "clamp(6px,1.5vw,14px)",
          alignItems: "center",
          flexWrap: "wrap",
          justifyContent: "center",
          opacity: phase === 3 ? 1 : 0.2,
          transform: phase === 3 ? "translateX(0)" : "translateX(-16px)",
          transition: "transform 0.7s cubic-bezier(.22,1,.36,1), opacity 0.5s",
        }}>
          {inputCubes.map((c, i) => (
            <EncryptedCube key={i} size={Math.min(50, Math.max(34, window.innerWidth * 0.04))}
              colors={[c.colors[0], c.colors[1], c.colors[2]]}
              spin={i % 2 === 0 ? "spin-r" : "spin"}
              floatDelay={i * 0.3}
              label={c.label}
              locked={true} />
          ))}
        </div>
      </div>

      {/* Caption bar */}
      <div className="he-caption-bar">
        <div style={{
          fontFamily: "system-ui,sans-serif",
          fontSize: "clamp(11px, 1.5vw, 13px)",
          color: "#cbd5e1",
          transition: "opacity 0.3s",
        }}>
          <span style={{ color: "#f0c030", marginRight: 6 }}>▸</span>
          {captions[phase]}
        </div>
        <div style={{ display: "flex", gap: 16 }}>
          {[
            { label: "BFV", color: "#4285f4" },
            { label: "BGV", color: "#9333ea" },
            { label: "BFV", color: "#10b981" },
          ].map((s, i) => (
            <span key={i} style={{
              fontFamily: "'Press Start 2P',monospace",
              fontSize: 7,
              color: s.color,
              opacity: 0.8,
            }}>{s.label}</span>
          ))}
        </div>
      </div>

      {/* KEY INSIGHT row */}
      <div className="he-key-insights">
        {[
          { icon: "🔒", text: "Data stays encrypted throughout computation" },
          { icon: "⚡", text: "5 HE primitives measured: KeyGen · Encrypt · Add · Multiply · Decrypt" },
          { icon: "🧮", text: "SEAL & OpenFHE use BFV · HELib uses BGV" },
        ].map((item, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 8,
            fontFamily: "system-ui,sans-serif",
            fontSize: "clamp(10px, 1.3vw, 12px)",
            color: "#94a3b8",
          }}>
            <span style={{ fontSize: 13 }}>{item.icon}</span>
            {item.text}
          </div>
        ))}
      </div>
    </div>
  );
}
