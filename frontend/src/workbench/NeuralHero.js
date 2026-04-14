import { useEffect, useRef, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";

/* ═══════════════════════════════════════════════════════════
   NeuralHero — Manga × anime.js style
   · One giant CSS-3D cube centred on screen
   · Continuous Y-axis rotation
   · Scroll drives: dark → vivid colour shift
   · Manga halftone bg, speed lines, ink outlines
   · SplitText staggered headline
   ═══════════════════════════════════════════════════════════ */

/* ─── Staggered split-text ─── */
function SplitText({ text, delay = 0, staggerMs = 48, style = {} }) {
  const chars = text.split("");
  return (
    <span style={{ display: "inline", ...style }}>
      {chars.map((ch, i) => (
        <span key={i} style={{ display: "inline-block", overflow: "hidden", verticalAlign: "bottom" }}>
          <motion.span
            style={{ display: "inline-block", whiteSpace: "pre" }}
            initial={{ y: "115%", opacity: 0 }}
            animate={{ y: "0%", opacity: 1 }}
            transition={{ duration: 0.7, delay: delay + i * (staggerMs / 1000), ease: [0.16, 1, 0.3, 1] }}
          >
            {ch === " " ? "\u00a0" : ch}
          </motion.span>
        </span>
      ))}
    </span>
  );
}

/* ─── Speed lines (manga) ─── */
function SpeedLines({ opacity }) {
  const lines = Array.from({ length: 24 }, (_, i) => {
    const angle = (i / 24) * 360;
    return { angle, len: 38 + (i % 3) * 12 };
  });
  return (
    <svg
      viewBox="0 0 200 200"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", opacity }}
    >
      {lines.map((l, i) => {
        const rad = (l.angle * Math.PI) / 180;
        const cx = 100, cy = 100;
        const x1 = cx + Math.cos(rad) * 22;
        const y1 = cy + Math.sin(rad) * 22;
        const x2 = cx + Math.cos(rad) * (22 + l.len);
        const y2 = cy + Math.sin(rad) * (22 + l.len);
        return (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke="currentColor" strokeWidth={i % 4 === 0 ? "1.5" : "0.6"} opacity="0.18" />
        );
      })}
    </svg>
  );
}

/* ─── Halftone dot grid ─── */
function HalftoneBg({ color }) {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none",
      backgroundImage: `radial-gradient(circle, ${color} 1.2px, transparent 1.2px)`,
      backgroundSize: "22px 22px",
      transition: "background-image 0.3s",
    }} />
  );
}

/* ─── Ink border panel (manga panel style) ─── */
function InkBorder({ color }) {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none",
      border: `4px solid ${color}`,
      boxShadow: `8px 8px 0 ${color}`,
      zIndex: 30,
    }} />
  );
}

/* ─── Lock SVG with ink outline ─── */
function LockIcon({ size, color, bg }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect x="3" y="10" width="18" height="13" rx="2" fill={bg} stroke={color} strokeWidth="2.5"/>
      <path d="M8 10V6.5a4 4 0 0 1 8 0V10" stroke={color} strokeWidth="2.5" strokeLinecap="round" fill="none"/>
      <circle cx="12" cy="16.5" r="2" fill={color}/>
      <rect x="11" y="17" width="2" height="3" rx="1" fill={color}/>
    </svg>
  );
}

/* ─── Number cipher face ─── */
function CipherFace({ nums, color, bg, size }) {
  const rows = [
    ["FF","A3","7B","1E"],
    ["9C","2D","E8","04"],
    ["5F","B1","3A","CC"],
  ];
  return (
    <div style={{
      width: size, height: size, background: bg,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2,
      overflow: "hidden", padding: 4,
    }}>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex", gap: 4 }}>
          {row.map((n, ci) => (
            <span key={ci} style={{
              fontFamily: "monospace", fontSize: size * 0.085,
              color, opacity: 0.55 + (ri * 4 + ci) % 3 * 0.15,
              fontWeight: 700, letterSpacing: "0.02em",
            }}>
              {nums[(ri * 4 + ci) % nums.length] || n}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

/* ─── The Big 3D Cube ─── */
function BigCube({ rotY, rotX, size, def, brightness }) {
  const h = size / 2;
  const borderAlpha = Math.round((0.3 + brightness * 0.7) * 255).toString(16).padStart(2,"0");
  const outlineColor = def.accent + borderAlpha;

  const faceBase = {
    position: "absolute", width: size, height: size, boxSizing: "border-box",
    border: `3px solid ${outlineColor}`,
    backfaceVisibility: "hidden", WebkitBackfaceVisibility: "hidden",
    overflow: "hidden",
  };

  const faceBg = (lightness) => {
    // dark mode: near-black tinted; bright mode: vivid tinted
    const darkBg = def.darkFace;
    const brightBg = def.brightFace;
    return `color-mix(in srgb, ${brightBg} ${Math.round(brightness * 100)}%, ${darkBg})`;
  };

  return (
    <div
      style={{
        width: size, height: size, position: "relative",
        transformStyle: "preserve-3d",
        transform: `rotateX(${rotX}deg) rotateY(${rotY}deg)`,
      }}
    >
      {/* FRONT — Lock face */}
      <div style={{ ...faceBase, background: faceBg(1), transform: `translateZ(${h}px)` }}>
        <div style={{ width:"100%", height:"100%", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap: size*0.04 }}>
          <LockIcon size={size * 0.38} color={def.accent} bg="transparent" />
          <span style={{
            fontFamily: "'Press Start 2P', monospace",
            fontWeight: 700,
            fontSize: size * 0.075, color: def.accent, letterSpacing: "0.1em",
            textTransform: "uppercase", opacity: 0.9,
          }}>
            {def.name}
          </span>
        </div>
        {/* Manga outline accent */}
        <div style={{ position:"absolute", inset:0, border:`5px solid ${def.accent}`, opacity: brightness * 0.25 }} />
      </div>

      {/* BACK — cipher */}
      <div style={{ ...faceBase, background: faceBg(0.7), transform: `translateZ(-${h}px) rotateY(180deg)` }}>
        <CipherFace nums={def.nums} color={def.accent} bg="transparent" size={size} />
      </div>

      {/* LEFT */}
      <div style={{ ...faceBase, background: faceBg(0.5), transform: `translateX(-${h}px) rotateY(-90deg)` }}>
        <CipherFace nums={def.nums} color={def.accent} bg="transparent" size={size} />
      </div>

      {/* RIGHT */}
      <div style={{ ...faceBase, background: faceBg(0.5), transform: `translateX(${h}px) rotateY(90deg)` }}>
        <div style={{ width:"100%", height:"100%", display:"flex", alignItems:"center", justifyContent:"center" }}>
          <span style={{ fontFamily:"monospace", fontSize: size*0.1, color:def.accent, opacity:0.7, fontWeight:700 }}>
            {def.label}
          </span>
        </div>
      </div>

      {/* TOP */}
      <div style={{ ...faceBase, background: faceBg(0.3), transform: `translateY(-${h}px) rotateX(90deg)` }}>
        <div style={{ width:"100%", height:"100%", display:"flex", alignItems:"center", justifyContent:"center" }}>
          <span style={{ fontFamily:"monospace", fontSize: size*0.1, color:def.accent, opacity:0.5, fontWeight:700 }}>
            {def.scheme}
          </span>
        </div>
      </div>

      {/* BOTTOM */}
      <div style={{ ...faceBase, background: faceBg(0.2), transform: `translateY(${h}px) rotateX(-90deg)` }} />

      {/* Glow behind */}
      <div style={{
        position:"absolute",
        top: -h * 0.5, left: -h * 0.5, right: -h * 0.5, bottom: -h * 0.5,
        background: `radial-gradient(circle, ${def.accent}${Math.round(brightness * 0.35 * 255).toString(16).padStart(2,"0")} 0%, transparent 65%)`,
        transform: `translateZ(-${h * 0.6}px)`,
        pointerEvents: "none",
      }} />
    </div>
  );
}

/* ─── Library defs — Stardew Valley pixel palette ─── */
const LIBS = [
  {
    name: "SEAL",    scheme: "BFV", label: "MICROSOFT",
    accent: "#6aabf7",
    darkFace: "#0b1a30", brightFace: "#2a5a8a",
    nums: ["FF","A3","7B","1E","9C","2D","E8","04"],
    tagline: "Microsoft SEAL",
    by: "Microsoft Research, 2015",
    scheme_full: "BFV / CKKS",
    why: "The most widely deployed HE library. SEAL implements the BFV and CKKS schemes with a clean C++ API, making it the go-to baseline for benchmarking. Its batching via NTT packs 4096 integers into a single ciphertext, giving it strong throughput on addition-heavy workloads.",
    facts: [
      "Used in Azure Confidential Computing",
      "CKKS enables approximate real-number arithmetic",
      "128-bit security with ring dimension n = 4096",
    ],
  },
  {
    name: "HELib",   scheme: "BGV", label: "IBM",
    accent: "#c890f0",
    darkFace: "#150828", brightFace: "#5a2a88",
    nums: ["3F","C1","88","EA","27","B4","6D","F0"],
    tagline: "IBM HELib",
    by: "IBM Research, 2013",
    scheme_full: "BGV / CKKS",
    why: "The oldest production-grade HE library, built around the BGV scheme by Brakerski-Gentry-Vaikuntanathan. HELib introduced bootstrapping and the Chinese Remainder Theorem-based slot packing, making it the academic reference point. It is slower in raw throughput but uniquely supports deep computation circuits.",
    facts: [
      "First library to support bootstrapping (2014)",
      "BGV scheme: modulus switching drains noise",
      "Basis for much of the academic HE literature",
    ],
  },
  {
    name: "OpenFHE", scheme: "CKKS", label: "COMMUNITY",
    accent: "#58c896",
    darkFace: "#071a10", brightFace: "#1a6040",
    nums: ["7E","51","A0","3C","D9","14","8B","F3"],
    tagline: "OpenFHE",
    by: "PALISADE / Duality Technologies, 2022",
    scheme_full: "BFV / BGV / CKKS / FHEW / TFHE",
    why: "The newest and most scheme-complete open-source library. OpenFHE unifies five FHE schemes under one API, is hardware-acceleration-ready, and targets industry adoption. It emerged from the PALISADE project and is now the benchmark target for NIST PQC-aligned HE research.",
    facts: [
      "Supports five schemes: BFV, BGV, CKKS, FHEW, TFHE",
      "Multi-party HE and threshold decryption built in",
      "Designed for cloud and edge deployment",
    ],
  },
];

/* ═══════════════════════════════ MAIN COMPONENT ══════════════════════════════ */
export default function NeuralHero() {
  const containerRef = useRef(null);
  const { scrollY }  = useScroll();
  const vh = typeof window !== "undefined" ? window.innerHeight : 800;

  /* Scroll fraction 0→1 over 400vh */
  const scrollPct = useTransform(scrollY, [0, vh * 4], [0, 1]);

  /* brightness: 0 (dark) → 1 (vivid) as user scrolls */
  const brightnessVal = useTransform(scrollY, [0, vh * 2], [0, 1]);

  /* Text overlays */
  const titleOp   = useTransform(scrollY, [0, vh * 0.6], [1, 0]);
  const hintOp    = useTransform(scrollY, [0, vh * 0.3], [1, 0]);

  /* Live state */
  const [t, setT]           = useState(0);
  const [bright, setBright] = useState(0);
  useEffect(() => scrollPct.on("change", v => setT(v)), [scrollPct]);
  useEffect(() => brightnessVal.on("change", v => setBright(Math.max(0, Math.min(1, v)))), [brightnessVal]);

  /* Which lib cube is active — cycles every 1/3 of scroll */
  const activeLib = Math.min(2, Math.floor(t * 3));

  /* Rotation: continuous spin, speed controlled by scroll */
  const tick = useRef(0);
  const [rotY, setRotY] = useState(0);
  const [rotX, setRotX] = useState(-18);

  useEffect(() => {
    let raf;
    const loop = (ts) => {
      tick.current = ts * 0.001;
      // Base spin + scroll-driven tilt
      setRotY(ts * 0.05);        // ~18°/s
      setRotX(-18 + t * 36);     // tilt from -18° to +18°
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [t]);

  /* Background interpolation: dark (#050508) → vivid lib accent */
  const def = LIBS[activeLib];
  const eOut = v => 1 - Math.pow(1 - Math.max(0, Math.min(1, v)), 3);
  const br = eOut(bright);

  /* Stardew palette:
     dark start  = deep night sky  #0e1a2a  (14,26,42)
     bright end  = warm daytime sky per lib  */
  const bgVivid = activeLib === 0 ? "30,58,100"   // SEAL  → warm dusk blue
                : activeLib === 1 ? "38,22,68"     // HELib → twilight purple
                                  : "18,52,36";    // OpenFHE → forest evening
  const bgR = Math.round(14 + br * (parseInt(bgVivid.split(",")[0]) - 14));
  const bgG = Math.round(26 + br * (parseInt(bgVivid.split(",")[1]) - 26));
  const bgB = Math.round(42 + br * (parseInt(bgVivid.split(",")[2]) - 42));
  const bgColor = `rgb(${bgR},${bgG},${bgB})`;

  /* Dot grid: starts as soft white stars, warms to gold/accent as brightness grows */
  const dotColor = br < 0.4
    ? `rgba(255,248,220,${0.06 + br * 0.12})`   // cream-white star dots at night
    : `rgba(240,192,48,${0.06 + (br-0.4) * 0.22})`; // warm gold in daytime

  /* Cube size — larger, textbook diagram scale */
  const cubeSize = typeof window !== "undefined"
    ? Math.min(Math.round(Math.min(window.innerWidth, window.innerHeight) * 0.46), 420)
    : 300;

  return (
    <div ref={containerRef} style={{ height: "500vh" }}>
      <div
        className="sticky top-0 w-full overflow-hidden"
        style={{ height: "100vh", background: bgColor, transition: "background 0.12s linear" }}
      >
        {/* Halftone bg */}
        <HalftoneBg color={dotColor} />

        {/* Speed lines — warm cream at night, gold in daytime */}
        <div style={{
          position:"absolute", inset:0, pointerEvents:"none",
          color: br < 0.4 ? "rgba(255,248,220,1)" : "rgba(240,192,48,1)",
          transition: "color 0.6s",
        }}>
          <SpeedLines opacity={0.4 + br * 0.35} />
        </div>

        {/* Ink border — gold tinted like Stardew dialogue box */}
        <InkBorder color={`rgba(240,192,48,${0.08 + br * 0.45})`} />

        {/* ─── Hero title (fades out on scroll) ─── */}
        <motion.div
          style={{
            opacity: titleOp,
            position:"absolute", top:0, left:0, right:0,
            display:"flex", flexDirection:"column", alignItems:"center",
            paddingTop:"clamp(2rem,5vh,4rem)",
            zIndex:20, pointerEvents:"none",
          }}
        >
          {/* Eyebrow — warm cream, humble tag */}
          <motion.div
            initial={{ opacity:0, y:10 }}
            animate={{ opacity:1, y:0 }}
            transition={{ duration:0.6, delay:0.1 }}
            style={{
              fontFamily:"'Press Start 2P', monospace",
              fontSize:"clamp(0.42rem,0.7vw,0.58rem)",
              letterSpacing:"0.28em", textTransform:"uppercase",
              color:"rgba(240,220,160,0.55)", marginBottom:"1.6rem",
              display:"flex", alignItems:"center", gap:14,
            }}
          >
            <span style={{ width:24, height:1, background:"rgba(240,192,48,0.3)", display:"inline-block" }} />
            a little explorer for HE libraries
            <span style={{ width:24, height:1, background:"rgba(240,192,48,0.3)", display:"inline-block" }} />
          </motion.div>

          {/* Headline — warm, humble, pixel-ish weight */}
          <h1 style={{ margin:0, textAlign:"center", lineHeight:1.12, letterSpacing:"-0.01em" }}>
            {/* Line 1 — cream white, friendly size */}
            <div style={{
              display:"block",
              fontSize:"clamp(2rem,5vw,4.5rem)",
              fontWeight:800,
              fontFamily:"system-ui,-apple-system,sans-serif",
              color:"rgba(255,248,220,0.92)",   // warm cream
              letterSpacing:"-0.02em",
            }}>
              <SplitText text="Explore Homomorphic" delay={0.2} staggerMs={32} />
            </div>
            {/* Line 2 — gold accent, stardew warm */}
            <div style={{
              display:"block",
              fontSize:"clamp(2rem,5vw,4.5rem)",
              fontWeight:800,
              fontFamily:"system-ui,-apple-system,sans-serif",
              color:"#f0c030",   // Stardew gold
              letterSpacing:"-0.02em",
              marginTop:"0.06em",
            }}>
              <SplitText text="Encryption." delay={0.48} staggerMs={38} />
            </div>
          </h1>

          {/* Sub — gentle, inviting */}
          <motion.p
            initial={{ opacity:0 }}
            animate={{ opacity:1 }}
            transition={{ delay:1.5, duration:0.8 }}
            style={{
              marginTop:"1.4rem",
              fontFamily:"system-ui,sans-serif",
              fontSize:"clamp(0.85rem,1.3vw,1.05rem)",
              fontWeight:400,
              color:"rgba(255,240,190,0.42)",
              textAlign:"center", maxWidth:460,
              letterSpacing:"0.01em", lineHeight:1.7,
            }}
          >
            See how SEAL, HELib &amp; OpenFHE compute on locked data —
            no decryption needed.
          </motion.p>
        </motion.div>

        {/* ─── Scroll hint ─── */}
        <motion.div
          style={{
            opacity: hintOp,
            position:"absolute", bottom:"6%", left:"50%",
            transform:"translateX(-50%)", zIndex:20, pointerEvents:"none",
            display:"flex", flexDirection:"column", alignItems:"center", gap:6,
          }}
        >
          <motion.div
            animate={{ y:[0,7,0] }}
            transition={{ duration:2, repeat:Infinity, ease:"easeInOut" }}
            style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:5 }}
          >
            <span style={{
              fontFamily:"'Press Start 2P',monospace", fontSize:"clamp(0.4rem,0.6vw,0.52rem)",
              letterSpacing:"0.3em", textTransform:"uppercase",
              color:"rgba(240,220,160,0.28)",
            }}>Scroll</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(240,192,48,0.28)" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7"/>
            </svg>
          </motion.div>
        </motion.div>

        {/* ─── 3D Cube (centred, always visible) ─── */}
        {(() => {
          // progress within the current library's scroll phase (0 → 1)
          const libT = Math.max(0, Math.min(1, (t - activeLib / 3) * 3));
          // ruler line draws in across first 35% of phase
          const lineScale = Math.min(1, libT / 0.35);
          // main body text fades in after 20%
          const textOp   = Math.max(0, Math.min(1, (libT - 0.20) / 0.38));
          // callout facts appear after 45%
          const factsOp  = Math.max(0, Math.min(1, (libT - 0.42) / 0.35));
          const descDef  = LIBS[activeLib];

          return (
        <div style={{
          position:"absolute", inset:0,
          display:"flex", flexDirection:"column", alignItems:"center",
          justifyContent:"flex-start",
          perspective: cubeSize * 4,
          perspectiveOrigin:"50% 38%",
          zIndex:10,
          paddingTop:"clamp(4rem,9vh,7rem)",
        }}>

          {/* Lib selector pills */}
          <div style={{
            display:"flex", justifyContent:"center", gap:22, marginBottom:"clamp(0.8rem,2vh,1.4rem)",
          }}>
            {LIBS.map((lib, i) => (
              <div key={i} style={{
                fontFamily:"'Press Start 2P', monospace",
                fontSize:"clamp(0.42rem,0.7vw,0.58rem)",
                letterSpacing:"0.18em", textTransform:"uppercase",
                color: i === activeLib ? "#f0c030" : "rgba(240,220,160,0.28)",
                fontWeight:700,
                borderBottom: i === activeLib ? "2px solid #f0c030" : "2px solid transparent",
                paddingBottom:4,
                transition:"all 0.5s",
              }}>
                {lib.name}
              </div>
            ))}
          </div>

          {/* ─── Cube — large, centred, textbook figure ─── */}
          <BigCube rotY={rotY} rotX={rotX} size={cubeSize} def={def} brightness={br} />

          {/* ─── Textbook figure label + ruler ─── */}
          <div style={{
            width:"100%", maxWidth:780,
            marginTop:"clamp(1.6rem,3.5vh,2.8rem)",
            padding:"0 clamp(1rem,5vw,3rem)",
            boxSizing:"border-box",
            opacity: br > 0.06 ? 1 : 0,
            transition:"opacity 0.5s",
          }}>

            {/* Figure label row — "Figure 1 · SEAL" like a textbook */}
            <div style={{
              display:"flex", alignItems:"center", gap:12,
              marginBottom:"clamp(0.55rem,1.2vh,0.9rem)",
              opacity: textOp, transition:"opacity 0.4s",
            }}>
              {/* Figure number badge */}
              <span style={{
                fontFamily:"'Press Start 2P', monospace",
                fontSize:"clamp(0.38rem,0.55vw,0.5rem)",
                letterSpacing:"0.18em", textTransform:"uppercase",
                color: descDef.accent,
                background:`${descDef.accent}1a`,
                border:`1.5px solid ${descDef.accent}50`,
                borderRadius:3, padding:"4px 10px",
                flexShrink:0,
              }}>
                Fig {activeLib + 1}
              </span>
              {/* Scheme badge */}
              <span style={{
                fontFamily:"'Press Start 2P', monospace",
                fontSize:"clamp(0.38rem,0.55vw,0.5rem)",
                letterSpacing:"0.16em", textTransform:"uppercase",
                color:"rgba(255,248,210,0.5)",
                flexShrink:0,
              }}>
                {descDef.scheme_full}
              </span>
              {/* Thin rule to fill remaining width */}
              <div style={{
                flex:1, height:1,
                background:`linear-gradient(to right, ${descDef.accent}60, transparent)`,
                transformOrigin:"left center",
                transform:`scaleX(${lineScale})`,
                transition:"transform 0.05s linear",
              }} />
            </div>

            {/* Library name — textbook section heading size */}
            <div style={{
              fontFamily:"system-ui, -apple-system, sans-serif",
              fontSize:"clamp(1.1rem,2.2vw,1.7rem)",
              fontWeight:700,
              color:"rgba(255,248,220,0.96)",
              letterSpacing:"-0.01em",
              lineHeight:1.2,
              marginBottom:"clamp(0.5rem,1vh,0.8rem)",
              opacity: textOp, transition:"opacity 0.4s",
            }}>
              {descDef.tagline}
              <span style={{
                fontSize:"0.52em", fontWeight:400,
                color:"rgba(255,248,210,0.38)",
                marginLeft:"0.7em", letterSpacing:"0.02em",
              }}>
                — {descDef.by}
              </span>
            </div>

            {/* Body text — generous size, textbook paragraph style */}
            <p style={{
              fontFamily:"Georgia, 'Times New Roman', serif",
              fontSize:"clamp(0.85rem,1.3vw,1.05rem)",
              color:"rgba(255,248,210,0.78)",
              lineHeight:1.85,
              letterSpacing:"0.01em",
              margin:"0 0 clamp(0.9rem,2vh,1.4rem) 0",
              opacity: textOp, transition:"opacity 0.5s",
            }}>
              {descDef.why}
            </p>

            {/* ─── Callout facts row — like textbook infoboxes ─── */}
            <div style={{
              display:"flex", gap:"clamp(8px,1.5vw,16px)",
              opacity: factsOp, transition:"opacity 0.5s",
            }}>
              {descDef.facts.map((f, i) => (
                <div key={i} style={{
                  flex:1,
                  background:`${descDef.accent}0d`,
                  border:`1px solid ${descDef.accent}30`,
                  borderLeft:`3px solid ${descDef.accent}`,
                  borderRadius:4,
                  padding:"clamp(6px,1.2vh,10px) clamp(8px,1.2vw,14px)",
                  display:"flex", flexDirection:"column", gap:4,
                }}>
                  {/* Fact number */}
                  <span style={{
                    fontFamily:"'Press Start 2P', monospace",
                    fontSize:"clamp(0.32rem,0.48vw,0.42rem)",
                    color: descDef.accent, opacity:0.65,
                    letterSpacing:"0.15em",
                  }}>
                    {String(i + 1).padStart(2,"0")}
                  </span>
                  {/* Fact text */}
                  <span style={{
                    fontFamily:"system-ui, sans-serif",
                    fontSize:"clamp(0.65rem,0.95vw,0.82rem)",
                    color:"rgba(255,248,210,0.72)",
                    lineHeight:1.55,
                    fontWeight:500,
                  }}>
                    {f}
                  </span>
                </div>
              ))}
            </div>

          </div>
        </div>
          );
        })()}

        {/* ─── Progress bar — Stardew gold ─── */}
        <div style={{
          position:"absolute", bottom:0, left:0, right:0, height:3,
          background:"rgba(240,192,48,0.08)", zIndex:25,
        }}>
          <div style={{
            height:"100%", width:`${t*100}%`,
            background:`linear-gradient(90deg,#6aabf7,#c890f0,#58c896)`,
            transition:"width 0.06s linear",
          }} />
        </div>
      </div>

      {/* Bottom page fade */}
      <div style={{
        position:"absolute", bottom:0, left:0, right:0, height:220, zIndex:20, pointerEvents:"none",
        background:`linear-gradient(to bottom, transparent 0%, ${bgColor} 40%, #f7f7f7 100%)`,
      }} />
    </div>
  );
}
