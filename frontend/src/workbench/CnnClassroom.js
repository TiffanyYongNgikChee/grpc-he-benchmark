/**
 * CnnClassroom.js
 * ───────────────
 * "Archie's CNN Classroom" — animated explainer between the hero and the
 * MNIST inference panel.
 *
 * Layout
 * ──────
 *   • Intro row: Owl + speech bubble + two action buttons (Tutorial / Skip)
 *   • Tutorial mode: full animated conveyor-belt walkthrough (10 stages)
 *     Each stage auto-advances every 3.5 s OR user clicks Next / Prev.
 *   • All stages emphasise that steps 2-8 are fully encrypted.
 *   • After the last stage (or Skip), collapses smoothly to a compact
 *     "How it works" info strip that stays permanently visible above the
 *     MNIST canvas.
 */

import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════
   CSS / keyframes
   ═══════════════════════════════════════════════════════════ */
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

  @keyframes owlBob     { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
  @keyframes pxBlink    { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes stageSlide { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
  @keyframes convSweep  { 0%{left:0%} 100%{left:66.6%} }
  @keyframes encBlur    { 0%{filter:blur(0px);opacity:1} 50%{filter:blur(3px);opacity:.5} 100%{filter:blur(1.5px);opacity:.8} }
  @keyframes decBlur    { 0%{filter:blur(1.5px);opacity:.8} 100%{filter:blur(0px);opacity:1} }
  @keyframes pxPulse    { 0%,100%{box-shadow:0 0 0 2px #f0c03080} 50%{box-shadow:0 0 0 5px #f0c03050} }
  @keyframes barGrow    { from{width:0} to{width:var(--w)} }
  @keyframes tickFly    { 0%{transform:scale(0) rotate(-10deg);opacity:0} 70%{transform:scale(1.3) rotate(5deg);opacity:1} 100%{transform:scale(1) rotate(0deg);opacity:1} }
  @keyframes lockShake  { 0%,100%{transform:rotate(0deg)} 20%{transform:rotate(-6deg)} 40%{transform:rotate(6deg)} 60%{transform:rotate(-3deg)} 80%{transform:rotate(3deg)} }
  @keyframes pxFadeIn   { from{opacity:0} to{opacity:1} }

  .owl-bob   { animation: owlBob 1.6s ease-in-out infinite; }
  .px-blink  { animation: pxBlink 0.8s step-end infinite; }
  .stage-in  { animation: stageSlide 0.45s cubic-bezier(.22,1,.36,1) both; }
  .conv-sweep { animation: convSweep 1.4s linear infinite; }
  .px-pulse  { animation: pxPulse 1.4s ease-in-out infinite; }
  .px-fadein { animation: pxFadeIn 0.5s ease both; }
`;

/* ═══════════════════════════════════════════════════════════
   Palette
   ═══════════════════════════════════════════════════════════ */
const P = {
  bg:        "#0d0a06",
  panel:     "#1a1208",
  panelMid:  "#241809",
  border:    "#5a3a10",
  borderHi:  "#c07820",
  gold:      "#f0c030",
  goldDim:   "#9a7a20",
  cream:     "#fff4d0",
  dim:       "#7a6040",
  encrypt:   "#4a8fff",   // blue = encrypted
  plain:     "#20d090",   // green = plaintext
  danger:    "#ff5050",
  conv:      "#f0a030",
  act:       "#c060ff",
  pool:      "#20d090",
  fc:        "#4a8fff",
};

/* ═══════════════════════════════════════════════════════════
   Reusable pixel owl (same as OwlGuide)
   ═══════════════════════════════════════════════════════════ */
function PixelOwl({ size = 90, frame = 0, flipped = false }) {
  const EyeL = () => {
    if (frame === 2) return <rect x="16" y="24" width="12" height="3" fill="#0a0502"/>;
    if (frame === 1) return (<><rect x="16" y="19" width="12" height="9" fill="white"/><rect x="16" y="24" width="12" height="4" fill="#0a0502"/></>);
    return (<><rect x="16" y="18" width="12" height="12" fill="white"/><rect x="18" y="20" width="8" height="8" fill="#0a0502"/><rect x="19" y="21" width="2" height="2" fill="white"/></>);
  };
  const EyeR = () => {
    if (frame === 2) return <rect x="36" y="24" width="12" height="3" fill="#0a0502"/>;
    if (frame === 1) return (<><rect x="36" y="19" width="12" height="9" fill="white"/><rect x="36" y="24" width="12" height="4" fill="#0a0502"/></>);
    return (<><rect x="36" y="18" width="12" height="12" fill="white"/><rect x="38" y="20" width="8" height="8" fill="#0a0502"/><rect x="39" y="21" width="2" height="2" fill="white"/></>);
  };
  return (
    <svg viewBox="0 0 64 84" width={size} height={size * 1.31}
      style={{ imageRendering:"pixelated", display:"block", transform: flipped ? "scaleX(-1)" : "none" }}>
      <rect x="10" y="0"  width="8"  height="14" fill="#c07808"/>
      <rect x="46" y="0"  width="8"  height="14" fill="#c07808"/>
      <rect x="12" y="2"  width="4"  height="10" fill="#e09018"/>
      <rect x="48" y="2"  width="4"  height="10" fill="#e09018"/>
      <rect x="8"  y="10" width="48" height="32" fill="#f0c030"/>
      <rect x="4"  y="14" width="56" height="24" fill="#f0c030"/>
      <rect x="8"  y="10" width="48" height="5"  fill="#d8a020"/>
      <EyeL/><EyeR/>
      <rect x="28" y="30" width="8"  height="5"  fill="#f07010"/>
      <rect x="30" y="35" width="4"  height="4"  fill="#f07010"/>
      <rect x="8"  y="27" width="6"  height="4"  fill="#ffaaaa" opacity="0.55"/>
      <rect x="50" y="27" width="6"  height="4"  fill="#ffaaaa" opacity="0.55"/>
      <rect x="4"  y="42" width="56" height="30" fill="#e09018"/>
      <rect x="14" y="44" width="36" height="26" fill="#fce090"/>
      <rect x="20" y="50" width="4"  height="4"  fill="#f0c030" opacity="0.5"/>
      <rect x="32" y="54" width="4"  height="4"  fill="#f0c030" opacity="0.5"/>
      <rect x="26" y="60" width="4"  height="4"  fill="#f0c030" opacity="0.5"/>
      <rect x="0"  y="44" width="10" height="22" fill="#c07808"/>
      <rect x="54" y="44" width="10" height="22" fill="#c07808"/>
      <rect x="14" y="72" width="14" height="6"  fill="#f07010"/>
      <rect x="36" y="72" width="14" height="6"  fill="#f07010"/>
      <rect x="10" y="74" width="4"  height="4"  fill="#f07010"/>
      <rect x="50" y="74" width="4"  height="4"  fill="#f07010"/>
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════
   Typewriter hook
   ═══════════════════════════════════════════════════════════ */
function useTypewriter(text, speed = 28) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    setDisplayed("");
    setDone(false);
    if (!text) return;
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) { clearInterval(timer); setDone(true); }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);
  return { displayed, done };
}

/* ═══════════════════════════════════════════════════════════
   Mini pixel grid (represents feature map)
   ═══════════════════════════════════════════════════════════ */
function PixelGrid({ size, color, encrypted = false, animated = false }) {
  const cells = size * size;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${size}, 1fr)`,
      gap: 1,
      width: size * 8,
      height: size * 8,
      flexShrink: 0,
      filter: encrypted ? "blur(0.8px)" : "none",
    }}>
      {Array.from({ length: cells }).map((_, i) => {
        const brightness = encrypted
          ? Math.random() > 0.5 ? 0.9 : 0.3   // noise pattern
          : 0.15 + Math.sin(i * 0.7) * 0.1 + Math.cos(i * 0.3) * 0.1;
        return (
          <div key={i} style={{
            background: encrypted
              ? `hsl(${210 + (i % 40)}deg, 80%, ${30 + (i % 3) * 20}%)`
              : `rgba(240,192,48,${Math.max(0.1, Math.min(0.95, brightness + (i % 7) * 0.07))})`,
            aspectRatio: "1",
          }}/>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   Convolution sweep animation
   ═══════════════════════════════════════════════════════════ */
function ConvSweepGrid({ inputSize = 7, filterSize = 3 }) {
  const [pos, setPos] = useState({ r: 0, c: 0 });
  useEffect(() => {
    const stride = inputSize - filterSize;  // e.g. 4
    let step = 0;
    const timer = setInterval(() => {
      step = (step + 1) % ((stride + 1) * (stride + 1));
      const r = Math.floor(step / (stride + 1));
      const c = step % (stride + 1);
      setPos({ r, c });
    }, 300);
    return () => clearInterval(timer);
  }, [inputSize, filterSize]);

  const CELL = 18;
  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center" }}>
      {/* Input grid */}
      <div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, marginBottom: 4, textAlign: "center" }}>Input ({inputSize}×{inputSize})</div>
        <div style={{ position: "relative", display: "inline-block" }}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${inputSize}, ${CELL}px)`, gap: 1 }}>
            {Array.from({ length: inputSize * inputSize }).map((_, i) => {
              const row = Math.floor(i / inputSize);
              const col = i % inputSize;
              const inFilter = row >= pos.r && row < pos.r + filterSize && col >= pos.c && col < pos.c + filterSize;
              const val = [0,0,0,0,0,0,0, 0,1,1,1,1,1,0, 0,0,0,0,0,2,0, 0,0,0,1,0,0,0, 0,0,0,2,0,0,0, 0,0,0,1,0,0,0, 0,0,0,0,0,0,0][i] || 0;
              return (
                <div key={i} style={{
                  width: CELL, height: CELL,
                  background: inFilter ? `${P.conv}33` : P.panelMid,
                  border: `1px solid ${inFilter ? P.conv : P.border}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "system-ui", fontSize: 9, color: inFilter ? P.gold : P.dim,
                  fontWeight: inFilter ? 700 : 400,
                  transition: "background 0.15s, border 0.15s",
                }}>{val}</div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Filter */}
      <div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, marginBottom: 4, textAlign: "center" }}>Filter ({filterSize}×{filterSize})</div>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${filterSize}, ${CELL}px)`, gap: 1 }}>
          {[-1,0,1, -1,0,1, -1,0,1].map((v, i) => (
            <div key={i} style={{
              width: CELL, height: CELL,
              background: v < 0 ? "#ff505033" : v > 0 ? `${P.plain}33` : P.panelMid,
              border: `1px solid ${P.border}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "system-ui", fontSize: 9, color: v < 0 ? "#ff8080" : v > 0 ? P.plain : P.dim,
              fontWeight: 700,
            }}>{v > 0 ? `+${v}` : v}</div>
          ))}
        </div>
      </div>

      {/* Output */}
      <div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, marginBottom: 4, textAlign: "center" }}>Output ({inputSize - filterSize + 1}×{inputSize - filterSize + 1})</div>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${inputSize - filterSize + 1}, ${CELL}px)`, gap: 1 }}>
          {Array.from({ length: (inputSize - filterSize + 1) ** 2 }).map((_, i) => {
            const row = Math.floor(i / (inputSize - filterSize + 1));
            const col = i % (inputSize - filterSize + 1);
            const computed = row < pos.r || (row === pos.r && col < pos.c) || row < pos.r;
            const isActive = row === pos.r && col === pos.c;
            return (
              <div key={i} style={{
                width: CELL, height: CELL,
                background: isActive ? `${P.gold}55` : computed ? `${P.conv}22` : P.panelMid,
                border: `1px solid ${isActive ? P.gold : computed ? P.conv : P.border}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: "system-ui", fontSize: 9, color: isActive ? P.gold : P.dim,
                transition: "all 0.15s",
              }}>{computed || isActive ? (isActive ? "?" : "✓") : ""}</div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   x² activation visualiser
   ═══════════════════════════════════════════════════════════ */
function ActivationVis() {
  const before = [3, -1, 2, 4, -2, 5, 0, 1, 1, -3, 4, -1, 2, 0, -2, 3];
  const after  = before.map(v => v * v);
  const [showAfter, setShowAfter] = useState(false);
  useEffect(() => { const t = setTimeout(() => setShowAfter(s => !s), 1400); return () => clearTimeout(t); }, [showAfter]);
  const vals = showAfter ? after : before;
  const max  = Math.max(...after);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim }}>
        {showAfter ? "After x² activation" : "Before activation"}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 3 }}>
        {vals.map((v, i) => (
          <div key={i} style={{
            width: 32, height: 32,
            background: showAfter ? `rgba(192,96,255,${0.15 + (v/max)*0.75})` : (v < 0 ? "#ff505044" : `rgba(240,160,48,${Math.abs(v)/5*0.6 + 0.1})`),
            border: `1px solid ${showAfter ? P.act : P.border}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: "system-ui", fontSize: 10, fontWeight: 700,
            color: showAfter ? "#e0a0ff" : (v < 0 ? "#ff8080" : P.gold),
            transition: "all 0.4s",
          }}>{v}</div>
        ))}
      </div>
      <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim, fontStyle: "italic" }}>
        {showAfter ? "All values ≥ 0, non-linearity added" : "Negative values allowed before"}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   Pooling visualiser (2×2 avg)
   ═══════════════════════════════════════════════════════════ */
function PoolVis() {
  const grid = [9,1,4,16, 4,25,0,1, 1,9,16,1, 4,0,4,9];
  const pooled = [
    Math.round((9+1+4+25)/4),
    Math.round((4+16+0+1)/4),
    Math.round((1+9+4+0)/4),
    Math.round((16+1+4+9)/4),
  ];
  const [highlight, setHighlight] = useState(0);
  useEffect(() => { const t = setInterval(() => setHighlight(h => (h+1)%4), 900); return () => clearInterval(t); }, []);
  const blockRow = Math.floor(highlight/2), blockCol = highlight%2;

  return (
    <div style={{ display: "flex", gap: 20, alignItems: "center", justifyContent: "center", flexWrap: "wrap" }}>
      <div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, marginBottom: 4, textAlign: "center" }}>After x² (4×4)</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,28px)", gap: 2 }}>
          {grid.map((v, i) => {
            const r = Math.floor(i/4), c = i%4;
            const inBlock = Math.floor(r/2) === blockRow && Math.floor(c/2) === blockCol;
            return (
              <div key={i} style={{
                width: 28, height: 28,
                background: inBlock ? `${P.pool}33` : P.panelMid,
                border: `1px solid ${inBlock ? P.pool : P.border}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: "system-ui", fontSize: 9, color: inBlock ? P.plain : P.dim,
                fontWeight: inBlock ? 700 : 400, transition: "all 0.2s",
              }}>{v}</div>
            );
          })}
        </div>
      </div>
      <div style={{ fontFamily: "system-ui", fontSize: 20, color: P.dim }}>→</div>
      <div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, marginBottom: 4, textAlign: "center" }}>After Pool (2×2)</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2,28px)", gap: 2 }}>
          {pooled.map((v, i) => (
            <div key={i} style={{
              width: 28, height: 28,
              background: i === highlight ? `${P.pool}55` : `${P.pool}22`,
              border: `1px solid ${i === highlight ? P.pool : P.border}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "system-ui", fontSize: 11, fontWeight: 700, color: P.plain,
              transition: "all 0.2s",
            }}>{v}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   FC layer visualiser
   ═══════════════════════════════════════════════════════════ */
function FCVis() {
  const logits = [12, 5, 8, 3, 14, 7, 9, 41, 6, 11];
  const max    = Math.max(...logits);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5, width: "100%" }}>
      {logits.map((v, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: "system-ui", fontSize: 11, color: i === 7 ? P.gold : P.dim, width: 14, textAlign: "right", flexShrink: 0 }}>{i}</span>
          <div style={{ flex: 1, height: 14, background: P.panelMid, border: `1px solid ${P.border}`, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${(v/max)*100}%`,
              background: i === 7 ? P.gold : `${P.fc}66`,
              "--w": `${(v/max)*100}%`,
              animation: "barGrow 0.6s ease both",
              animationDelay: `${i * 0.06}s`,
            }}/>
          </div>
          <span style={{ fontFamily: "system-ui", fontSize: 11, fontWeight: 700, color: i === 7 ? P.gold : P.dim, width: 24, flexShrink: 0 }}>{v}</span>
          {i === 7 && <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: P.gold, animation: "tickFly 0.5s ease 0.8s both" }}>★</span>}
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   Lock icon (encrypted badge)
   ═══════════════════════════════════════════════════════════ */
function LockBadge() {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: `${P.encrypt}22`, border: `1.5px solid ${P.encrypt}`,
      borderRadius: 2, padding: "3px 8px",
    }}>
      <svg width="10" height="12" viewBox="0 0 10 12" style={{ flexShrink: 0 }}>
        <rect x="2" y="5" width="6" height="7" fill={P.encrypt}/>
        <rect x="3" y="2" width="4" height="4" fill="none" stroke={P.encrypt} strokeWidth="1.5"/>
        <rect x="4" y="7" width="2" height="2" fill="#0d0a06"/>
      </svg>
      <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: P.encrypt, letterSpacing: "0.05em" }}>ENCRYPTED</span>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   STAGE DEFINITIONS
   ═══════════════════════════════════════════════════════════ */
const STAGES = [
  {
    id: "input",
    label: "1. Input",
    sublabel: "28×28 pixels",
    color: P.plain,
    encrypted: false,
    owlSpeech: "This is your drawing — 784 numbers from 0 (black) to 255 (white). Still fully readable. We're about to lock it up!",
    detail: "Your handwritten digit is captured as a 28×28 grayscale image — 784 pixel values. Each value is then scaled to an integer (e.g. ÷255 × 1000) so it can be loaded into a BFV ciphertext slot.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <PixelGrid size={7} color={P.plain} encrypted={false}/>
        <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim }}>
          784 pixel values · 0–255 · plaintext
        </div>
      </div>
    ),
  },
  {
    id: "encrypt",
    label: "2. Encrypt",
    sublabel: "784 → 1 ciphertext",
    color: P.encrypt,
    encrypted: true,
    owlSpeech: "Now ALL 784 pixels are packed into one BFV ciphertext using OpenFHE. Even I can't read it anymore! The server holds the keys.",
    detail: "OpenFHE packs all 784 scaled pixel values into the 4096 slots of a single BFV ciphertext using NTT (Number Theoretic Transform). The result looks like random numbers — completely unreadable without the secret key.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <PixelGrid size={5} color={P.plain} encrypted={false}/>
          <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: P.gold, animation: "lockShake 0.6s ease 0.3s" }}>→🔒→</div>
          <PixelGrid size={5} color={P.encrypt} encrypted={true}/>
        </div>
        <LockBadge/>
        <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim, textAlign: "center" }}>
          Pixels → BFV ciphertext · 4096 slots · NTT encoded
        </div>
      </div>
    ),
  },
  {
    id: "conv1",
    label: "3. Conv1",
    sublabel: "28×28 → 24×24",
    color: P.conv,
    encrypted: true,
    owlSpeech: "A 5×5 filter slides across the encrypted image — 576 multiplications, all on ciphertext. Zero decryption. The server is blind the whole time!",
    detail: "The Conv1 layer applies a 5×5 filter across all 28×28 pixel positions (stride 1), producing a 24×24 output. Every multiply and add happens between ciphertexts — the server never decrypts. This is why it takes ~seconds instead of microseconds.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <LockBadge/>
        <ConvSweepGrid inputSize={7} filterSize={3}/>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, textAlign: "center" }}>
          Real CNN uses 28×28 input, 5×5 filter → 24×24 output
        </div>
      </div>
    ),
  },
  {
    id: "act1",
    label: "4. x² Activation",
    sublabel: "Non-linearity",
    color: P.act,
    encrypted: true,
    owlSpeech: "Normal CNNs use ReLU — but ReLU needs max(0,x) which requires comparison. You CANNOT compare encrypted numbers! So we use x² instead.",
    detail: "ReLU (max(0,x)) is impossible on ciphertext because it requires a conditional branch. HE can only do addition and multiplication, so we use x² as the activation function — a polynomial that adds non-linearity with just one HE multiplication.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <LockBadge/>
        <ActivationVis/>
      </div>
    ),
  },
  {
    id: "pool1",
    label: "5. AvgPool",
    sublabel: "24×24 → 12×12",
    color: P.pool,
    encrypted: true,
    owlSpeech: "4 neighbours are averaged into 1 — on encrypted data! This halves the grid, keeping the most important features while discarding fine detail.",
    detail: "Average pooling computes the mean of each 2×2 block of ciphertext values. Since HE supports addition and scalar multiplication, this is done as (a+b+c+d) × ¼ — all on ciphertext, no decryption.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <LockBadge/>
        <PoolVis/>
      </div>
    ),
  },
  {
    id: "conv2",
    label: "6. Conv2",
    sublabel: "12×12 → 8×8",
    color: P.conv,
    encrypted: true,
    owlSpeech: "A second 5×5 filter runs on the already-halved encrypted feature map. It looks for higher-level shapes like loops and lines that identify specific digits.",
    detail: "Conv2 applies the same idea as Conv1 but on the smaller 12×12 encrypted feature map. The filter has learned (in PyTorch training) to detect digit-specific patterns. Still fully encrypted — the server is still blind.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <LockBadge/>
        <ConvSweepGrid inputSize={6} filterSize={3}/>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, textAlign: "center" }}>
          12×12 encrypted input · 5×5 filter → 8×8 output
        </div>
      </div>
    ),
  },
  {
    id: "act2pool2",
    label: "7. Act + Pool",
    sublabel: "8×8 → 4×4",
    color: P.act,
    encrypted: true,
    owlSpeech: "x² again, then pool down to 4×4. All encrypted! We now have 16 encrypted features representing your digit. Still fully private.",
    detail: "The second x² activation and 2×2 average pool bring the feature map from 8×8 down to 4×4 = 16 values. These 16 encrypted values capture the essential features of your digit.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <LockBadge/>
        <PoolVis/>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim }}>8×8 → 4×4 · 16 encrypted features remain</div>
      </div>
    ),
  },
  {
    id: "fc",
    label: "8. FC Layer",
    sublabel: "16 → 10 logits",
    color: P.fc,
    encrypted: true,
    owlSpeech: "A 16×10 matrix multiplication on encrypted data — each of the 10 output nodes is a vote for a digit class. All 160 multiplications happen on ciphertext!",
    detail: "The fully-connected layer multiplies the 16 flattened encrypted features by a 16×10 weight matrix, producing 10 encrypted logit values — one score per digit class (0–9). This is a single ciphertext matmul.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, width: "100%" }}>
        <LockBadge/>
        <div style={{ width: "100%", maxWidth: 300 }}>
          <FCVis/>
        </div>
        <div style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim }}>10 encrypted logits — one per digit class</div>
      </div>
    ),
  },
  {
    id: "decrypt",
    label: "9. Decrypt",
    sublabel: "10 ints revealed",
    color: P.plain,
    encrypted: false,
    owlSpeech: "ONLY the 10 final logit numbers are decrypted — nothing else! Your image stays private. This is the ONLY moment the server ever sees a real number.",
    detail: "Using the secret key, OpenFHE decrypts only the 10 output ciphertext slots. The 774 other slots (and your original 784 pixels) are never decrypted. This is privacy-preserving inference.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, width: "100%" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
          <PixelGrid size={4} encrypted={true} color={P.encrypt}/>
          <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: P.gold }}>🔓→</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {[12,5,8,3,14,7,9,41,6,11].map((v,i) => (
              <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim, width: 10 }}>{i}:</span>
                <span style={{ fontFamily: "system-ui", fontSize: 10, fontWeight: 700, color: i === 7 ? P.gold : P.plain }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim, textAlign: "center" }}>
          Only 10 numbers decrypted · your 784 pixels stay private
        </div>
      </div>
    ),
  },
  {
    id: "predict",
    label: "10. Predict",
    sublabel: "argmax → digit",
    color: P.gold,
    encrypted: false,
    owlSpeech: "argmax! The highest logit wins — that's the predicted digit. In this example it's 7 with a score of 41, far above the rest. Your digit is safe and the answer is ready!",
    detail: "The server takes argmax of the 10 decrypted logits — finds the index with the highest value. That index is the predicted digit. The result (just one number) is returned to your browser via gRPC → Spring Boot → React.",
    visual: () => (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
        <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 32, color: P.gold, textShadow: `3px 3px 0 #7a5800`, animation: "tickFly 0.6s ease both" }}>7</div>
        <div style={{ fontFamily: "system-ui", fontSize: 13, color: P.dim }}>argmax([12,5,8,3,14,7,9,<strong style={{color:P.gold}}>41</strong>,6,11]) = 7</div>
        <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim, textAlign: "center", maxWidth: 280 }}>
          Your pixel values were never seen by the server. Only this single digit was revealed.
        </div>
      </div>
    ),
  },
];

/* ═══════════════════════════════════════════════════════════
   Pipeline progress bar (breadcrumb strip)
   ═══════════════════════════════════════════════════════════ */
function PipelineStrip({ current, onSelect }) {
  return (
    <div style={{ display: "flex", alignItems: "center", overflowX: "auto", gap: 0, padding: "6px 0 2px" }}>
      {STAGES.map((s, i) => {
        const done    = i < current;
        const active  = i === current;
        return (
          <div key={s.id} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
            <div
              onClick={() => onSelect(i)}
              title={s.label}
              style={{
                width: 28, height: 28,
                background: done ? s.color : active ? `${s.color}33` : P.panelMid,
                border: `2px solid ${active ? s.color : done ? s.color : P.border}`,
                boxShadow: active ? `0 0 0 3px ${s.color}44` : "none",
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer",
                transition: "all 0.2s",
                fontFamily: "system-ui", fontSize: 9, fontWeight: 700,
                color: done ? P.bg : active ? s.color : P.dim,
              }}
            >{i + 1}</div>
            {i < STAGES.length - 1 && (
              <div style={{ width: 18, height: 2, background: i < current ? STAGES[i].color : P.border, transition: "background 0.3s" }}/>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   INFO STRIP (shown after tutorial or always visible)
   ═══════════════════════════════════════════════════════════ */
function InfoStrip({ onReplay }) {
  return (
    <div style={{
      background: P.panel,
      border: `2px solid ${P.border}`,
      boxShadow: `3px 3px 0 #0a0702`,
      padding: "14px 20px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10, marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <PixelOwl size={40} frame={0}/>
          <div>
            <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: P.gold, marginBottom: 3 }}>
              HOW ENCRYPTED INFERENCE WORKS
            </div>
            <div style={{ fontFamily: "system-ui", fontSize: 12, color: P.dim }}>
              Steps 2–8 run entirely on encrypted ciphertext · Only 10 final numbers are decrypted
            </div>
          </div>
        </div>
        <button onClick={onReplay} style={{
          fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.gold,
          background: P.panelMid, border: `2px solid ${P.gold}`,
          padding: "6px 12px", cursor: "pointer", boxShadow: "2px 2px 0 #0a0702",
        }}>▶ REPLAY TUTORIAL</button>
      </div>

      {/* Mini pipeline pills */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {STAGES.map((s, i) => (
          <div key={s.id} style={{
            display: "flex", alignItems: "center", gap: 4,
            background: s.encrypted ? `${s.color}18` : `${P.plain}10`,
            border: `1.5px solid ${s.encrypted ? s.color : P.plain}55`,
            padding: "3px 8px",
          }}>
            {s.encrypted && (
              <svg width="7" height="9" viewBox="0 0 10 12">
                <rect x="2" y="5" width="6" height="7" fill={s.color}/>
                <rect x="3" y="2" width="4" height="4" fill="none" stroke={s.color} strokeWidth="1.5"/>
              </svg>
            )}
            <span style={{ fontFamily: "system-ui", fontSize: 10, color: s.encrypted ? s.color : P.plain, fontWeight: 600 }}>
              {s.label}
            </span>
          </div>
        ))}
      </div>

      {/* Key facts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginTop: 12 }}>
        {[
          { label: "Encrypted steps", value: "8 of 10", color: P.encrypt },
          { label: "Data ever seen", value: "10 logits only", color: P.plain },
          { label: "Activation fn", value: "x² (not ReLU)", color: P.act },
        ].map(f => (
          <div key={f.label} style={{ background: P.panelMid, border: `1px solid ${P.border}`, padding: "8px 10px" }}>
            <div style={{ fontFamily: "system-ui", fontSize: 18, fontWeight: 800, color: f.color, lineHeight: 1 }}>{f.value}</div>
            <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim, marginTop: 3 }}>{f.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TUTORIAL MODAL (full-width, in-page, not fixed)
   ═══════════════════════════════════════════════════════════ */
function TutorialPanel({ onClose }) {
  const [stage, setStage]       = useState(0);
  const [paused, setPaused]     = useState(false);
  const [owlFrame, setOwlFrame] = useState(0);
  const autoRef = useRef(null);

  const s = STAGES[stage];

  // Typewriter for speech
  const { displayed: speech, done: speechDone } = useTypewriter(s.owlSpeech, 24);

  // Blink owl eyes periodically
  useEffect(() => {
    const t = setInterval(() => setOwlFrame(f => f === 0 ? 2 : 0), 3200);
    return () => clearInterval(t);
  }, []);

  // Auto-advance
  const advance = useCallback(() => {
    setStage(st => {
      if (st < STAGES.length - 1) return st + 1;
      return st; // stay on last
    });
  }, []);

  useEffect(() => {
    if (paused) return;
    autoRef.current = setTimeout(advance, 3800);
    return () => clearTimeout(autoRef.current);
  }, [stage, paused, advance]);

  return (
    <div style={{
      background: P.bg,
      border: `3px solid ${P.borderHi}`,
      boxShadow: `6px 6px 0 #0a0702`,
    }} className="stage-in">
      <style>{CSS}</style>

      {/* Header bar */}
      <div style={{
        background: `linear-gradient(90deg, ${P.panelMid}, ${P.bg})`,
        borderBottom: `2px solid ${P.border}`,
        padding: "8px 16px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 9, color: P.gold }}>
            CNN TUTORIAL
          </span>
          <span style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim }}>
            {stage + 1} / {STAGES.length}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setPaused(p => !p)} style={{
            fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: paused ? P.gold : P.dim,
            background: P.panelMid, border: `2px solid ${paused ? P.gold : P.border}`,
            padding: "4px 8px", cursor: "pointer",
          }}>{paused ? "▶ PLAY" : "⏸ PAUSE"}</button>
          <button onClick={onClose} style={{
            fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: P.dim,
            background: P.panelMid, border: `2px solid ${P.border}`,
            padding: "4px 8px", cursor: "pointer",
          }}>✕ SKIP</button>
        </div>
      </div>

      {/* Pipeline strip */}
      <div style={{ padding: "6px 16px 4px", borderBottom: `1px solid ${P.border}` }}>
        <PipelineStrip current={stage} onSelect={(i) => { setStage(i); clearTimeout(autoRef.current); }}/>
        {/* Encrypted / plaintext legend */}
        <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 10, height: 10, background: P.encrypt }}/>
            <span style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim }}>Encrypted (steps 2–8)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 10, height: 10, background: P.plain }}/>
            <span style={{ fontFamily: "system-ui", fontSize: 10, color: P.dim }}>Plaintext</span>
          </div>
        </div>
      </div>

      {/* Stage content */}
      <div key={stage} className="stage-in" style={{
        display: "grid", gridTemplateColumns: "180px 1fr 1fr", gap: 0,
        minHeight: 320,
      }}>
        {/* Owl column */}
        <div style={{
          background: `linear-gradient(180deg, ${P.panelMid} 0%, ${P.bg} 100%)`,
          borderRight: `2px solid ${P.border}`,
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "flex-end", padding: "12px 8px 0",
          position: "relative", overflow: "hidden",
        }}>
          {/* Stage label above owl */}
          <div style={{ textAlign: "center", marginBottom: 8 }}>
            <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: s.color, marginBottom: 3 }}>
              {s.label}
            </div>
            <div style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim }}>{s.sublabel}</div>
            <div style={{ marginTop: 6 }}>
              {s.encrypted ? <LockBadge/> : (
                <div style={{ display: "inline-flex", alignItems: "center", gap: 4, background: `${P.plain}22`, border: `1.5px solid ${P.plain}`, padding: "3px 8px" }}>
                  <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 7, color: P.plain }}>PLAINTEXT</span>
                </div>
              )}
            </div>
          </div>
          <div className="owl-bob">
            <PixelOwl size={80} frame={owlFrame}/>
          </div>
        </div>

        {/* Speech bubble + detail column */}
        <div style={{
          borderRight: `1px solid ${P.border}`,
          padding: "20px 18px",
          display: "flex", flexDirection: "column", gap: 12,
        }}>
          {/* Speech bubble */}
          <div style={{
            background: "#fffbe0",
            border: `2px solid ${P.gold}`,
            boxShadow: `2px 2px 0 #7a5800`,
            padding: "12px 14px",
            position: "relative",
          }}>
            <div style={{
              position: "absolute", bottom: -10, left: 28,
              width: 0, height: 0,
              borderLeft: "10px solid transparent",
              borderRight: "10px solid transparent",
              borderTop: `10px solid ${P.gold}`,
            }}/>
            <div style={{ fontFamily: "system-ui", fontSize: 13, color: "#3a2008", lineHeight: 1.7, minHeight: 60 }}>
              {speech}
              {!speechDone && <span className="px-blink" style={{ color: P.gold }}>▌</span>}
            </div>
          </div>

          {/* Detail text */}
          <div style={{
            background: P.panelMid, border: `1px solid ${P.border}`,
            padding: "12px 14px",
          }}>
            <div style={{ fontFamily: "system-ui", fontSize: 13, color: P.cream, lineHeight: 1.75, opacity: 0.85 }}>
              {s.detail}
            </div>
          </div>
        </div>

        {/* Visual column */}
        <div style={{
          padding: "20px 18px",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <s.visual/>
        </div>
      </div>

      {/* Nav footer */}
      <div style={{
        borderTop: `2px solid ${P.border}`,
        padding: "10px 16px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: P.panelMid,
      }}>
        <button
          disabled={stage === 0}
          onClick={() => { setStage(s => s - 1); clearTimeout(autoRef.current); }}
          style={{
            fontFamily: "'Press Start 2P', monospace", fontSize: 8,
            color: stage === 0 ? P.dim : P.gold,
            background: P.panel, border: `2px solid ${stage === 0 ? P.border : P.gold}`,
            padding: "6px 14px", cursor: stage === 0 ? "default" : "pointer",
            boxShadow: stage === 0 ? "none" : "2px 2px 0 #0a0702",
          }}>◀ PREV</button>

        {/* Progress dots */}
        <div style={{ display: "flex", gap: 5 }}>
          {STAGES.map((_, i) => (
            <div key={i} style={{
              width: i === stage ? 16 : 6, height: 6,
              background: i < stage ? P.gold : i === stage ? s.color : P.border,
              transition: "all 0.3s",
            }}/>
          ))}
        </div>

        {stage < STAGES.length - 1 ? (
          <button
            onClick={() => { setStage(s => s + 1); clearTimeout(autoRef.current); }}
            style={{
              fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.bg,
              background: P.gold, border: `2px solid ${P.borderHi}`,
              padding: "6px 14px", cursor: "pointer", boxShadow: "2px 2px 0 #0a0702",
            }}>NEXT ▶</button>
        ) : (
          <button onClick={onClose} style={{
            fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.bg,
            background: P.plain, border: `2px solid ${P.pool}`,
            padding: "6px 14px", cursor: "pointer", boxShadow: "2px 2px 0 #0a0702",
          }}>DONE ✓</button>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN EXPORT
   ═══════════════════════════════════════════════════════════ */
export default function CnnClassroom() {
  const [mode, setMode]       = useState("intro"); // "intro" | "tutorial" | "done"
  const [owlFrame, setOwlFrame] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setOwlFrame(f => f === 0 ? 2 : 0), 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{
      background: P.bg,
      borderTop:    `3px solid ${P.borderHi}`,
      borderBottom: `3px solid ${P.border}`,
    }}>
      <style>{CSS}</style>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 24px" }}>

        {/* Section header */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
          <div style={{ width: 3, height: 22, background: P.gold, flexShrink: 0 }}/>
          <span style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: P.gold, letterSpacing: "0.08em" }}>
            HOW DOES ENCRYPTED INFERENCE WORK?
          </span>
        </div>

        {mode === "intro" && (
          <div className="px-fadein" style={{
            display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap",
            background: P.panel, border: `2px solid ${P.border}`,
            boxShadow: `3px 3px 0 #0a0702`, padding: "20px 24px",
          }}>
            <div className="owl-bob" style={{ flexShrink: 0 }}>
              <PixelOwl size={70} frame={owlFrame}/>
            </div>
            <div style={{ flex: 1, minWidth: 260 }}>
              <div style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10, color: P.gold, marginBottom: 8 }}>
                Before you draw your digit…
              </div>
              <div style={{ fontFamily: "system-ui", fontSize: 14, color: P.cream, lineHeight: 1.75, opacity: 0.85, marginBottom: 14 }}>
                When you draw a digit and press run, your pixels are <strong style={{color:P.encrypt}}>encrypted</strong> and sent through a full CNN — conv layers, activations, pooling, and a fully connected layer — <strong style={{color:P.encrypt}}>all on ciphertext</strong>. Only the final 10 scores are decrypted. Want me to show you how?
              </div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button onClick={() => setMode("tutorial")} style={{
                  fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.bg,
                  background: P.gold, border: `2px solid ${P.borderHi}`,
                  padding: "8px 18px", cursor: "pointer", boxShadow: "3px 3px 0 #0a0702",
                }}>▶ SHOW ME HOW</button>
                <button onClick={() => setMode("done")} style={{
                  fontFamily: "'Press Start 2P', monospace", fontSize: 8, color: P.dim,
                  background: P.panelMid, border: `2px solid ${P.border}`,
                  padding: "8px 14px", cursor: "pointer",
                }}>SKIP →</button>
              </div>
            </div>
            {/* Mini encrypted hint */}
            <div style={{
              display: "flex", flexDirection: "column", gap: 6, flexShrink: 0,
              background: P.panelMid, border: `1px solid ${P.border}`, padding: "12px 14px",
            }}>
              {[
                { label: "Steps 1", color: P.plain, text: "Plaintext" },
                { label: "Steps 2–8", color: P.encrypt, text: "Encrypted 🔒" },
                { label: "Step 9–10", color: P.plain, text: "Plaintext" },
              ].map(r => (
                <div key={r.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 8, height: 8, background: r.color, flexShrink: 0 }}/>
                  <span style={{ fontFamily: "system-ui", fontSize: 11, color: r.color, fontWeight: 600 }}>{r.label}</span>
                  <span style={{ fontFamily: "system-ui", fontSize: 11, color: P.dim }}>{r.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {mode === "tutorial" && (
          <TutorialPanel onClose={() => setMode("done")}/>
        )}

        {mode === "done" && (
          <InfoStrip onReplay={() => setMode("tutorial")}/>
        )}

      </div>
    </div>
  );
}
