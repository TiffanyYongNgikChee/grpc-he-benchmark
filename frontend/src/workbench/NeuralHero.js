import { useEffect, useRef, useCallback } from "react";
import { motion, useScroll, useTransform } from "framer-motion";

/*
 * NeuralHero — 3D rotating brain neural network.
 *
 * Phase 1 (scroll 0→0.4): A rotating sphere of neurons — like a brain.
 *         Nodes are arranged on a sphere surface, slowly rotating.
 *         Connections shimmer between nearby nodes.
 *
 * Phase 2 (scroll 0.4→0.8): The sphere unfolds / morphs into a flat CNN
 *         architecture. Nodes slide from their sphere positions into
 *         structured layer columns. Connections rewire.
 *
 * Phase 3 (scroll 0.8→1.0): The CNN fires — signal propagates left→right,
 *         layers light up, output glows.
 *
 * Colors: warm whites, soft rose-gold, electric blue accents.
 */

/* ─── Palette — bright "lights on" look ─── */
const COLORS = {
  bg:        "#030712",
  node:      [255, 240, 220],   // warm bright white-gold
  nodeHot:   [255, 255, 255],   // pure white when active
  conn:      [180, 160, 220],   // visible lavender
  connHot:   [140, 200, 255],   // bright electric blue
  glow1:     [230, 190, 255],   // bright violet glow
  glow2:     [100, 200, 255],   // bright cyan-blue glow
  accent:    [255, 180, 100],   // bright warm orange
  output:    [100, 255, 200],   // bright teal-green for output
};

const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* CNN layer definitions for the flat layout — bright vivid colors */
const CNN = [
  { label: "INPUT",   x: 0.05, n: 6, col: [220,220,240] },
  { label: "ENCRYPT", x: 0.14, n: 4, col: [120,210,255] },
  { label: "CONV 1",  x: 0.23, n: 5, col: [190,160,255] },
  { label: "x²",      x: 0.32, n: 4, col: [230,170,255] },
  { label: "POOL",    x: 0.41, n: 3, col: [230,170,255] },
  { label: "CONV 2",  x: 0.52, n: 5, col: [190,160,255] },
  { label: "x²",      x: 0.61, n: 4, col: [230,170,255] },
  { label: "POOL",    x: 0.70, n: 3, col: [230,170,255] },
  { label: "DENSE",   x: 0.81, n: 4, col: [255,190,120] },
  { label: "OUTPUT",  x: 0.93, n: 3, col: [100,255,200] },
];
const TOTAL_CNN_NODES = CNN.reduce((s, l) => s + l.n, 0);

/* ─── Build data ─── */
function buildData(w, h) {
  const cx = w / 2, cy = h / 2;
  const R = Math.min(w, h) * 0.32; // sphere radius — bigger

  const nodes = [];
  let cnnIdx = 0;

  // For each CNN layer node, create a sphere position + CNN target position
  CNN.forEach((layer, li) => {
    const padX = 60, padY = 100;
    const uW = w - padX * 2, uH = h - padY * 2;
    const sp = uH / (layer.n + 1);

    for (let ni = 0; ni < layer.n; ni++) {
      // Distribute on sphere using golden spiral
      const i = cnnIdx;
      const total = TOTAL_CNN_NODES;
      const phi = Math.acos(1 - 2 * (i + 0.5) / total);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;

      // Sphere 3D coordinates
      const sx = R * Math.sin(phi) * Math.cos(theta);
      const sy = R * Math.sin(phi) * Math.sin(theta);
      const sz = R * Math.cos(phi);

      // CNN flat target
      const tx = padX + layer.x * uW;
      const ty = padY + sp * (ni + 1);

      nodes.push({
        // sphere coords (will be rotated)
        sx, sy, sz,
        // CNN target (flat)
        tx, ty,
        // visual
        layerIdx: li,
        col: layer.col,
        r: 6 + (layer.n <= 3 ? 2.5 : 0),  // much bigger nodes
        breathPhase: Math.random() * Math.PI * 2,
        breathSpeed: 0.5 + Math.random() * 0.8,
      });
      cnnIdx++;
    }
  });

  // Connections — for sphere mode, connect to K nearest neighbours
  // For CNN mode, connect between adjacent layers
  const cnnConns = [];
  const byLayer = {};
  nodes.forEach((n, i) => { (byLayer[n.layerIdx] ??= []).push(i); });
  for (let li = 0; li < CNN.length - 1; li++) {
    (byLayer[li] || []).forEach(fi => {
      const targets = [...(byLayer[li + 1] || [])].sort(() => Math.random() - 0.5).slice(0, 3);
      targets.forEach(ti => cnnConns.push({ from: fi, to: ti }));
    });
  }

  // Sphere proximity connections
  const sphereConns = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].sx - nodes[j].sx;
      const dy = nodes[i].sy - nodes[j].sy;
      const dz = nodes[i].sz - nodes[j].sz;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dist < R * 0.7) {
        sphereConns.push({ from: i, to: j, dist });
      }
    }
  }

  // Particles
  const particles = Array.from({ length: 50 }, () => ({
    connIdx: 0,
    progress: Math.random(),
    speed: 0.003 + Math.random() * 0.005,
    size: 1 + Math.random() * 1.8,
  }));

  return { nodes, cnnConns, sphereConns, particles, cx, cy, R };
}

export default function NeuralHero() {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const dataRef = useRef(null);
  const scrollRef = useRef(0);
  const sizeRef = useRef({ w: 1200, h: 800 });

  const { scrollY } = useScroll();

  useEffect(() => {
    return scrollY.on("change", (v) => {
      scrollRef.current = Math.min(1, Math.max(0, v / (window.innerHeight * 4)));
    });
  }, [scrollY]);

  useEffect(() => {
    const onR = () => {
      sizeRef.current = { w: window.innerWidth, h: window.innerHeight };
      dataRef.current = buildData(window.innerWidth, window.innerHeight);
    };
    onR();
    window.addEventListener("resize", onR);
    return () => window.removeEventListener("resize", onR);
  }, []);

  const draw = useCallback(() => {
    const cvs = canvasRef.current, data = dataRef.current;
    if (!cvs || !data) { animRef.current = requestAnimationFrame(draw); return; }

    const { w, h } = sizeRef.current;
    const dpr = window.devicePixelRatio || 1;
    cvs.width = w * dpr; cvs.height = h * dpr;
    const ctx = cvs.getContext("2d");
    ctx.scale(dpr, dpr);

    const t = scrollRef.current; // 0..1
    const now = Date.now() * 0.001;
    const { nodes, cnnConns, sphereConns, particles, cx, cy, R } = data;

    /* ── Phases ──
     * 0.0 → 0.4  : Sphere rotating (morph = 0)
     * 0.4 → 0.75 : Sphere morphs into flat CNN (morph 0→1)
     * 0.75→ 1.0  : CNN fires signal left→right
     */
    const morphT = clamp((t - 0.35) / 0.35, 0, 1);
    const morph = ease(morphT);
    const fireT = clamp((t - 0.72) / 0.28, 0, 1);

    // Rotation angle driven by scroll + slow auto-rotation
    const autoRot = now * 0.15;
    const scrollRot = t * Math.PI * 3;
    const rotY = autoRot + scrollRot;
    const rotX = 0.3 + t * 0.2; // slight tilt

    /* ── Background ── */
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Subtle radial glow in center
    const bg1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 2);
    bg1.addColorStop(0, `rgba(${COLORS.glow1[0]},${COLORS.glow1[1]},${COLORS.glow1[2]},${0.08 * (1 - morph * 0.5)})`);
    bg1.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = bg1;
    ctx.fillRect(0, 0, w, h);

    /* ── Project each node ── */
    const projected = nodes.map((n) => {
      // Rotate sphere coordinates
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

      let x1 = n.sx * cosY - n.sz * sinY;
      let z1 = n.sx * sinY + n.sz * cosY;
      let y1 = n.sy;

      let y2 = y1 * cosX - z1 * sinX;
      let z2 = y1 * sinX + z1 * cosX;

      // 2D projection (weak perspective)
      const perspective = 800;
      const scale = perspective / (perspective + z2);
      const sphereX = cx + x1 * scale;
      const sphereY = cy + y2 * scale;
      const depth = z2; // for sizing/alpha

      // Blend between sphere and CNN positions
      const breath = Math.sin(now * n.breathSpeed + n.breathPhase) * 0.15;
      const finalX = lerp(sphereX, n.tx, morph);
      const finalY = lerp(sphereY, n.ty, morph);
      const finalR = n.r * (1 + breath * 0.3) * lerp(scale, 1, morph);
      const depthAlpha = lerp(clamp(0.5 + (depth + R) / (2 * R) * 0.5, 0.35, 1), 1, morph);

      return { x: finalX, y: finalY, r: finalR, depth, depthAlpha, scale, layerIdx: n.layerIdx };
    });

    // Sort by depth for sphere mode (back-to-front rendering)
    const sortedIndices = projected.map((_, i) => i)
      .sort((a, b) => projected[a].depth - projected[b].depth);

    /* ── Draw connections ── */
    // Sphere connections (fade out as morphing) — bright glowing lines
    const sphereConnAlpha = (1 - morph) * 0.55;
    if (sphereConnAlpha > 0.005) {
      sphereConns.forEach(c => {
        const pA = projected[c.from], pB = projected[c.to];
        const avgDepth = (pA.depthAlpha + pB.depthAlpha) / 2;
        const alpha = sphereConnAlpha * avgDepth;
        if (alpha < 0.005) return;

        const col = COLORS.conn;
        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha})`;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });
    }

    // CNN connections (fade in as morphing)
    const cnnConnAlpha = morph;
    if (cnnConnAlpha > 0.01) {
      cnnConns.forEach(c => {
        const pA = projected[c.from], pB = projected[c.to];

        // During fire phase, connections light up left-to-right
        const layerProgress = pA.layerIdx / (CNN.length - 1);
        const fireHere = clamp((fireT - layerProgress * 0.8) / 0.3, 0, 1);
        const baseAlpha = cnnConnAlpha * lerp(0.2, 0.6, fireHere);

        const colA = nodes[c.from].col;
        const colB = nodes[c.to].col;
        const fr = Math.round(lerp(colA[0], 255, fireHere * 0.3));
        const fg = Math.round(lerp(colA[1], 255, fireHere * 0.3));
        const fb = Math.round(lerp(colA[2], 255, fireHere * 0.3));
        const tr = Math.round(lerp(colB[0], 255, fireHere * 0.3));
        const tg = Math.round(lerp(colB[1], 255, fireHere * 0.3));
        const tb = Math.round(lerp(colB[2], 255, fireHere * 0.3));

        const g = ctx.createLinearGradient(pA.x, pA.y, pB.x, pB.y);
        g.addColorStop(0, `rgba(${fr},${fg},${fb},${baseAlpha})`);
        g.addColorStop(1, `rgba(${tr},${tg},${tb},${baseAlpha})`);

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.quadraticCurveTo((pA.x+pB.x)/2, (pA.y+pB.y)/2 - 6, pB.x, pB.y);
        ctx.strokeStyle = g;
        ctx.lineWidth = lerp(1, 2.5, fireHere);
        ctx.stroke();
      });
    }

    /* ── Draw nodes ── */
    sortedIndices.forEach(i => {
      const p = projected[i];
      const n = nodes[i];
      const breath = Math.sin(now * n.breathSpeed + n.breathPhase);

      // Fire activation for this layer
      const layerProgress = n.layerIdx / (CNN.length - 1);
      const fireHere = clamp((fireT - layerProgress * 0.8) / 0.3, 0, 1);

      // Color blending
      const baseR = lerp(COLORS.node[0], n.col[0], morph);
      const baseG = lerp(COLORS.node[1], n.col[1], morph);
      const baseB = lerp(COLORS.node[2], n.col[2], morph);
      const cr = Math.round(lerp(baseR, 255, fireHere * 0.5));
      const cg = Math.round(lerp(baseG, 255, fireHere * 0.5));
      const cb = Math.round(lerp(baseB, 255, fireHere * 0.5));

      const alpha = p.depthAlpha * lerp(0.8, 1, morph);  // much brighter base
      const r = p.r;

      // Outer glow — big and bright like a light turned on
      const glowSize = r * lerp(4, 8, fireHere);
      const glowAlpha = lerp(0.25, 0.6, fireHere) * alpha;
      if (glowAlpha > 0.005) {
        const grd = ctx.createRadialGradient(p.x, p.y, r*0.3, p.x, p.y, glowSize);
        grd.addColorStop(0, `rgba(${cr},${cg},${cb},${glowAlpha})`);
        grd.addColorStop(0.5, `rgba(${cr},${cg},${cb},${glowAlpha * 0.2})`);
        grd.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
        ctx.beginPath(); ctx.arc(p.x, p.y, glowSize, 0, Math.PI*2);
        ctx.fillStyle = grd; ctx.fill();
      }

      // Core
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha})`;
      ctx.fill();

      // Hot center — always glowing like a light bulb
      {
        const cAlpha = lerp(0.4, 0.85, fireHere) * alpha * (1 + breath * 0.2);
        ctx.beginPath(); ctx.arc(p.x, p.y, r * 0.4, 0, Math.PI*2);
        ctx.fillStyle = `rgba(255,255,255,${cAlpha})`;
        ctx.fill();
      }
    });

    /* ── Particles (CNN mode only) ── */
    if (morph > 0.5 && fireT > 0) {
      particles.forEach(p => {
        p.progress += p.speed;
        if (p.progress > 1) {
          p.progress = 0;
          p.connIdx = Math.floor(Math.random() * cnnConns.length);
        }
        const c = cnnConns[p.connIdx]; if (!c) return;
        const pA = projected[c.from], pB = projected[c.to];

        const layerP = nodes[c.from].layerIdx / (CNN.length - 1);
        const fireHere = clamp((fireT - layerP * 0.8) / 0.3, 0, 1);
        if (fireHere < 0.1) return;

        const px = lerp(pA.x, pB.x, p.progress);
        const py = lerp(pA.y, pB.y, p.progress);
        const fade = Math.sin(p.progress * Math.PI) * fireHere * 0.7;
        const col = nodes[c.from].col;

        ctx.beginPath(); ctx.arc(px, py, p.size * 2.5, 0, Math.PI*2);
        ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${fade * 0.1})`;
        ctx.fill();
        ctx.beginPath(); ctx.arc(px, py, p.size, 0, Math.PI*2);
        ctx.fillStyle = `rgba(255,255,255,${fade * 0.8})`;
        ctx.fill();
      });
    }

    /* ── CNN Layer labels (appear during morph) ── */
    if (morph > 0.3) {
      ctx.textAlign = "center";
      ctx.font = "600 9px 'Roboto Mono', monospace";
      const padX = 60, uW = w - padX * 2;
      CNN.forEach((layer, li) => {
        const layerP = li / (CNN.length - 1);
        const fireHere = clamp((fireT - layerP * 0.8) / 0.3, 0, 1);
        const alpha = lerp(0, 0.5, morph) + fireHere * 0.3;
        const col = layer.col;
        const cr = Math.round(lerp(col[0], 255, fireHere * 0.3));
        const cg = Math.round(lerp(col[1], 255, fireHere * 0.3));
        const cb = Math.round(lerp(col[2], 255, fireHere * 0.3));
        ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha * morph})`;
        ctx.fillText(layer.label, padX + layer.x * uW, h - 50);
      });
    }

    /* ── Progress bar ── */
    const barY = h - 2;
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(0, barY, w, 2);

    // Phase color
    const phaseCol = t < 0.35 ? COLORS.glow1 :
                     t < 0.72 ? COLORS.glow2 :
                     COLORS.output;
    const pg = ctx.createLinearGradient(0, barY, w * t, barY);
    pg.addColorStop(0, `rgba(${phaseCol[0]},${phaseCol[1]},${phaseCol[2]},0.1)`);
    pg.addColorStop(1, `rgba(${phaseCol[0]},${phaseCol[1]},${phaseCol[2]},0.7)`);
    ctx.fillStyle = pg;
    ctx.fillRect(0, barY, w * t, 2);
    if (t > 0.005) {
      ctx.beginPath(); ctx.arc(w * t, barY, 3, 0, Math.PI*2);
      ctx.fillStyle = `rgba(${phaseCol[0]},${phaseCol[1]},${phaseCol[2]},0.9)`;
      ctx.fill();
    }

    // Phase labels
    ctx.font = "500 7px 'Roboto Mono', monospace";
    ctx.textAlign = "center";
    const phases = [
      { name: "ROTATE", at: 0.17, col: COLORS.glow1 },
      { name: "UNFOLD", at: 0.52, col: COLORS.glow2 },
      { name: "FIRE",   at: 0.86, col: COLORS.output },
    ];
    phases.forEach(ph => {
      const on = t >= (ph.at - 0.15);
      ctx.fillStyle = `rgba(${ph.col[0]},${ph.col[1]},${ph.col[2]},${on ? 0.4 : 0.08})`;
      ctx.fillText(ph.name, ph.at * w, h - 10);
    });

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  /* ── Text overlays ── */
  const titleOpacity = useTransform(scrollY, [0, 250], [1, 0]);
  const subtitleOpacity = useTransform(scrollY, [30, 200], [1, 0]);
  const headerOpacity = useTransform(scrollY, [100, 350, 5000], [0, 1, 1]);
  const headerY = useTransform(scrollY, [100, 350], [12, 0]);

  const vh = typeof window !== "undefined" ? window.innerHeight : 800;
  const answerOpacity = useTransform(scrollY, [vh * 3.5, vh * 3.8], [0, 1]);
  const answerScale = useTransform(scrollY, [vh * 3.5, vh * 3.8], [0.85, 1]);
  const scrollHintOpacity = useTransform(scrollY, [0, 100], [1, 0]);

  return (
    <div ref={containerRef} className="relative" style={{ height: "500vh", background: "#030712" }}>
      <div className="sticky top-0 left-0 w-full" style={{ height: "100vh" }}>
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ display: "block" }} />

        {/* Title */}
        <motion.div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10"
          style={{ opacity: titleOpacity }}
        >
          <motion.h1
            className="text-white text-center text-3xl md:text-5xl lg:text-[3.5rem] font-extralight leading-tight tracking-wide"
            initial={{ opacity: 0, y: 25, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            style={{ filter: "drop-shadow(0 0 40px rgba(255,213,79,0.35))" }}
          >
            <span className="font-bold" style={{
              background: "linear-gradient(135deg, #ffffff 0%, #ffe082 20%, #ffcc02 45%, #ffab40 65%, #ff8a65 85%, #ea80fc 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              Encrypted Neural Networks
            </span>
          </motion.h1>
          <motion.p
            className="text-center mt-4 text-xs md:text-sm font-light tracking-widest uppercase"
            style={{ color: "rgba(255,255,255,0.2)", opacity: subtitleOpacity }}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.7 }}
          >
            Scroll to unfold the network
          </motion.p>
        </motion.div>

        {/* Header */}
        <motion.header
          className="absolute top-0 left-0 right-0 z-20 pointer-events-none"
          style={{ opacity: headerOpacity, y: headerY }}
        >
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center gap-2.5">
              <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#818cf8", boxShadow: "0 0 10px #818cf8" }} />
              <span className="text-[10px] font-semibold tracking-[0.25em] uppercase" style={{ color: "rgba(255,255,255,0.45)" }}>
                HE Benchmark
              </span>
            </div>
            <div className="flex items-center gap-4 text-[9px] tracking-wider uppercase" style={{ color: "rgba(255,255,255,0.18)" }}>
              <span>BFV</span>
              <span style={{color:"rgba(255,255,255,0.06)"}}>·</span>
              <span>OpenFHE</span>
              <span style={{color:"rgba(255,255,255,0.06)"}}>·</span>
              <span>SEAL</span>
              <span style={{color:"rgba(255,255,255,0.06)"}}>·</span>
              <span>HElib</span>
            </div>
          </div>
        </motion.header>

        {/* Output answer */}
        <motion.div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10"
          style={{ opacity: answerOpacity, scale: answerScale }}
        >
          <p className="text-[9px] uppercase tracking-[0.35em] mb-3 font-medium"
            style={{ color: `rgba(${COLORS.output[0]},${COLORS.output[1]},${COLORS.output[2]},0.7)` }}>
            Signal Propagated
          </p>
          <p className="text-7xl md:text-9xl font-black tabular-nums" style={{
            background: "linear-gradient(180deg, #5eead4 0%, #60a5fa 100%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            filter: "drop-shadow(0 0 50px rgba(94,234,212,0.25))",
          }}>7</p>
          <p className="text-xs mt-3 font-light tracking-wide" style={{ color: "rgba(255,255,255,0.18)" }}>
            Predicted digit · Fully encrypted CNN inference
          </p>
        </motion.div>

        {/* Scroll hint */}
        <motion.div
          className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-10"
          style={{ opacity: scrollHintOpacity }}
        >
          <span className="text-[8px] uppercase tracking-[0.3em]" style={{ color: "rgba(255,255,255,0.12)" }}>Scroll</span>
          <motion.div animate={{ y: [0, 5, 0] }} transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(129,140,248,0.25)" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7-7-7" />
            </svg>
          </motion.div>
        </motion.div>
      </div>

      {/* Bottom transition */}
      <div className="absolute bottom-0 left-0 right-0 h-48 pointer-events-none z-20"
        style={{ background: "linear-gradient(to bottom, transparent 0%, #030712 40%, #f7f7f7 100%)" }} />
    </div>
  );
}
