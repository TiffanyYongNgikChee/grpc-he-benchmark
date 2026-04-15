import { useState, useEffect, useRef } from "react";

// ─── Pixel font + keyframe animations (injected once) ────────────────────────
const PIXEL_STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
  @keyframes cloudDriftL { from{transform:translateX(-140%)} to{transform:translateX(0%)} }
  @keyframes cloudDriftR { from{transform:translateX(140%)}  to{transform:translateX(0%)} }
  @keyframes boxRise     { from{opacity:0;transform:translateY(36px)} to{opacity:1;transform:translateY(0)} }
  @keyframes owlBob      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
  @keyframes cursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
  .owl-bob  { animation: owlBob 1.5s ease-in-out infinite; }
  .px-cursor{ animation: cursorBlink 0.7s step-end infinite; }
`;

// ─── Stardew-inspired palette ─────────────────────────────────────────────────
const C = {
  sky:        "#78c8e0",
  skyBot:     "#4fa8c8",
  cloud:      "#fff9f2",
  cloudEdge:  "#d8c090",
  box:        "#1e0f04",
  boxBorder:  "#f0c030",
  boxInner:   "#120a02",
  cream:      "#fff4d0",
  dim:        "#b09060",
  accent:     "#f0c030",
  btnDark:    "#3a2008",
  btnGold:    "#f0c030",
  libBlue:    "#4285f4",
  libPurple:  "#9333ea",
  libGreen:   "#10b981",
};

// ─── Pixel Owl SVG ────────────────────────────────────────────────────────────
function PixelOwl({ frame }) {
  // frame: 0=open 1=half 2=blink
  const EyeL = () => {
    if (frame === 2) return <rect x="16" y="24" width="12" height="3" fill="#0a0502"/>;
    if (frame === 1) return (<><rect x="16" y="19" width="12" height="9" fill="white"/>
      <rect x="16" y="24" width="12" height="4" fill="#0a0502"/></>);
    return (<><rect x="16" y="18" width="12" height="12" fill="white"/>
      <rect x="18" y="20" width="8" height="8" fill="#0a0502"/>
      <rect x="19" y="21" width="2" height="2" fill="white"/></>);
  };
  const EyeR = () => {
    if (frame === 2) return <rect x="36" y="24" width="12" height="3" fill="#0a0502"/>;
    if (frame === 1) return (<><rect x="36" y="19" width="12" height="9" fill="white"/>
      <rect x="36" y="24" width="12" height="4" fill="#0a0502"/></>);
    return (<><rect x="36" y="18" width="12" height="12" fill="white"/>
      <rect x="38" y="20" width="8" height="8" fill="#0a0502"/>
      <rect x="39" y="21" width="2" height="2" fill="white"/></>);
  };

  return (
    <svg viewBox="0 0 64 84" width="156" height="205"
      style={{ imageRendering:"pixelated", display:"block" }}>
      {/* Ear tufts */}
      <rect x="10" y="0" width="8" height="14" fill="#c07808"/>
      <rect x="46" y="0" width="8" height="14" fill="#c07808"/>
      <rect x="12" y="2" width="4" height="10" fill="#e09018"/>
      <rect x="48" y="2" width="4" height="10" fill="#e09018"/>
      {/* Head */}
      <rect x="8"  y="10" width="48" height="32" fill="#f0c030"/>
      <rect x="4"  y="14" width="56" height="24" fill="#f0c030"/>
      <rect x="8"  y="10" width="48" height="5"  fill="#d8a020"/>
      {/* Eyes */}
      <EyeL/><EyeR/>
      {/* Beak */}
      <rect x="28" y="30" width="8" height="5" fill="#f07010"/>
      <rect x="30" y="35" width="4" height="4" fill="#f07010"/>
      {/* Blush */}
      <rect x="8"  y="27" width="6" height="4" fill="#ffaaaa" opacity="0.55"/>
      <rect x="50" y="27" width="6" height="4" fill="#ffaaaa" opacity="0.55"/>
      {/* Body */}
      <rect x="4"  y="42" width="56" height="30" fill="#e09018"/>
      {/* Belly */}
      <rect x="14" y="44" width="36" height="26" fill="#fce090"/>
      {/* Belly dots */}
      <rect x="20" y="50" width="4" height="4" fill="#f0c030" opacity="0.5"/>
      <rect x="32" y="54" width="4" height="4" fill="#f0c030" opacity="0.5"/>
      <rect x="26" y="60" width="4" height="4" fill="#f0c030" opacity="0.5"/>
      {/* Wings */}
      <rect x="0"  y="44" width="10" height="22" fill="#c07808"/>
      <rect x="54" y="44" width="10" height="22" fill="#c07808"/>
      {/* Feet */}
      <rect x="14" y="72" width="14" height="6"  fill="#f07010"/>
      <rect x="36" y="72" width="14" height="6"  fill="#f07010"/>
      <rect x="10" y="74" width="4"  height="4"  fill="#f07010"/>
      <rect x="50" y="74" width="4"  height="4"  fill="#f07010"/>
    </svg>
  );
}

// ─── Pixel Cloud SVG ──────────────────────────────────────────────────────────
function PixelCloud({ width = 220, flip = false }) {
  return (
    <svg viewBox="0 0 110 56" width={width}
      style={{ imageRendering:"pixelated", display:"block",
               transform: flip ? "scaleX(-1)" : "none" }}>
      <rect x="22" y="34" width="66" height="18" fill={C.cloud}/>
      <rect x="10" y="38" width="90" height="14" fill={C.cloud}/>
      <rect x="30" y="18" width="30" height="20" fill={C.cloud}/>
      <rect x="18" y="26" width="22" height="16" fill={C.cloud}/>
      <rect x="58" y="22" width="26" height="16" fill={C.cloud}/>
      {/* edge shadow */}
      <rect x="22" y="32" width="66" height="4" fill={C.cloudEdge} opacity="0.3"/>
      <rect x="30" y="16" width="30" height="4" fill={C.cloudEdge} opacity="0.3"/>
    </svg>
  );
}

// ─── Steps ────────────────────────────────────────────────────────────────────
const STEPS = [
  {
    speech: "Hi! I'm your encryption guide!",
    body:   "I'll walk you through what just happened. We benchmarked three HE libraries and measured exactly how fast each one is. Ready to learn?",
    table:  null,
  },
  {
    speech: "We are comparing three libraries: SEAL, HELib, OpenFHE",
    body:   "Each is a different implementation of Homomorphic Encryption — math that lets a server compute on encrypted data without ever seeing the original values!",
    table: {
      headers: ["Library", "Made by", "Scheme"],
      rows: [
        ["SEAL",    "Microsoft", "BFV"],
        ["HELib",   "IBM",       "BGV"],
        ["OpenFHE", "Community", "BFV"],
      ],
    },
  },
  {
    speech: "First: Key Generation (Setup)",
    body:   "Before encrypting, each library generates a key pair. You share the padlock (public key) but keep the key (secret key) yourself.",
    table: {
      headers: ["",         "SEAL",          "HELib",      "OpenFHE"],
      rows: [
        ["Scheme",   "BFV",           "BGV",        "BFV"],
        ["Ring n",   "4096",          "~4096",      "4096"],
        ["Security", "128-bit",       "~128-bit",   "128-bit"],
        ["Keys",     "Pub+Sec+Relin", "Pub+Sec",    "Pub+Sec+Relin"],
      ],
    },
  },
  {
    speech: "Notice SEAL and OpenFHE use BFV, HELib uses BGV",
    body:   "BFV puts the message in the high bits of the ciphertext. BGV puts it in the low bits and uses modulus switching to drain noise. Different math — different speeds!",
    table: {
      headers: ["Property",  "BFV (SEAL/OpenFHE)",     "BGV (HELib)"],
      rows: [
        ["Message",  "High bits: Δ·m + noise",  "Low bits: m + p·noise"],
        ["Noise",    "Scale factor Δ",           "Modulus switching"],
        ["Relin?",   "Yes",                      "Yes"],
      ],
    },
  },
  {
    speech: "Now look at encoding differences...",
    body:   "SEAL and OpenFHE pack 4096 numbers into one ciphertext using NTT. This is why their encrypt/decrypt speeds are not directly comparable to HELib!",
    table: {
      headers: ["",          "SEAL",         "HELib",     "OpenFHE"],
      rows: [
        ["Slots",    "4096 slots",   "1 slot",    "4096 slots"],
        ["Ciphers",  "10",           "10",        "10"],
        ["Encoding", "NTT batch",    "Single int","NTT batch"],
      ],
    },
  },
  {
    speech: "Multiplication uses fresh ciphertext pairs in all libraries",
    body:   "All three use ct[i] op ct[i+1] — fresh pairs every time. This makes the comparison fair! Multiplication is slowest because of relinearisation.",
    table: {
      headers: ["",          "SEAL",          "HELib",         "OpenFHE"],
      rows: [
        ["Add",      "ct[i]+ct[i+1]", "ct[i]+ct[i+1]", "ct[i]+ct[i+1]"],
        ["Multiply", "ct[i]×ct[i+1]", "ct[i]×ct[i+1]", "ct[i]×ct[i+1]"],
        ["Relin?",   "Yes",           "Yes",            "Yes"],
      ],
    },
  },
  {
    speech: "Now let's look at the results!",
    body:   "Below you will see total time bars, a speed radar, and per-operation breakdowns. Click any row to see the full math explanation!",
    table:  null,
  },
];

// ─── Pixel table ──────────────────────────────────────────────────────────────
function PixelTable({ table }) {
  if (!table) return null;
  const libColors = [C.libBlue, C.libPurple, C.libGreen];
  return (
    <div style={{ overflowX:"auto", marginTop:10 }}>
      <table style={{ borderCollapse:"collapse", width:"100%" }}>
        <thead>
          <tr>
            {table.headers.map((h, i) => (
              <th key={i} style={{
                padding:"7px 10px",
                background: i === 0 ? "#0e0702" : (libColors[i-1]||C.accent)+"28",
                color: i === 0 ? C.dim : (libColors[i-1]||C.accent),
                borderBottom:`3px solid ${i===0 ? C.dim : (libColors[i-1]||C.accent)}`,
                textAlign: i === 0 ? "left" : "center",
                whiteSpace:"nowrap",
                fontFamily:"'Press Start 2P',monospace",
                fontSize: 9,
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri%2===0 ? "#0e0702" : "#1e0f04" }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{
                  padding:"7px 10px",
                  color: ci===0 ? C.dim : (libColors[ci-1]||C.cream),
                  textAlign: ci===0 ? "left" : "center",
                  borderBottom:"1px solid #301a08",
                  whiteSpace:"nowrap",
                  fontWeight: ci===0 ? "normal" : "600",
                  fontFamily:"system-ui,-apple-system,sans-serif",
                  fontSize: 12,
                }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function OwlGuide({ onDone }) {
  const [step, setStep]       = useState(0);
  const [phase, setPhase]     = useState("clouds"); // clouds→box→typing→done
  const [displayed, setDisplayed] = useState("");
  const [showBody, setShowBody]   = useState(false);
  const [owlFrame, setOwlFrame]   = useState(0);
  const typingRef = useRef(null);
  const current = STEPS[step];
  const isLast  = step === STEPS.length - 1;

  // Blink loop
  useEffect(() => {
    const id = setInterval(() => {
      setOwlFrame(1);
      setTimeout(() => setOwlFrame(2), 90);
      setTimeout(() => setOwlFrame(1), 180);
      setTimeout(() => setOwlFrame(0), 270);
    }, 3000);
    return () => clearInterval(id);
  }, []);

  // Cloud phase → show box after 900ms
  useEffect(() => {
    if (phase !== "clouds") return;
    const t = setTimeout(() => setPhase("box"), 900);
    return () => clearTimeout(t);
  }, [phase]);

  // Box phase → start typing after 500ms
  useEffect(() => {
    if (phase !== "box") return;
    const t = setTimeout(() => startTyping(), 200);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, step]);

  const startTyping = () => {
    clearInterval(typingRef.current);
    setPhase("typing");
    setDisplayed("");
    setShowBody(false);
    const text = current.speech;
    let i = 0;
    typingRef.current = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(typingRef.current);
        setTimeout(() => { setShowBody(true); setPhase("done"); }, 200);
      }
    }, 18);
  };

  // Step change → re-enter box phase to retype
  useEffect(() => {
    if (phase === "clouds") return;
    clearInterval(typingRef.current);
    setDisplayed("");
    setShowBody(false);
    setPhase("box");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  useEffect(() => () => clearInterval(typingRef.current), []);

  const handleNext = () => {
    if (isLast) { onDone(); return; }
    setStep(s => s + 1);
  };

  const btnBase = {
    fontFamily: "'Press Start 2P',monospace",
    cursor: "pointer",
    imageRendering: "pixelated",
  };

  return (
    <div style={{
      width:"100%",
      background:`linear-gradient(180deg,${C.sky} 0%,${C.skyBot} 100%)`,
      borderRadius:12,
      overflow:"hidden",
      position:"relative",
      minHeight: "min(90vh, 680px)",
      userSelect:"none",
    }}>
      <style>{PIXEL_STYLE}</style>

      {/* Sky pixel star dots */}
      {[[6,8],[38,6],[72,12],[88,9],[18,22],[55,5],[92,18]].map(([x,y],i)=>(
        <div key={i} style={{position:"absolute",left:`${x}%`,top:`${y}%`,
          width:4,height:4,background:"rgba(255,255,255,0.45)",imageRendering:"pixelated"}}/>
      ))}

      {/* Cloud left */}
      <div style={{position:"absolute",top:16,left:"2%",
        animation:"cloudDriftL 0.85s cubic-bezier(.22,1,.36,1) both",zIndex:2}}>
        <PixelCloud width={190}/>
      </div>
      {/* Cloud right */}
      <div style={{position:"absolute",top:6,right:"2%",
        animation:"cloudDriftR 0.85s cubic-bezier(.22,1,.36,1) both",zIndex:2}}>
        <PixelCloud width={210} flip/>
      </div>

      {/* Owl centre */}
      <div style={{display:"flex",justifyContent:"center",paddingTop:62,position:"relative",zIndex:3}}>
        <div className="owl-bob">
          <PixelOwl frame={owlFrame}/>
        </div>
      </div>

      {/* Dialogue box */}
      {(phase==="box"||phase==="typing"||phase==="done") && (
        <div style={{
          margin:"14px 14px 16px",
          background:C.box,
          border:`4px solid ${C.boxBorder}`,
          boxShadow:`4px 4px 0 #000,0 0 0 2px #000 inset`,
          borderRadius:4,
          animation:"boxRise 0.45s cubic-bezier(.22,1,.36,1) both",
          position:"relative",
          zIndex:10,
        }}>
          {/* Inner inset */}
          <div style={{margin:4,border:"2px solid #4a2e0a",borderRadius:2,
                       padding:"13px 17px 12px",background:C.boxInner}}>

            {/* Step pips */}
            <div style={{display:"flex",gap:6,marginBottom:12}}>
              {STEPS.map((_,i)=>(
                <div key={i} style={{
                  width:i===step?18:6, height:6,
                  background:i===step?C.accent:i<step?"#c08020":"#301a08",
                  transition:"all 0.2s",
                }}/>
              ))}
            </div>

            {/* Speech (typewriter) — pixel font, large, gold */}
            <div style={{fontSize:15,color:C.accent,lineHeight:1.8,
                         marginBottom:12,minHeight:28,letterSpacing:"0.3px",
                         fontFamily:"'Press Start 2P',monospace"}}>
              {displayed}
              {phase==="typing" && (
                <span className="px-cursor" style={{
                  display:"inline-block",width:9,height:14,
                  background:C.accent,marginLeft:3,verticalAlign:"middle"
                }}/>
              )}
            </div>

            {/* Body — normal readable font, large */}
            {showBody && (
              <div style={{fontSize:13,color:C.cream,lineHeight:1.75,
                           fontFamily:"system-ui,-apple-system,sans-serif",
                           marginBottom:10,transition:"opacity 0.3s"}}>
                {current.body}
              </div>
            )}

            {/* Table */}
            {showBody && <PixelTable table={current.table}/>}

            {/* Buttons */}
            {phase==="done" && (
              <div style={{display:"flex",justifyContent:"space-between",
                           alignItems:"center",marginTop:14,gap:8}}>
                <button onClick={onDone} style={{...btnBase,
                  fontSize:9,color:C.dim,background:"transparent",
                  border:"none",textDecoration:"underline",padding:0}}>
                  skip tour
                </button>
                <div style={{display:"flex",gap:8}}>
                  {step>0 && (
                    <button onClick={()=>setStep(s=>s-1)} style={{...btnBase,
                      fontSize:9,color:C.cream,background:C.btnDark,
                      border:`3px solid ${C.boxBorder}`,
                      boxShadow:"3px 3px 0 #000",padding:"8px 14px"}}
                      onMouseDown={e=>e.currentTarget.style.transform="translate(2px,2px)"}
                      onMouseUp={e=>e.currentTarget.style.transform=""}>
                      ◀ BACK
                    </button>
                  )}
                  <button onClick={handleNext} style={{...btnBase,
                    fontSize:9,color:"#0a0502",background:C.accent,
                    border:"3px solid #b08010",
                    boxShadow:"3px 3px 0 #000",padding:"8px 18px"}}
                    onMouseDown={e=>e.currentTarget.style.transform="translate(2px,2px)"}
                    onMouseUp={e=>e.currentTarget.style.transform=""}>
                    {isLast?"▶ RESULTS!":"NEXT ▶"}
                  </button>
                </div>
              </div>
            )}

            {/* "wait..." while typing */}
            {phase==="typing" && (
              <div style={{fontSize:9,color:"#4a2e0a",marginTop:8,textAlign:"right",
                           fontFamily:"'Press Start 2P',monospace"}}>
                wait...
              </div>
            )}

            {/* Step counter */}
            <div style={{fontSize:9,color:"#3d2410",marginTop:phase==="done"?5:8,textAlign:"right",
                         fontFamily:"'Press Start 2P',monospace"}}>
              {step+1} / {STEPS.length}
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
