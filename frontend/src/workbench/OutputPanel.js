import CountUp from "../components/CountUp";

/**
 * OutputPanel — Right column showing prediction results.
 * Dark Stardew night theme — big digit, clear timing, prominent logit chart.
 */
export default function OutputPanel({ result, error, loading, pixels, layerStatus, elapsedMs }) {
  /* Empty state */
  if (!result && !error && !loading) {
    return (
      <div style={{
        display:"flex", flexDirection:"column", alignItems:"center",
        justifyContent:"center", height:"100%", textAlign:"center", gap:12,
      }}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
          stroke="rgba(240,192,48,0.22)" strokeWidth="1.5">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 12h8M12 8v8" strokeLinecap="round" />
        </svg>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.7rem", letterSpacing:"0.14em",
          color:"#5a4800", lineHeight:1.9,
        }}>
          {pixels ? "Press RUN to start\nencrypted inference" : "Draw a digit\non the left"}
        </p>
      </div>
    );
  }

  /* Loading — show progress */
  if (loading) {
    const pipelineLayers = [
      "encrypt","conv1","bias1","relu1","pool1",
      "conv2","bias2","relu2","pool2",
      "fc","biasfc","decrypt",
    ];
    const doneCount = layerStatus
      ? pipelineLayers.filter((id) => layerStatus[id] === "done").length
      : 0;
    const currentLayer = layerStatus
      ? pipelineLayers.find((id) => layerStatus[id] === "processing")
      : null;
    const pct = Math.round((doneCount / pipelineLayers.length) * 100);
    const elapsed = elapsedMs ? (elapsedMs / 1000).toFixed(1) : "0.0";

    return (
      <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:"100%", gap:16 }}>
        {/* Spinner */}
        <svg className="animate-spin" style={{ width:36, height:36, color:"#f0c030" }} viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>

        {/* Elapsed */}
        <div style={{ textAlign:"center" }}>
          <p style={{
            fontFamily:"'Press Start 2P', monospace",
            fontSize:"1.4rem", letterSpacing:"0.06em",
            color:"#f0c030", textShadow:"0 0 16px rgba(240,192,48,0.5)",
            margin:0,
          }}>{elapsed}s</p>
          <p style={{
            fontFamily:"'Press Start 2P', monospace",
            fontSize:"0.35rem", letterSpacing:"0.2em",
            color:"rgba(240,192,48,0.4)", textTransform:"uppercase",
            margin:"4px 0 0",
          }}>elapsed</p>
        </div>

        {/* Current layer label */}
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.65rem", letterSpacing:"0.14em",
          color:"#c890f0", margin:0,
        }}>
          {currentLayer ? currentLayer.toUpperCase() : "ENCRYPTING…"}
        </p>

        {/* Progress bar */}
        <div style={{
          width:"80%", height:8, borderRadius:4, overflow:"hidden",
          background:"rgba(255,248,220,0.07)",
          border:"1px solid rgba(240,192,48,0.15)",
        }}>
          <div style={{
            height:"100%", borderRadius:4,
            width:`${pct}%`,
            background:"linear-gradient(90deg, #f0c030, #58c896)",
            transition:"width 0.5s ease",
            boxShadow:"0 0 8px rgba(88,200,150,0.4)",
          }} />
        </div>

        {/* Layer count */}
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.55rem", letterSpacing:"0.12em",
          color:"#555", margin:0,
        }}>
          {doneCount}/{pipelineLayers.length} layers
        </p>
      </div>
    );
  }

  /* Error */
  if (error) {
    return (
      <div style={{
        borderRadius:6, padding:"14px 16px",
        background:"rgba(224,90,90,0.08)",
        border:"1.5px solid rgba(224,90,90,0.3)",
      }}>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.42rem", letterSpacing:"0.14em",
          color:"#e05a5a", marginBottom:8,
        }}>Error</p>
        <p style={{ fontFamily:"system-ui,sans-serif", fontSize:"0.72rem", color:"rgba(224,90,90,0.8)" }}>{error}</p>
        <p style={{ fontFamily:"system-ui,sans-serif", fontSize:"0.68rem", color:"rgba(224,90,90,0.5)", marginTop:6 }}>
          Make sure Docker + Spring Boot are running.
        </p>
      </div>
    );
  }

  /* Result */
  const logits = result.logits || [];
  const predicted = result.predictedDigit;

  /* Timing breakdown rows */
  const timingRows = [
    { label:"Encrypt",      key:"encryptionMs",  color:"#6aabf7" },
    { label:"Conv1",        key:"conv1Ms",        color:"#c890f0" },
    { label:"Conv2",        key:"conv2Ms",        color:"#c890f0" },
    { label:"Activations",  key:"act1Ms",         color:"#f0c030" },
    { label:"Pooling",      key:"pool1Ms",        color:"#f0c030" },
    { label:"FC",           key:"fcMs",           color:"#e05a5a" },
    { label:"Decrypt",      key:"decryptionMs",   color:"#58c896" },
  ].filter(r => result[r.key] != null && result[r.key] > 0);

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>

      {/* ── BIG PREDICTED DIGIT ── */}
      <div style={{
        textAlign:"center", padding:"20px 12px",
        background:"rgba(0,0,0,0.06)",
        border:"1.5px solid rgba(0,0,0,0.15)",
        borderRadius:8,
        boxShadow:"inset 0 1px 2px rgba(0,0,0,0.08)",
      }}>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.65rem", letterSpacing:"0.22em",
          color:"#5a4400", textTransform:"uppercase",
          margin:"0 0 8px",
        }}>Predicted Digit</p>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"4.5rem", lineHeight:1,
          color:"#f0c030",
          textShadow:"0 0 40px rgba(240,192,48,0.6), 0 0 80px rgba(240,192,48,0.2)",
          margin:0,
        }}>
          <CountUp end={predicted} duration={400} decimals={0} />
        </p>
        {/* Status badge */}
        <div style={{ marginTop:10 }}>
          <span style={{
            display:"inline-flex", alignItems:"center", gap:6,
            padding:"4px 12px", borderRadius:20,
            background: result.status === "success"
              ? "rgba(88,200,150,0.12)" : "rgba(224,90,90,0.12)",
            border: `1px solid ${result.status === "success" ? "rgba(88,200,150,0.35)" : "rgba(224,90,90,0.35)"}`,
          }}>
            <span style={{
              width:6, height:6, borderRadius:"50%",
              background: result.status === "success" ? "#58c896" : "#e05a5a",
              boxShadow: result.status === "success"
                ? "0 0 6px rgba(88,200,150,0.6)" : "0 0 6px rgba(224,90,90,0.6)",
            }} />
            <span style={{
              fontFamily:"'Press Start 2P', monospace",
              fontSize:"0.55rem", letterSpacing:"0.16em",
              color: result.status === "success" ? "#58c896" : "#e05a5a",
              textTransform:"uppercase",
            }}>{result.status}</span>
          </span>
        </div>
      </div>

      {/* ── TOTAL TIME — hero display ── */}
      <div style={{
        textAlign:"center", padding:"20px 12px",
        background:"rgba(10,80,40,0.1)",
        border:"1.5px solid rgba(10,80,40,0.25)",
        borderRadius:8,
      }}>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.65rem", letterSpacing:"0.22em",
          color:"#155530", textTransform:"uppercase",
          margin:"0 0 10px",
        }}>Total Inference Time</p>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"3.2rem", lineHeight:1,
          color:"#0d6e3a",
          textShadow:"0 0 20px rgba(13,110,58,0.25)",
          margin:"0 0 8px",
        }}>
          <CountUp end={result.totalMs} duration={800} decimals={0} suffix="ms" />
        </p>
        <p style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"1rem",
          color:"#2a7a50", margin:0,
          letterSpacing:"0.06em",
        }}>
          ≈ {(result.totalMs / 1000).toFixed(2)}s
        </p>
      </div>

      {/* ── PER-PHASE TIMING ── */}
      {timingRows.length > 0 && (
        <div style={{
          background:"rgba(0,0,0,0.06)",
          border:"1px solid rgba(0,0,0,0.14)",
          borderRadius:8, padding:"12px 14px",
        }}>
          <p style={{
            fontFamily:"'Press Start 2P', monospace",
            fontSize:"0.6rem", letterSpacing:"0.2em",
            color:"#5a4400", textTransform:"uppercase",
            margin:"0 0 12px",
          }}>Phase Breakdown</p>
          <div style={{ display:"flex", flexDirection:"column", gap:9 }}>
            {timingRows.map(({ label, key, color }) => {
              const ms = result[key] || 0;
              const pct = (ms / result.totalMs) * 100;
              return (
                <div key={key} style={{ display:"flex", flexDirection:"column", gap:4 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                    <span style={{
                      fontFamily:"system-ui,sans-serif",
                      fontSize:"0.9rem", fontWeight:600, color:"#2a2a2a",
                    }}>{label}</span>
                    <span style={{
                      fontFamily:"'Press Start 2P', monospace",
                      fontSize:"0.6rem", letterSpacing:"0.08em",
                      color,
                    }}>{ms.toFixed(1)}ms
                      <span style={{ color:"#555", marginLeft:6 }}>
                        {pct.toFixed(0)}%
                      </span>
                    </span>
                  </div>
                  <div style={{
                    height:6, borderRadius:3, overflow:"hidden",
                    background:"rgba(0,0,0,0.12)",
                  }}>
                    <div style={{
                      height:"100%", borderRadius:3,
                      width:`${Math.max(pct,0.5)}%`,
                      background:color,
                      boxShadow:`0 0 6px ${color}66`,
                      transition:"width 0.8s ease",
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── LOGIT BAR CHART — prominent FHE output visualisation ── */}
      {logits.length > 0 && (() => {
        const maxLogit = Math.max(...logits);
        const minLogit = Math.min(...logits);
        const range = Math.max(maxLogit - minLogit, 1);
        // confidence = winner share of all positive logits
        const sumPos = logits.reduce((s, v) => s + Math.max(0, v), 0);
        const confidence = sumPos > 0
          ? ((Math.max(0, logits[predicted]) / sumPos) * 100).toFixed(1)
          : "—";

        return (
          <div style={{
            background:"rgba(0,0,0,0.06)",
            border:"1.5px solid rgba(0,0,0,0.16)",
            borderRadius:10,
            padding:"16px 14px 12px",
          }}>
            {/* Header row */}
            <div style={{
              display:"flex", alignItems:"baseline",
              justifyContent:"space-between", marginBottom:14,
            }}>
              <div>
                <p style={{
                  fontFamily:"'Press Start 2P', monospace",
                  fontSize:"0.7rem", letterSpacing:"0.2em",
                  color:"#5a4400", textTransform:"uppercase",
                  margin:0,
                }}>FHE Output Logits</p>
                <p style={{
                  fontFamily:"system-ui,sans-serif",
                  fontSize:"0.85rem",
                  color:"#555",
                  margin:"4px 0 0",
                }}>Raw decrypted scores — not a label, real numeric data</p>
              </div>
              {/* Confidence badge */}
              <div style={{
                textAlign:"right", flexShrink:0, marginLeft:10,
              }}>
                <p style={{
                  fontFamily:"'Press Start 2P', monospace",
                  fontSize:"0.6rem", letterSpacing:"0.14em",
                  color:"#1a4a7a", textTransform:"uppercase",
                  margin:"0 0 4px",
                }}>Confidence</p>
                <p style={{
                  fontFamily:"'Press Start 2P', monospace",
                  fontSize:"1.6rem", letterSpacing:"0.06em",
                  color:"#1a5aaa",
                  margin:0,
                }}>
                  <CountUp end={parseFloat(confidence)} duration={800} decimals={1} suffix="%" />
                </p>
              </div>
            </div>

            {/* Bars — one per digit 0–9 */}
            <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
              {logits.map((val, digit) => {
                const isWinner = digit === predicted;
                // bar fill = normalised 0→100% of the positive range; negatives get a tiny stub
                const fillPct = Math.max(
                  ((val - minLogit) / range) * 100,
                  1
                );
                const barBg    = isWinner ? "rgba(200,144,0,0.12)" : "rgba(34,85,170,0.07)";
                const barBorder= isWinner ? "rgba(200,144,0,0.35)" : "rgba(34,85,170,0.18)";

                return (
                  <div key={digit} style={{
                    display:"flex", alignItems:"center", gap:8,
                    padding:"5px 8px",
                    borderRadius:5,
                    background: isWinner ? "rgba(200,144,0,0.08)" : "transparent",
                    border: `1px solid ${isWinner ? "rgba(200,144,0,0.25)" : "transparent"}`,
                  }}>
                    {/* Digit label */}
                    <span style={{
                      fontFamily:"'Press Start 2P', monospace",
                      fontSize: isWinner ? "1.1rem" : "0.8rem",
                      color: isWinner ? "#c89000" : "#444",
                      width:26, textAlign:"center", flexShrink:0,
                      textShadow: isWinner ? "0 0 8px rgba(200,144,0,0.4)" : "none",
                      transition:"all 0.4s",
                    }}>{digit}</span>

                    {/* Bar track */}
                    <div style={{
                      flex:1, height: isWinner ? 24 : 16,
                      borderRadius:3, overflow:"hidden",
                      background: barBg,
                      border:`1px solid ${barBorder}`,
                      transition:"height 0.4s",
                    }}>
                      <div style={{
                        height:"100%", borderRadius:3,
                        width:`${fillPct}%`,
                        background: isWinner
                          ? "linear-gradient(90deg,#b07c00,#c89000,#d4a820)"
                          : `linear-gradient(90deg,rgba(34,85,170,0.5),rgba(34,85,170,0.7))`,
                        boxShadow: isWinner ? "0 0 10px rgba(240,192,48,0.5)" : "none",
                        transition:"width 0.9s cubic-bezier(0.16,1,0.3,1)",
                      }} />
                    </div>

                    {/* Raw logit value */}
                    <span style={{
                      fontFamily:"'Press Start 2P', monospace",
                      fontSize: isWinner ? "0.72rem" : "0.55rem",
                      color: isWinner ? "#c89000" : "#555",
                      width:60, textAlign:"right", flexShrink:0,
                      letterSpacing:"0.04em",
                      transition:"all 0.4s",
                    }}>{val}</span>

                    {/* Winner crown */}
                    {isWinner && (
                      <span style={{
                        fontSize:"1rem", flexShrink:0,
                        filter:"drop-shadow(0 0 4px rgba(240,192,48,0.8))",
                      }}>★</span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Footer note */}
            <p style={{
              fontFamily:"system-ui,sans-serif",
              fontSize:"0.8rem",
              color:"#666",
              margin:"14px 0 0",
              textAlign:"center",
              lineHeight:1.5,
            }}>
              Logits are integer-valued BFV ciphertext outputs — decrypted without ever revealing the input.
            </p>
          </div>
        );
      })()}

      {/* Float model accuracy */}
      {result.floatModelAccuracy != null && (
        <p style={{
          fontFamily:"system-ui,sans-serif",
          fontSize:"0.85rem", textAlign:"center",
          color:"#555", margin:0,
        }}>
          Float model accuracy:{" "}
          <span style={{ color:"#222", fontWeight:600 }}>{result.floatModelAccuracy}%</span>
        </p>
      )}
    </div>
  );
}
