import { LAYERS, CATEGORY_COLORS } from "./CnnPipeline";

/**
 * MetricsStrip — Per-layer timing breakdown.
 * Dark Stardew palette to match the workbench.
 */
export default function MetricsStrip({ result }) {
  if (!result) {
    return (
      <div style={{
        textAlign:"center", padding:"14px 0",
        fontFamily:"'Press Start 2P', monospace",
        fontSize:"0.36rem", letterSpacing:"0.18em",
        color:"rgba(255,248,220,0.2)",
      }}>
        Run inference to see per-layer timing breakdown
      </div>
    );
  }

  const timedLayers = LAYERS.filter((l) => l.key && result[l.key] != null);
  const total = result.totalMs || 1;

  return (
    <div style={{ padding:"12px 0 6px" }}>

      {/* Header row */}
      <div style={{
        display:"flex", justifyContent:"space-between",
        alignItems:"center", marginBottom:10,
      }}>
        <span style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.36rem", letterSpacing:"0.22em",
          color:"rgba(240,192,48,0.45)", textTransform:"uppercase",
        }}>Layer Timing</span>
        <span style={{
          fontFamily:"'Press Start 2P', monospace",
          fontSize:"0.55rem", letterSpacing:"0.08em",
          color:"#58c896",
          textShadow:"0 0 10px rgba(88,200,150,0.5)",
        }}>
          Total: {total.toFixed(1)}ms
        </span>
      </div>

      {/* Stacked bar — taller */}
      <div style={{
        display:"flex", borderRadius:6, overflow:"hidden",
        height:18, marginBottom:12,
        border:"1px solid rgba(240,192,48,0.12)",
        boxShadow:"0 0 14px rgba(0,0,0,0.3)",
      }}>
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const pct = (ms / total) * 100;
          const cat = CATEGORY_COLORS[layer.category];
          return (
            <div
              key={layer.id}
              title={`${layer.label}: ${ms.toFixed(2)}ms (${pct.toFixed(1)}%)`}
              style={{
                width:`${Math.max(pct,1.5)}%`,
                background:cat.active,
                display:"flex", alignItems:"center", justifyContent:"center",
                fontSize:"0.6rem", fontWeight:600, color:"rgba(0,0,0,0.75)",
                transition:"opacity 0.15s",
                cursor:"default",
                overflow:"hidden",
              }}
              onMouseEnter={e => e.currentTarget.style.opacity="0.75"}
              onMouseLeave={e => e.currentTarget.style.opacity="1"}
            >
              {pct > 6 && (
                <span style={{
                  fontFamily:"'Press Start 2P', monospace",
                  fontSize:"0.28rem", letterSpacing:"0.06em",
                  whiteSpace:"nowrap", overflow:"hidden",
                  paddingInline:4,
                }}>{layer.label}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend — readable tiles */}
      <div style={{ display:"flex", flexWrap:"wrap", gap:"8px 16px" }}>
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const pct = (ms / total) * 100;
          const cat = CATEGORY_COLORS[layer.category];
          return (
            <div key={layer.id} style={{
              display:"flex", alignItems:"center", gap:6,
            }}>
              <span style={{
                width:10, height:10, borderRadius:2, flexShrink:0,
                background:cat.active,
                boxShadow:`0 0 5px ${cat.active}88`,
              }} />
              <span style={{
                fontFamily:"system-ui,sans-serif",
                fontSize:"0.75rem",
                color:"rgba(255,248,220,0.5)",
              }}>{layer.label}</span>
              <span style={{
                fontFamily:"'Press Start 2P', monospace",
                fontSize:"0.38rem", letterSpacing:"0.06em",
                color:"rgba(255,248,220,0.8)",
              }}>{ms.toFixed(1)}ms</span>
              <span style={{
                fontFamily:"system-ui,sans-serif",
                fontSize:"0.65rem",
                color:"rgba(255,248,220,0.25)",
              }}>({pct.toFixed(1)}%)</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
