import { LAYERS, CATEGORY_COLORS } from "./CnnPipeline";

/**
 * MetricsStrip — Bottom bar with per-layer timing breakdown.
 * Light theme stacked-bar matching TF Playground's minimal aesthetic.
 */
export default function MetricsStrip({ result }) {
  if (!result) {
    return (
      <div className="text-center text-xs py-1" style={{ color: "#bbb" }}>
        Run inference to see per-layer timing breakdown
      </div>
    );
  }

  const timedLayers = LAYERS.filter((l) => l.key && result[l.key] != null);
  const total = result.totalMs || 1;

  return (
    <div>
      {/* Stacked bar */}
      <div className="flex rounded-md overflow-hidden h-6 mb-2" style={{ border: "1px solid #d9d9d9" }}>
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const pct = (ms / total) * 100;
          const cat = CATEGORY_COLORS[layer.category];

          return (
            <div
              key={layer.id}
              className="relative group flex items-center justify-center text-[9px] font-medium transition-opacity hover:opacity-80 cursor-default"
              style={{
                width: `${Math.max(pct, 1.5)}%`,
                backgroundColor: cat.active,
                color: "#fff",
              }}
              title={`${layer.label}: ${ms.toFixed(2)}ms (${pct.toFixed(1)}%)`}
            >
              {pct > 5 && <span className="truncate px-1">{layer.label}</span>}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px]" style={{ color: "#999" }}>
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const cat = CATEGORY_COLORS[layer.category];
          return (
            <span key={layer.id} className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-sm"
                style={{ backgroundColor: cat.active }}
              />
              <span style={{ color: "#888" }}>{layer.label}</span>
              <span className="font-mono" style={{ color: "#555" }}>{ms.toFixed(1)}ms</span>
            </span>
          );
        })}
        <span className="ml-auto font-medium font-mono" style={{ color: "#f4743a" }}>
          Total: {total.toFixed(1)}ms
        </span>
      </div>
    </div>
  );
}
