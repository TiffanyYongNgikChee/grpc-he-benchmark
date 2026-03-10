import { LAYERS, CATEGORY_COLORS } from "./CnnPipeline";

/**
 * MetricsStrip — Bottom bar showing per-layer timing breakdown in a
 * horizontal stacked-bar style, similar to a flame chart or TF Playground's
 * epoch/loss strip.
 */
export default function MetricsStrip({ result }) {
  if (!result) {
    return (
      <div className="text-center text-slate-600 text-xs py-1">
        Run inference to see per-layer timing breakdown
      </div>
    );
  }

  /* Collect layers that have timing data */
  const timedLayers = LAYERS.filter((l) => l.key && result[l.key] != null);
  const total = result.totalMs || 1;

  return (
    <div>
      {/* Horizontal stacked bar */}
      <div className="flex rounded-md overflow-hidden h-6 mb-2">
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const pct = (ms / total) * 100;
          const cat = CATEGORY_COLORS[layer.category];

          return (
            <div
              key={layer.id}
              className="relative group flex items-center justify-center text-[9px] font-medium
                         transition-opacity hover:opacity-90 cursor-default"
              style={{
                width: `${Math.max(pct, 1.5)}%`,
                backgroundColor: cat.active,
                color: "#0f172a",
              }}
              title={`${layer.label}: ${ms.toFixed(2)}ms (${pct.toFixed(1)}%)`}
            >
              {pct > 5 && (
                <span className="truncate px-1">
                  {layer.label}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-400">
        {timedLayers.map((layer) => {
          const ms = result[layer.key] || 0;
          const cat = CATEGORY_COLORS[layer.category];
          return (
            <span key={layer.id} className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-sm"
                style={{ backgroundColor: cat.active }}
              />
              <span className="text-slate-500">{layer.label}</span>
              <span className="font-mono text-slate-300">{ms.toFixed(1)}ms</span>
            </span>
          );
        })}
        <span className="ml-auto font-medium text-emerald-400 font-mono">
          Total: {total.toFixed(1)}ms
        </span>
      </div>
    </div>
  );
}
