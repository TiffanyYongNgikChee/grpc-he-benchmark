/**
 * BenchmarkTable — Tabular breakdown of benchmark timings.
 *
 * Columns: Operation | Library₁ | Library₂ | … (or just one library in single mode)
 * Rows: KeyGen, Encrypt, Add, Multiply, Decrypt, Total
 *
 * The fastest value per row is highlighted in emerald.
 *
 * Props:
 *  - results: LibraryResult[]  (1 or 3 items)
 */

const OPERATIONS = [
  { key: "keyGenTimeMs",         label: "KeyGen" },
  { key: "encryptionTimeMs",     label: "Encrypt" },
  { key: "additionTimeMs",       label: "Add" },
  { key: "multiplicationTimeMs", label: "Multiply" },
  { key: "decryptionTimeMs",     label: "Decrypt" },
  { key: "totalTimeMs",          label: "Total" },
];

export default function BenchmarkTable({ results }) {
  const multi = results.length > 1;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left text-xs text-slate-400 font-medium py-3 pr-4">
              Operation
            </th>
            {results.map((lib) => (
              <th
                key={lib.library}
                className="text-right text-xs text-slate-400 font-medium py-3 px-4"
              >
                {lib.library}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {OPERATIONS.map((op) => {
            const values = results.map((lib) => lib[op.key] ?? 0);
            const minVal = multi ? Math.min(...values) : null;

            return (
              <tr
                key={op.key}
                className={`border-b border-slate-700/50 ${
                  op.key === "totalTimeMs" ? "font-semibold" : ""
                }`}
              >
                <td className="py-2.5 pr-4 text-slate-300">
                  {op.label}
                </td>
                {values.map((val, i) => {
                  const isFastest = multi && val === minVal && val > 0;
                  return (
                    <td
                      key={results[i].library}
                      className={`py-2.5 px-4 text-right font-mono ${
                        isFastest
                          ? "text-emerald-400"
                          : "text-slate-300"
                      }`}
                    >
                      {val.toFixed(2)}
                      <span className="text-slate-500 ml-1">ms</span>
                      {isFastest && (
                        <span className="ml-2 text-[10px] text-emerald-500 font-sans font-medium">
                          fastest
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>

        {/* Status footer row */}
        <tfoot>
          <tr>
            <td className="pt-3 pr-4 text-xs text-slate-500">Status</td>
            {results.map((lib) => (
              <td key={lib.library} className="pt-3 px-4 text-right">
                <span
                  className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium ${
                    lib.success
                      ? "bg-emerald-900/40 text-emerald-400 border border-emerald-700/50"
                      : "bg-red-900/40 text-red-400 border border-red-700/50"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      lib.success ? "bg-emerald-400" : "bg-red-400"
                    }`}
                  />
                  {lib.success ? "success" : "failed"}
                </span>
              </td>
            ))}
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
