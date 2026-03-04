/**
 * BenchmarkPage - Compare HE library performance.
 * This page will contain the library comparison bar chart,
 * results table, and single-library benchmark mode.
 */
export default function BenchmarkPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-white mb-2">
        Library Benchmark
      </h1>
      <p className="text-slate-400 mb-8">
        Compare encryption performance across SEAL, HELib, and OpenFHE.
      </p>
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-8 text-center text-slate-500">
        Benchmark comparison charts coming in Commit 12–15...
      </div>
    </div>
  );
}
