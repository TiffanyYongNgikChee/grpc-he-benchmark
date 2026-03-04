/**
 * PredictPage - Draw a digit and get an encrypted prediction.
 * This page will contain the drawing canvas, result display,
 * logits chart, and layer timing breakdown.
 */
export default function PredictPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-white mb-2">
        Encrypted Digit Prediction
      </h1>
      <p className="text-slate-400 mb-8">
        Draw a digit (0–9) and run inference on encrypted data using homomorphic encryption.
      </p>
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-8 text-center text-slate-500">
        Drawing canvas and prediction results coming in Commit 5–8...
      </div>
    </div>
  );
}
