import { useState } from "react";
import DrawingCanvas from "../components/DrawingCanvas";

/**
 * PredictPage - Draw a digit and get an encrypted prediction.
 * This page will contain the drawing canvas, result display,
 * logits chart, and layer timing breakdown.
 */
export default function PredictPage() {
  const [pixels, setPixels] = useState(null);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-white mb-2">
        Encrypted Digit Prediction
      </h1>
      <p className="text-slate-400 mb-8">
        Draw a digit (0–9) and run inference on encrypted data using homomorphic encryption.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left column: Drawing canvas */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Draw a Digit</h2>
          <DrawingCanvas onPixelsReady={setPixels} />

          {/* Pixel preview — shows the 28×28 downscaled version */}
          {pixels && (
            <div className="mt-4">
              <p className="text-xs text-slate-500 mb-2">
                28×28 preview (what the model sees):
              </p>
              <canvas
                width={28}
                height={28}
                className="border border-slate-600 rounded"
                style={{ width: 112, height: 112, imageRendering: "pixelated" }}
                ref={(el) => {
                  if (!el || !pixels) return;
                  const ctx = el.getContext("2d");
                  const img = ctx.createImageData(28, 28);
                  for (let i = 0; i < 784; i++) {
                    const v = pixels[i];
                    img.data[i * 4] = v;
                    img.data[i * 4 + 1] = v;
                    img.data[i * 4 + 2] = v;
                    img.data[i * 4 + 3] = 255;
                  }
                  ctx.putImageData(img, 0, 0);
                }}
              />
            </div>
          )}
        </div>

        {/* Right column: Results (coming next commit) */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Prediction Result</h2>
          <div className="text-center text-slate-500 py-12">
            {pixels
              ? "Predict button and results coming next..."
              : "Draw a digit on the left to get started"}
          </div>
        </div>
      </div>
    </div>
  );
}
