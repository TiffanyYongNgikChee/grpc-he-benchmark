import { useRef, useState, useEffect, useCallback } from "react";

/**
 * MiniCanvas — A compact drawing canvas for the workbench layout.
 * Draws white strokes on a black background, then extracts a centered
 * 28×28 grayscale image matching MNIST preprocessing:
 *   1. Find bounding box of drawn strokes
 *   2. Crop to a square around the digit
 *   3. Scale to 20×20 (MNIST inner region)
 *   4. Center in a 28×28 frame with 4px padding
 */
export default function MiniCanvas({ onPixelsReady, disabled = false }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);

  const CANVAS_SIZE = 280;
  const BRUSH_SIZE = 14;
  const MNIST_SIZE = 28;

  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(() => { initCanvas(); }, [initCanvas]);

  function getPos(e) {
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = CANVAS_SIZE / rect.width;
    if (e.touches) {
      return {
        x: (e.touches[0].clientX - rect.left) * scale,
        y: (e.touches[0].clientY - rect.top) * scale,
      };
    }
    return {
      x: (e.clientX - rect.left) * scale,
      y: (e.clientY - rect.top) * scale,
    };
  }

  function drawLine(x1, y1, x2, y2) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = BRUSH_SIZE * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  function drawDot(x, y) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_SIZE, 0, Math.PI * 2);
    ctx.fill();
  }

  const lastPos = useRef(null);

  function handleStart(e) {
    if (disabled) return;
    e.preventDefault();
    setIsDrawing(true);
    setHasDrawn(true);
    const pos = getPos(e);
    drawDot(pos.x, pos.y);
    lastPos.current = pos;
  }

  function handleMove(e) {
    if (!isDrawing || disabled) return;
    e.preventDefault();
    const pos = getPos(e);
    if (lastPos.current) {
      drawLine(lastPos.current.x, lastPos.current.y, pos.x, pos.y);
    }
    lastPos.current = pos;
  }

  function handleEnd(e) {
    if (!isDrawing) return;
    e.preventDefault();
    setIsDrawing(false);
    lastPos.current = null;
    extractPixels();
  }

  function extractPixels() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Step 1: Get full-resolution image data from the drawing canvas
    const srcCtx = canvas.getContext("2d");
    const srcData = srcCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;

    // Step 2: Find bounding box of the drawn digit (non-black pixels)
    let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = 0, maxY = 0;
    const THRESHOLD = 30; // ignore near-black pixels from anti-aliasing
    for (let y = 0; y < CANVAS_SIZE; y++) {
      for (let x = 0; x < CANVAS_SIZE; x++) {
        const idx = (y * CANVAS_SIZE + x) * 4;
        if (srcData[idx] > THRESHOLD) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // If nothing drawn, bail
    if (maxX <= minX || maxY <= minY) {
      onPixelsReady?.(null);
      return;
    }

    // Step 3: Crop to bounding box and fit into a square with padding
    // MNIST digits occupy ~20×20 inside the 28×28 frame, centered.
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const side = Math.max(bw, bh);

    // Create a square crop canvas and center the digit
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = side;
    cropCanvas.height = side;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.fillStyle = "#000000";
    cropCtx.fillRect(0, 0, side, side);
    const offsetX = Math.floor((side - bw) / 2);
    const offsetY = Math.floor((side - bh) / 2);
    cropCtx.drawImage(canvas, minX, minY, bw, bh, offsetX, offsetY, bw, bh);

    // Step 4: Scale to 20×20 (MNIST convention) then paste centered in 28×28
    const INNER = 20; // MNIST digits are roughly 20×20 inside 28×28
    const PAD = Math.floor((MNIST_SIZE - INNER) / 2); // 4px padding each side

    const outCanvas = document.createElement("canvas");
    outCanvas.width = MNIST_SIZE;
    outCanvas.height = MNIST_SIZE;
    const outCtx = outCanvas.getContext("2d");
    outCtx.fillStyle = "#000000";
    outCtx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);
    // Enable smoothing for better anti-aliased downscale
    outCtx.imageSmoothingEnabled = true;
    outCtx.imageSmoothingQuality = "high";
    outCtx.drawImage(cropCanvas, 0, 0, side, side, PAD, PAD, INNER, INNER);

    // Step 5: Extract pixel values
    const data = outCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE).data;
    const pixels = [];
    for (let i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
      pixels.push(data[i * 4]); // red channel = grayscale
    }
    onPixelsReady?.(pixels);
  }

  function handleClear() {
    initCanvas();
    setHasDrawn(false);
    onPixelsReady?.(null);
  }

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="rounded border-2 cursor-crosshair w-full max-w-[196px]"
        style={{ imageRendering: "pixelated", touchAction: "none", borderColor: "#d9d9d9" }}
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
      />
      {hasDrawn && (
        <button
          onClick={handleClear}
          className="absolute top-1 right-1 bg-white/80 hover:bg-gray-100 text-gray-500
                     rounded px-2 py-0.5 text-xs transition-colors border border-gray-300"
        >
          Clear
        </button>
      )}
    </div>
  );
}
