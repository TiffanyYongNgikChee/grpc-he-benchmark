import { useRef, useState, useEffect, useCallback } from "react";

/**
 * MiniCanvas — A compact 140×140 drawing canvas for the playground layout.
 * Same functionality as DrawingCanvas but sized to fit in the playground's
 * input column (TF-playground style).
 */
export default function MiniCanvas({ onPixelsReady, disabled = false }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);

  const CANVAS_SIZE = 196;
  const BRUSH_SIZE = 10;
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
    const offscreen = document.createElement("canvas");
    offscreen.width = MNIST_SIZE;
    offscreen.height = MNIST_SIZE;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);
    const data = offCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE).data;
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
        className="rounded border-2 border-slate-600 cursor-crosshair w-full"
        style={{ imageRendering: "pixelated", touchAction: "none" }}
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
          className="absolute top-1 right-1 bg-slate-700/80 hover:bg-slate-600 text-slate-300
                     rounded px-2 py-0.5 text-xs transition-colors"
        >
          Clear
        </button>
      )}
    </div>
  );
}
