import { useRef, useState, useEffect, useCallback } from "react";

/**
 * DrawingCanvas - A 280×280 HTML canvas that lets users draw a digit with mouse or touch.
 *
 * The canvas is 280×280 on screen (10× the MNIST 28×28 size) so it's easy to draw on.
 * It draws white strokes on a black background (matching MNIST format).
 *
 * Props:
 *   onPixelsReady(pixels) - called with 784-element array when canvas changes
 *   disabled - if true, disables drawing (e.g. while prediction is loading)
 */
export default function DrawingCanvas({ onPixelsReady, disabled = false }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);

  const CANVAS_SIZE = 280; // Display size (10× MNIST's 28×28)
  const BRUSH_SIZE = 16;   // Brush radius — thick enough to look like a marker
  const MNIST_SIZE = 28;   // Output size for the model

  // Initialise canvas with black background
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, [CANVAS_SIZE]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  // Get mouse/touch position relative to canvas
  function getPos(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  // Draw a circle at the given position (simulates a thick brush stroke)
  function draw(x, y) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_SIZE, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw a line between two points (smooth strokes between mouse events)
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

  // Track last position for smooth line drawing
  const lastPos = useRef(null);

  function handleStart(e) {
    if (disabled) return;
    e.preventDefault();
    setIsDrawing(true);
    setHasDrawn(true);
    const pos = getPos(e);
    draw(pos.x, pos.y);
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
    // Extract pixels after the stroke is finished
    extractPixels();
  }

  /**
   * Extract 784 pixel values from the canvas.
   * Downscale 280×280 → 28×28 using an offscreen canvas, then read grayscale values.
   */
  function extractPixels() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Create offscreen canvas at MNIST size
    const offscreen = document.createElement("canvas");
    offscreen.width = MNIST_SIZE;
    offscreen.height = MNIST_SIZE;
    const offCtx = offscreen.getContext("2d");

    // Smooth downscale
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = "high";
    offCtx.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);

    // Read pixel data (RGBA format, 4 bytes per pixel)
    const imageData = offCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE);
    const pixels = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      // Use red channel as grayscale (canvas is black/white so R=G=B)
      pixels.push(imageData.data[i]);
    }

    if (onPixelsReady) {
      onPixelsReady(pixels);
    }
  }

  // Clear the canvas back to black
  function handleClear() {
    initCanvas();
    setHasDrawn(false);
    lastPos.current = null;
    if (onPixelsReady) {
      onPixelsReady(null);
    }
  }

  return (
    <div className="flex flex-col items-center">
      {/* Drawing canvas */}
      <div
        className={`relative rounded-lg overflow-hidden border-2 ${
          disabled
            ? "border-slate-600 opacity-60 cursor-not-allowed"
            : "border-slate-600 hover:border-emerald-500 cursor-crosshair"
        }`}
      >
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="block"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
          // Mouse events
          onMouseDown={handleStart}
          onMouseMove={handleMove}
          onMouseUp={handleEnd}
          onMouseLeave={handleEnd}
          // Touch events (mobile)
          onTouchStart={handleStart}
          onTouchMove={handleMove}
          onTouchEnd={handleEnd}
        />
        {/* Hint text when canvas is empty */}
        {!hasDrawn && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <p className="text-slate-500 text-sm">Draw a digit here (0–9)</p>
          </div>
        )}
      </div>

      {/* Clear button */}
      <button
        onClick={handleClear}
        disabled={disabled}
        className="mt-3 px-4 py-1.5 text-sm rounded-md bg-slate-700 text-slate-300 
                   hover:bg-slate-600 hover:text-white transition-colors
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Clear Canvas
      </button>
    </div>
  );
}
