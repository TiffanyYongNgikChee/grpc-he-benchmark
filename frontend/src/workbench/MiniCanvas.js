import { useRef, useState, useEffect, useCallback } from "react";

/**
 * MiniCanvas — The drawing pad where users write a handwritten digit.
 *
 * HOW IT WORKS (in plain English):
 *
 * The user draws on a big 280×280 canvas (so it's easy to draw with a mouse).
 * But the AI model expects a tiny 28×28 image where the digit is centred and
 * size-normalised — that's how MNIST training images look.
 *
 * So after each stroke, we:
 *   1. FIND the drawing — scan all pixels to locate the bounding box (the
 *      smallest rectangle that wraps around the drawn digit).
 *   2. CROP it tight — cut out just the digit, no wasted black space.
 *   3. MAKE IT SQUARE — if the digit is tall and skinny (like "1"), pad the
 *      sides with black so it becomes a square.
 *   4. SHRINK it — scale that square down to 20×20 pixels.
 *   5. CENTRE it — place the 20×20 image in the middle of a 28×28 black
 *      frame with 4 pixels of padding on every side.
 *
 * The result is 784 grayscale values (28×28) that look just like the training
 * data, so the AI gets what it expects regardless of where or how big you drew.
 */
export default function MiniCanvas({ onPixelsReady, disabled = false }) {
  const canvasRef = useRef(null);       // reference to the HTML <canvas> element
  const [isDrawing, setIsDrawing] = useState(false);  // true while mouse/finger is held down
  const [hasDrawn, setHasDrawn] = useState(false);    // true once the user has drawn anything (shows "Clear" button)

  const CANVAS_SIZE = 280;   // internal canvas resolution (px) — big for smooth drawing
  const BRUSH_SIZE = 14;     // radius of the white brush stroke (px)
  const MNIST_SIZE = 28;     // output image size the AI model expects

  // Fill the canvas with solid black (fresh start / clear)
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(() => { initCanvas(); }, [initCanvas]); // fill black on first render

  /**
   * getPos — Convert a mouse/touch event to canvas coordinates.
   * The canvas displays smaller than 280px on screen (max-width 196px),
   * so we scale the click position up to match the internal 280×280 grid.
   */
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

  // Draw a white line between two points (called while dragging)
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

  // Draw a white dot at a single point (called on initial click/tap)
  function drawDot(x, y) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_SIZE, 0, Math.PI * 2);
    ctx.fill();
  }

  const lastPos = useRef(null); // remembers the previous cursor position for smooth lines

  // --- Mouse / touch event handlers ---
  // handleStart: user puts finger/mouse down → start drawing
  function handleStart(e) {
    if (disabled) return;
    e.preventDefault();
    setIsDrawing(true);
    setHasDrawn(true);
    const pos = getPos(e);
    drawDot(pos.x, pos.y);
    lastPos.current = pos;
  }

  // handleMove: user drags across the canvas → draw a line from the last point
  function handleMove(e) {
    if (!isDrawing || disabled) return;
    e.preventDefault();
    const pos = getPos(e);
    if (lastPos.current) {
      drawLine(lastPos.current.x, lastPos.current.y, pos.x, pos.y);
    }
    lastPos.current = pos;
  }

  // handleEnd: user lifts finger/mouse → stop drawing and extract the digit
  function handleEnd(e) {
    if (!isDrawing) return;
    e.preventDefault();
    setIsDrawing(false);
    lastPos.current = null;
    extractPixels(); // <-- this is where the 5-step pipeline kicks in
  }

  /**
   * extractPixels — the core 5-step pipeline that turns a freehand drawing
   * into a clean 28×28 grayscale image (784 numbers), just like the MNIST
   * dataset the model was trained on.
   *
   *  Step 1 → grab the raw pixel data from the big 280×280 canvas
   *  Step 2 → find the bounding box (the smallest rectangle around the digit)
   *  Step 3 → crop it out and stretch it into a perfect square
   *  Step 4 → shrink that square to 20×20 and centre it inside a 28×28 frame
   *  Step 5 → read out 784 brightness values (0 = black, 255 = white)
   */
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

    // If nothing was drawn (or only tiny speckles), bail out
    if (maxX <= minX || maxY <= minY) {
      onPixelsReady?.(null);
      return;
    }

    // Step 3: Make it square so the digit isn't stretched
    // MNIST digits sit in a ~20×20 area centred inside 28×28, so we need a
    // square crop first, then we scale down.
    const bw = maxX - minX + 1; // width of the bounding box
    const bh = maxY - minY + 1; // height of the bounding box
    const side = Math.max(bw, bh); // pick the longer edge → square

    // Create a temporary square canvas and paste the digit centred
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = side;
    cropCanvas.height = side;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.fillStyle = "#000000";
    cropCtx.fillRect(0, 0, side, side);
    const offsetX = Math.floor((side - bw) / 2); // horizontal nudge to centre
    const offsetY = Math.floor((side - bh) / 2); // vertical nudge to centre
    cropCtx.drawImage(canvas, minX, minY, bw, bh, offsetX, offsetY, bw, bh);

    // Step 4: Shrink to 20×20 and centre inside a 28×28 black frame
    // (This matches exactly how the MNIST training images are formatted.)
    const INNER = 20; // the digit lives in a 20×20 area
    const PAD = Math.floor((MNIST_SIZE - INNER) / 2); // 4px padding each side

    const outCanvas = document.createElement("canvas");
    outCanvas.width = MNIST_SIZE;
    outCanvas.height = MNIST_SIZE;
    const outCtx = outCanvas.getContext("2d");
    outCtx.fillStyle = "#000000";
    outCtx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE); // start with a black 28×28
    // Smooth scaling so the digit looks clean when shrunk
    outCtx.imageSmoothingEnabled = true;
    outCtx.imageSmoothingQuality = "high";
    outCtx.drawImage(cropCanvas, 0, 0, side, side, PAD, PAD, INNER, INNER);

    // Step 5: Read out the 784 pixel brightness values (one per pixel)
    const data = outCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE).data;
    const pixels = [];
    for (let i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
      pixels.push(data[i * 4]); // red channel = grayscale (R=G=B for grey)
    }
    onPixelsReady?.(pixels); // send the 784 values up to the parent component
  }

  // handleClear: wipe the canvas back to black and tell the parent there are no pixels
  function handleClear() {
    initCanvas();
    setHasDrawn(false);
    onPixelsReady?.(null);
  }

  // --- Render ---
  // A <canvas> element wired to mouse + touch events, plus a "Clear" button
  // that only appears once the user has drawn something.
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
