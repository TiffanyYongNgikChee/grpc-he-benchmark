/**
 * ArchitectureDiagram — Step-by-step visual walkthrough explaining
 * what happens to a handwritten digit image as it passes through
 * the encrypted CNN pipeline.
 *
 * Each step shows:
 *   - A visual representation of the data at that stage
 *   - What operation is being performed
 *   - The dimensions of the data
 *   - Whether it's encrypted or plaintext
 *
 * Pure SVG + HTML in React — no external dependencies.
 */

const C = {
  io:     "#0aa35e",
  crypto: "#0db7c4",
  conv:   "#7b3ff2",
  act:    "#e68a00",
  fc:     "#e03e52",
  bg:     "#f8f8f8",
  border: "#e5e5e5",
  text:   "#333",
  muted:  "#888",
  faint:  "#bbb",
};

/* ── Pixel grid: draws a tiny grid of cells to represent image data ── */
function PixelGrid({ rows, cols, cellSize, color, encrypted, label }) {
  const w = cols * cellSize;
  const h = rows * cellSize;
  return (
    <svg width={w + 2} height={h + 2} viewBox={`-1 -1 ${w + 2} ${h + 2}`}>
      {/* Background */}
      <rect x={0} y={0} width={w} height={h} rx={2}
        fill={encrypted ? color : "#fff"} fillOpacity={encrypted ? 0.08 : 1}
        stroke={color} strokeWidth={1.2}
      />
      {/* Grid lines */}
      {Array.from({ length: rows - 1 }, (_, i) => (
        <line key={`r${i}`}
          x1={0} y1={(i + 1) * cellSize} x2={w} y2={(i + 1) * cellSize}
          stroke={color} strokeOpacity={0.15} strokeWidth={0.5}
        />
      ))}
      {Array.from({ length: cols - 1 }, (_, i) => (
        <line key={`c${i}`}
          x1={(i + 1) * cellSize} y1={0} x2={(i + 1) * cellSize} y2={h}
          stroke={color} strokeOpacity={0.15} strokeWidth={0.5}
        />
      ))}
      {/* Encrypted overlay pattern */}
      {encrypted && (
        <>
          {Array.from({ length: Math.floor(rows * cols * 0.3) }, (_, i) => {
            const cx = ((i * 7 + 3) % cols) * cellSize + cellSize / 2;
            const cy = ((i * 11 + 5) % rows) * cellSize + cellSize / 2;
            return (
              <text key={`e${i}`} x={cx} y={cy + 1}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={cellSize * 0.6} fill={color} fillOpacity={0.3}
                fontFamily="monospace"
              >
                ?
              </text>
            );
          })}
        </>
      )}
      {/* Optional label inside */}
      {label && (
        <text x={w / 2} y={h / 2 + 1}
          textAnchor="middle" dominantBaseline="middle"
          fontSize={9} fontWeight={600} fill={color} fillOpacity={0.6}
        >
          {label}
        </text>
      )}
    </svg>
  );
}

/* ── Step arrow between stages ── */
function StepArrow() {
  return (
    <div className="flex items-center justify-center" style={{ minWidth: 28 }}>
      <svg width={28} height={20} viewBox="0 0 28 20">
        <line x1={2} y1={10} x2={20} y2={10} stroke={C.faint} strokeWidth={1.5} />
        <polygon points="20,5 28,10 20,15" fill={C.faint} />
      </svg>
    </div>
  );
}

/* ── Single step card ── */
function StepCard({ stepNum, title, color, encrypted, children, dimensions, operation }) {
  return (
    <div className="flex flex-col items-center text-center" style={{ minWidth: 0 }}>
      {/* Step number badge */}
      <div
        className="w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] font-bold mb-1.5"
        style={{ background: color }}
      >
        {stepNum}
      </div>
      {/* Title */}
      <p className="text-[11px] font-semibold mb-1" style={{ color }}>{title}</p>
      {/* Visual */}
      <div
        className="rounded-md p-2 mb-1.5"
        style={{
          background: encrypted ? `${color}08` : "#fff",
          border: `1.5px ${encrypted ? "dashed" : "solid"} ${color}40`,
        }}
      >
        {children}
      </div>
      {/* Dimensions */}
      {dimensions && (
        <p className="text-[10px] font-mono mb-0.5" style={{ color: C.muted }}>
          {dimensions}
        </p>
      )}
      {/* Operation description */}
      {operation && (
        <p className="text-[10px] max-w-[110px] leading-tight" style={{ color: C.faint }}>
          {operation}
        </p>
      )}
    </div>
  );
}

/* ── Logit bars (for FC output) ── */
function LogitBars({ highlight = 7 }) {
  const values = [2, 1, 3, 1, 2, 1, 0, 9, 1, 2]; // fake logits, 7 is highest
  const max = Math.max(...values);
  return (
    <svg width={80} height={50} viewBox="0 0 80 50">
      {values.map((v, i) => {
        const barH = (v / max) * 36;
        const isHighlight = i === highlight;
        return (
          <g key={i}>
            <rect
              x={i * 8} y={40 - barH}
              width={6} height={barH}
              rx={1}
              fill={isHighlight ? C.io : "#ddd"}
            />
            <text x={i * 8 + 3} y={48}
              textAnchor="middle" fontSize={5} fill={isHighlight ? C.io : C.faint}
              fontWeight={isHighlight ? 700 : 400}
            >
              {i}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── Main Component ── */
export default function ArchitectureDiagram() {
  return (
    <div
      className="rounded-lg p-5 mt-4"
      style={{ background: C.bg, border: `1px solid ${C.border}` }}
    >
      <p className="text-sm font-medium mb-1" style={{ color: C.text }}>
        Step-by-step: What happens to your drawn digit
      </p>
      <p className="text-xs mb-5" style={{ color: C.muted }}>
        Follow the data as it transforms from raw pixels through encrypted neural
        network layers to a final prediction.
      </p>

      {/* ── Pipeline walkthrough ── */}
      <div className="overflow-x-auto">
        <div className="flex items-start gap-0" style={{ minWidth: 960 }}>

          {/* Step 1: Input */}
          <StepCard
            stepNum={1} title="Your Drawing" color={C.io}
            dimensions="28 × 28 pixels"
            operation="Raw grayscale image, 784 values (0–255)"
          >
            <PixelGrid rows={7} cols={7} cellSize={8} color={C.io} />
          </StepCard>

          <StepArrow />

          {/* Step 2: Encrypt */}
          <StepCard
            stepNum={2} title="Encrypt" color={C.crypto} encrypted
            dimensions="784 ciphertexts"
            operation="Each pixel becomes an encrypted BFV integer — unreadable"
          >
            <PixelGrid rows={7} cols={7} cellSize={8} color={C.crypto} encrypted />
          </StepCard>

          <StepArrow />

          {/* Step 3: Conv1 */}
          <StepCard
            stepNum={3} title="Conv1" color={C.conv} encrypted
            dimensions="24 × 24"
            operation="Slide a 5×5 filter across encrypted data to detect edges"
          >
            <div className="relative">
              <PixelGrid rows={6} cols={6} cellSize={7} color={C.conv} encrypted />
              {/* Kernel overlay */}
              <div className="absolute top-[2px] left-[2px]" style={{
                width: 17, height: 17,
                border: `2px solid ${C.conv}`,
                borderRadius: 2,
                background: `${C.conv}20`,
              }} />
            </div>
          </StepCard>

          <StepArrow />

          {/* Step 4: Activation x² */}
          <StepCard
            stepNum={4} title="x² Activate" color={C.act} encrypted
            dimensions="24 × 24"
            operation="Square each value — adds non-linearity without decrypting"
          >
            <div className="flex flex-col items-center gap-1">
              <PixelGrid rows={6} cols={6} cellSize={7} color={C.act} encrypted />
              <svg width={50} height={16} viewBox="0 0 50 16">
                <text x={25} y={12} textAnchor="middle" fontSize={10}
                  fontFamily="monospace" fontWeight={700} fill={C.act}
                >
                  x → x²
                </text>
              </svg>
            </div>
          </StepCard>

          <StepArrow />

          {/* Step 5: Pool1 */}
          <StepCard
            stepNum={5} title="Pool1" color={C.act} encrypted
            dimensions="12 × 12"
            operation="Average 2×2 blocks — halves dimensions"
          >
            <PixelGrid rows={4} cols={4} cellSize={8} color={C.act} encrypted />
          </StepCard>

          <StepArrow />

          {/* Step 6: Conv2 */}
          <StepCard
            stepNum={6} title="Conv2" color={C.conv} encrypted
            dimensions="8 × 8"
            operation="Second 5×5 convolution — detects complex patterns"
          >
            <PixelGrid rows={4} cols={4} cellSize={7} color={C.conv} encrypted />
          </StepCard>

          <StepArrow />

          {/* Step 7: Act + Pool2 */}
          <StepCard
            stepNum={7} title="Act + Pool2" color={C.act} encrypted
            dimensions="4 × 4"
            operation="x² activation then 2×2 average pooling"
          >
            <PixelGrid rows={3} cols={3} cellSize={8} color={C.act} encrypted label="16" />
          </StepCard>

          <StepArrow />

          {/* Step 8: FC */}
          <StepCard
            stepNum={8} title="FC Layer" color={C.fc} encrypted
            dimensions="16 → 10"
            operation="Matrix multiply to produce 10 scores, one per digit"
          >
            <div className="flex flex-col items-center gap-1">
              {/* 10 output neurons */}
              <svg width={44} height={50} viewBox="0 0 44 50">
                {Array.from({ length: 10 }, (_, i) => (
                  <g key={i}>
                    <circle
                      cx={22} cy={2.5 + i * 4.8}
                      r={2}
                      fill={C.fc} fillOpacity={i === 7 ? 0.9 : 0.2}
                      stroke={C.fc} strokeWidth={0.5}
                    />
                    <text x={30} y={4 + i * 4.8}
                      fontSize={4} fill={i === 7 ? C.fc : C.faint}
                      fontWeight={i === 7 ? 700 : 400}
                    >
                      {i}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </StepCard>

          <StepArrow />

          {/* Step 9: Decrypt */}
          <StepCard
            stepNum={9} title="Decrypt" color={C.crypto}
            dimensions="10 integers"
            operation="Secret key reveals the 10 logit values"
          >
            <LogitBars highlight={7} />
          </StepCard>

          <StepArrow />

          {/* Step 10: Output */}
          <StepCard
            stepNum={10} title="Prediction" color={C.io}
            dimensions="argmax"
            operation="Highest logit wins — that index is the predicted digit"
          >
            <div
              className="flex items-center justify-center rounded-md"
              style={{
                width: 52, height: 52,
                background: `${C.io}10`,
                border: `2px solid ${C.io}`,
              }}
            >
              <span style={{ fontSize: 28, fontWeight: 800, color: C.io }}>7</span>
            </div>
          </StepCard>

        </div>
      </div>

      {/* ── Encryption boundary annotation ── */}
      <div className="mt-4 flex items-center gap-4">
        <div className="flex-1 h-px" style={{ background: C.border }} />
        <div className="flex items-center gap-3 text-[11px]" style={{ color: C.muted }}>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-6 h-0 border-t-2" style={{ borderColor: C.io, borderStyle: "solid" }} />
            Plaintext (steps 1, 9–10)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-6 h-0 border-t-2" style={{ borderColor: C.crypto, borderStyle: "dashed" }} />
            Encrypted (steps 2–8)
          </span>
        </div>
        <div className="flex-1 h-px" style={{ background: C.border }} />
      </div>

      <p className="text-[11px] text-center mt-2" style={{ color: C.faint }}>
        Steps 2–8 all run on encrypted data (ciphertext). The server performs convolutions,
        activations, pooling, and matrix multiplication without ever seeing the original pixel values.
        Only the final 10 logit numbers are decrypted — your image stays private throughout.
      </p>
    </div>
  );
}
