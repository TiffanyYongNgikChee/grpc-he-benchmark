/**
 * Component tests for the Workbench page and its sub-components.
 *
 * These tests validate rendering, user interaction, and state management
 * for OutputPanel, MetricsStrip, ParameterComparison, and BenchmarkResultsDashboard.
 *
 * Canvas & chart.js are mocked so tests run in JSDOM without WebGL.
 */

import React from "react";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom";

/* ══════════════════════════════════════════
   Mocks — chart.js, canvas, CountUp
   ══════════════════════════════════════════ */
// Mock react-chartjs-2 (chart canvas can't render in JSDOM)
jest.mock("react-chartjs-2", () => ({
  Bar: (props) => <div data-testid="mock-chart-bar" />,
}));

// Mock CountUp to just render the end value immediately
jest.mock("../components/CountUp", () => {
  return function MockCountUp({ end, suffix = "", decimals = 0 }) {
    return <span>{Number(end).toFixed(decimals)}{suffix}</span>;
  };
});

// Mock CnnPipeline exports used by MetricsStrip
jest.mock("./CnnPipeline", () => ({
  LAYERS: [
    { id: "encrypt", label: "Encrypt", sub: "BFV", category: "crypto", key: "encryptionMs" },
    { id: "conv1",   label: "Conv1",   sub: "5x5",  category: "conv",   key: "conv1Ms" },
    { id: "relu1",   label: "x²",      sub: "act",  category: "act",    key: "act1Ms" },
    { id: "pool1",   label: "Pool1",   sub: "2x2",  category: "pool",   key: "pool1Ms" },
    { id: "conv2",   label: "Conv2",   sub: "5x5",  category: "conv",   key: "conv2Ms" },
    { id: "relu2",   label: "x²",      sub: "act",  category: "act",    key: "act2Ms" },
    { id: "pool2",   label: "Pool2",   sub: "2x2",  category: "pool",   key: "pool2Ms" },
    { id: "fc",      label: "FC",      sub: "16→10", category: "fc",    key: "fcMs" },
    { id: "decrypt", label: "Decrypt", sub: "BFV", category: "crypto",  key: "decryptionMs" },
  ],
  CATEGORY_COLORS: {
    crypto: { active: "#0db7c4" },
    conv:   { active: "#7b3ff2" },
    act:    { active: "#e68a00" },
    pool:   { active: "#e68a00" },
    fc:     { active: "#e03e52" },
  },
}));

/* ══════════════════════════════════════════
   Import components AFTER mocks
   ══════════════════════════════════════════ */
import OutputPanel from "./OutputPanel";
import MetricsStrip from "./MetricsStrip";
import ParameterComparison from "./ParameterComparison";

/* ══════════════════════════════════════════
   OutputPanel
   ══════════════════════════════════════════ */
describe("OutputPanel", () => {
  it("shows draw prompt when no pixels provided", () => {
    render(<OutputPanel />);
    expect(screen.getByText("Draw a digit on the left to start")).toBeInTheDocument();
  });

  it("shows run prompt when pixels exist but no result", () => {
    render(<OutputPanel pixels={new Array(784).fill(0)} />);
    expect(screen.getByText("Press the run button to start encrypted inference")).toBeInTheDocument();
  });

  it("renders loading spinner with progress", () => {
    const layerStatus = { encrypt: "done", conv1: "done", bias1: "done", relu1: "processing" };
    render(<OutputPanel loading={true} layerStatus={layerStatus} elapsedMs={5000} />);
    // Should show elapsed time
    expect(screen.getByText(/5\.0s/)).toBeInTheDocument();
    // Should show processing layer
    expect(screen.getByText(/RELU1/)).toBeInTheDocument();
  });

  it("renders error message", () => {
    render(<OutputPanel error="gRPC connection failed" />);
    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("gRPC connection failed")).toBeInTheDocument();
    expect(screen.getByText(/Docker/)).toBeInTheDocument();
  });

  it("renders prediction result correctly", () => {
    const result = {
      predictedDigit: 7,
      confidence: 0.95,
      totalMs: 15000,
      status: "success",
      logits: [0.1, 0.05, 0.02, 0.01, 0.03, 0.02, 0.01, 0.95, 0.01, 0.02],
    };
    render(<OutputPanel result={result} />);

    // Predicted digit
    expect(screen.getByText("7")).toBeInTheDocument();
    // Confidence
    expect(screen.getByText("95.0%")).toBeInTheDocument();
    // Total time
    expect(screen.getByText("15000ms")).toBeInTheDocument();
    // Status badge
    expect(screen.getByText("success")).toBeInTheDocument();
    // Chart
    expect(screen.getByTestId("mock-chart-bar")).toBeInTheDocument();
  });

  it("shows float model accuracy when available", () => {
    const result = {
      predictedDigit: 3,
      confidence: 0.88,
      totalMs: 12000,
      status: "success",
      logits: Array(10).fill(0),
      floatModelAccuracy: 88.86,
    };
    render(<OutputPanel result={result} />);
    expect(screen.getByText(/88.86%/)).toBeInTheDocument();
  });
});

/* ══════════════════════════════════════════
   MetricsStrip
   ══════════════════════════════════════════ */
describe("MetricsStrip", () => {
  it("shows placeholder text when no result", () => {
    render(<MetricsStrip />);
    expect(screen.getByText("Run inference to see per-layer timing breakdown")).toBeInTheDocument();
  });

  it("renders timing breakdown with total", () => {
    const result = {
      encryptionMs: 64,
      conv1Ms: 3475,
      act1Ms: 342,
      pool1Ms: 474,
      conv2Ms: 3133,
      act2Ms: 307,
      pool2Ms: 485,
      fcMs: 5611,
      decryptionMs: 26,
      totalMs: 13919,
    };
    render(<MetricsStrip result={result} />);

    // Total should be displayed
    expect(screen.getByText("Total: 13919.0ms")).toBeInTheDocument();
    // Layer labels present (use getAllByText since labels appear in both bar + legend)
    expect(screen.getAllByText("Encrypt").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("FC").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Decrypt").length).toBeGreaterThanOrEqual(1);
  });
});

/* ══════════════════════════════════════════
   ParameterComparison
   ══════════════════════════════════════════ */
describe("ParameterComparison", () => {
  const mockBenchmarkJson = {
    generated_at: "auto",
    scheme: "BFV",
    library: "OpenFHE v1.2.2",
    plaintext_modulus: 100073473,
    scale_factor: 1000,
    total_images: 30,
    configs: [
      {
        degree: 2,
        label: "x\u00b2 (degree 2)",
        security: "128-bit",
        num_images: 10,
        accuracy: "10/10",
        accuracy_pct: 100.0,
        avg_total_ms: 13919.2,
        layer_averages: {
          encryption_ms: 64, conv1_ms: 3475, act1_ms: 342, pool1_ms: 474,
          conv2_ms: 3133, act2_ms: 307, pool2_ms: 485, fc_ms: 5611, decryption_ms: 26,
        },
      },
      {
        degree: 3,
        label: "x\u00b3 (degree 3)",
        security: "128-bit",
        num_images: 10,
        accuracy: "3/10",
        accuracy_pct: 30.0,
        avg_total_ms: 12160.2,
        layer_averages: {
          encryption_ms: 65, conv1_ms: 2777, act1_ms: 85, pool1_ms: 446,
          conv2_ms: 2885, act2_ms: 83, pool2_ms: 438, fc_ms: 5353, decryption_ms: 26,
        },
      },
      {
        degree: 4,
        label: "x\u2074 (degree 4)",
        security: "128-bit",
        num_images: 10,
        accuracy: "0/10",
        accuracy_pct: 0.0,
        avg_total_ms: 13695.8,
        layer_averages: {
          encryption_ms: 61, conv1_ms: 3454, act1_ms: 88, pool1_ms: 510,
          conv2_ms: 3348, act2_ms: 85, pool2_ms: 505, fc_ms: 5617, decryption_ms: 26,
        },
      },
    ],
  };

  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockBenchmarkJson),
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("renders activation tab by default", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      // Default tab shows activation degree configurations in the table
      expect(screen.getAllByText(/x\u00B2 \(degree 2\)/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/degree 3/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/degree 4/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows accuracy values for activation degrees", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("10/10")).toBeInTheDocument();
      expect(screen.getByText("3/10")).toBeInTheDocument();
      expect(screen.getByText("0/10")).toBeInTheDocument();
    });
  });

  it("switches to security level tab", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("Security Level")).toBeInTheDocument();
    });
    // Click security tab
    fireEvent.click(screen.getByText("Security Level"));
    // Security-specific data should appear (OOM notes only in security tab)
    expect(screen.getByText(/OOM/)).toBeInTheDocument();
    expect(screen.getAllByText("192-bit").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("256-bit").length).toBeGreaterThanOrEqual(1);
  });

  it("shows key findings callout for activation tab", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("Key Findings")).toBeInTheDocument();
    });
    // Activation-specific finding about polynomial matching training
    expect(screen.getByText(/matches training activation/i)).toBeInTheDocument();
  });

  it("shows different key findings for security tab", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("Security Level")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Security Level"));
    // Security-specific findings about memory requirements
    expect(screen.getAllByText(/ring dimension/i).length).toBeGreaterThanOrEqual(1);
  });

  it("renders experiment metadata", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText(/OpenFHE v1\.2\.2/)).toBeInTheDocument();
    });
    expect(screen.getByText(/p = 100,073,473/)).toBeInTheDocument();
  });

  it("shows per-layer breakdown chart on activation tab", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      // The stacked bar chart should be rendered (mocked)
      expect(screen.getByText("Per-Layer Timing Breakdown (ms)")).toBeInTheDocument();
    });
  });

  it("hides per-layer chart on security tab", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("Security Level")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Security Level"));
    expect(screen.queryByText("Per-Layer Timing Breakdown (ms)")).not.toBeInTheDocument();
  });

  it("shows dynamic image count in metadata", async () => {
    render(<ParameterComparison />);
    await waitFor(() => {
      expect(screen.getByText("10 MNIST test images per config")).toBeInTheDocument();
    });
  });

  it("shows loading state while fetching", () => {
    global.fetch.mockReturnValue(new Promise(() => {}));
    render(<ParameterComparison />);
    expect(screen.getByText(/loading experiment/i)).toBeInTheDocument();
  });
});

/* ══════════════════════════════════════════
   BenchmarkResultsDashboard
   ══════════════════════════════════════════ */
describe("BenchmarkResultsDashboard", () => {
  // We need to mock fetch for this component since it loads benchmark_data.json
  const mockBenchmarkData = {
    generated_at: "auto",
    description: "Test benchmark data",
    scheme: "BFV",
    library: "OpenFHE v1.2.2",
    plaintext_modulus: 100073473,
    scale_factor: 1000,
    total_images: 10,
    configs: [
      {
        degree: 2,
        label: "x² (degree 2)",
        security: "128-bit",
        num_images: 10,
        accuracy: "10/10",
        accuracy_pct: 100.0,
        avg_total_ms: 13919,
        median_total_ms: 13800,
        min_total_ms: 13500,
        max_total_ms: 14200,
        std_total_ms: 200,
        layer_labels: ["Encryption", "Conv1", "Activation1", "Pool1", "Conv2", "Activation2", "Pool2", "FC", "Decryption"],
        layer_averages: {
          encryption_ms: 64, conv1_ms: 3475, act1_ms: 342, pool1_ms: 474,
          conv2_ms: 3133, act2_ms: 307, pool2_ms: 485, fc_ms: 5611, decryption_ms: 26,
        },
        per_digit_accuracy: {
          "7": { correct: 1, total: 1, pct: 100.0 },
        },
      },
    ],
    raw_results: {
      deg2: [
        {
          image_index: 0,
          true_label: 7,
          predicted: 7,
          correct: true,
          confidence: 0.95,
          total_ms: 13919,
          status: "success",
          encryption_ms: 64,
          conv1_ms: 3475,
          act1_ms: 342,
          pool1_ms: 474,
          conv2_ms: 3133,
          act2_ms: 307,
          pool2_ms: 485,
          fc_ms: 5611,
          decryption_ms: 26,
        },
      ],
    },
  };

  // Dynamically import so we can set up fetch mock first
  let BenchmarkResultsDashboard;

  beforeAll(async () => {
    // Import after mocks are registered
    const mod = await import("./BenchmarkResultsDashboard");
    BenchmarkResultsDashboard = mod.default;
  });

  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("shows loading state initially", () => {
    // Make fetch hang (never resolve)
    global.fetch.mockReturnValue(new Promise(() => {}));
    render(<BenchmarkResultsDashboard />);
    expect(screen.getByText(/loading benchmark/i)).toBeInTheDocument();
  });

  it("renders dashboard with data", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockBenchmarkData),
    });

    render(<BenchmarkResultsDashboard />);

    await waitFor(() => {
      // Should show the "FHE Benchmark Results" header once data loads
      expect(screen.getByText("FHE Benchmark Results")).toBeInTheDocument();
    });
  });

  it("handles fetch error gracefully", async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
    });

    render(<BenchmarkResultsDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/not found|error|no benchmark/i)).toBeInTheDocument();
    });
  });
});
