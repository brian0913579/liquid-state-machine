import { useEffect, useMemo, useRef, useState } from "react";
import { FORMULA_LABELS, estimateOps } from "@lsm-core";
import type { CurveFamily, TrainingConfig, TrainingResult } from "@lsm-core";
import "./App.css";
import ControlPanel from "./components/ControlPanel";
import LineChart from "./components/LineChart";
import Heatmap from "./components/Heatmap";
import CurvePlot from "./components/CurvePlot";

const MAX_OPS = 2.5e7;

const DEFAULT_CONFIG: TrainingConfig = {
  formula: "cardioid",
  timesteps: 80,
  trainSamples: 50,
  epochs: 24,
  learningRate: 0.04,
  neurons: 48,
  connectivity: 0.08,
  seed: 42,
};

const PRECOMPUTED_CONFIG: TrainingConfig = {
  ...DEFAULT_CONFIG,
};

async function loadPrecomputed(formula: CurveFamily): Promise<TrainingResult | null> {
  try {
    const response = await fetch(`./assets/precomputed/${formula}.json`);
    if (!response.ok) return null;
    return (await response.json()) as TrainingResult;
  } catch {
    return null;
  }
}

function App() {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "precomputed" | "error">("idle");
  const [message, setMessage] = useState<string>("");
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("./worker/lsmWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event) => {
      const { type, payload } = event.data;
      if (type === "result") {
        setResult(payload as TrainingResult);
        setStatus("idle");
        setMessage("Live training complete.");
      } else if (type === "tooHeavy") {
        setStatus("error");
        setMessage("Configuration exceeds the 10s target. Try fewer neurons/epochs or load a precomputed run.");
      } else if (type === "error") {
        setStatus("error");
        setMessage(payload ?? "Training failed.");
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const estimate = useMemo(() => estimateOps(config), [config]);
  const isHeavy = estimate > MAX_OPS;

  const runTraining = async () => {
    setMessage("");
    if (isHeavy) {
      const precomputed = await loadPrecomputed(config.formula);
      if (precomputed) {
        setResult(precomputed);
        setStatus("precomputed");
        setMessage(
          `Using precomputed run (${PRECOMPUTED_CONFIG.neurons} neurons, ${PRECOMPUTED_CONFIG.epochs} epochs) to stay within 10s.`,
        );
        return;
      }
      setStatus("error");
      setMessage("Config too heavy and no precomputed run found. Reduce neurons/epochs.");
      return;
    }
    setStatus("running");
    workerRef.current?.postMessage({ type: "train", payload: config });
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">Liquid State Machine Explorer</p>
          <h1>Interactive LSM Learning Lab</h1>
          <p className="subhead">
            Train a spiking reservoir on polar formulas and inspect how it learns in real time.
          </p>
        </div>
        <div className="formula-chip">
          {FORMULA_LABELS[config.formula]}
        </div>
      </header>

      <div className="dashboard">
        <ControlPanel
          config={config}
          onChange={setConfig}
          onRun={runTraining}
          disabled={status === "running"}
          estimate={estimate}
          isHeavy={isHeavy}
        />

        <section className="results">
          <div className="status">
            <span className={`pill ${status}`}>{status === "running" ? "Training…" : "Ready"}</span>
            <span className="message">{message}</span>
          </div>

          {!result && (
            <div className="empty-state">
              <h2>Run a training session</h2>
              <p>Pick a formula, adjust the reservoir, and click Run Training to visualize learning.</p>
            </div>
          )}

          {result && (
            <div className="grid">
              <div className="card wide">
                <h3>Loss curve</h3>
                <LineChart data={result.losses} height={180} />
              </div>
              <div className="card wide">
                <h3>Error vs time</h3>
                <LineChart data={result.errorByTime.map((v) => Math.abs(v))} height={180} color="#d65f5f" />
              </div>
              <div className="card wide">
                <h3>Ground truth vs LSM prediction</h3>
                <CurvePlot groundTruth={result.groundTruth} prediction={result.prediction} height={260} />
              </div>
              <div className="card">
                <h3>Reservoir spikes</h3>
                <Heatmap data={result.spikes} height={220} mode="spikes" />
              </div>
              <div className="card">
                <h3>Reservoir state</h3>
                <Heatmap data={result.reservoir} height={220} mode="state" />
              </div>
              <div className="card">
                <h3>Recurrent weights</h3>
                <Heatmap data={result.weights.recurrent} height={220} mode="weights" />
              </div>
              <div className="card">
                <h3>Input weights</h3>
                <Heatmap data={result.weights.input} height={220} mode="weights" />
              </div>
              <div className="card wide">
                <h3>Readout formula</h3>
                <p className="formula">{result.readoutFormula}</p>
                <div className="metrics">
                  <div>
                    <span className="metric-label">MSE</span>
                    <span className="metric-value">{result.metrics.mse.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="metric-label">Max error</span>
                    <span className="metric-value">{result.metrics.maxError.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
