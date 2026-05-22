import { FORMULA_LABELS, FORMULA_ORDER } from "@lsm-core";
import type { CurveFamily, TrainingConfig } from "@lsm-core";
import "./ControlPanel.css";

interface ControlPanelProps {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
  onRun: () => void;
  disabled: boolean;
  estimate: number;
  isHeavy: boolean;
}

const formatOps = (value: number) => {
  if (value > 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value > 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value > 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${Math.round(value)}`;
};

function ControlPanel({ config, onChange, onRun, disabled, estimate, isHeavy }: ControlPanelProps) {
  const update = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  const handleFormula = (value: string) => {
    update("formula", value as CurveFamily);
  };

  return (
    <aside className="panel">
      <h2>Controls</h2>
      <div className="field">
        <label>Formula</label>
        <select value={config.formula} onChange={(e) => handleFormula(e.target.value)}>
          {FORMULA_ORDER.map((family) => (
            <option key={family} value={family}>
              {FORMULA_LABELS[family]}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Timesteps</label>
        <input
          type="range"
          min={48}
          max={192}
          step={8}
          value={config.timesteps}
          onChange={(e) => update("timesteps", Number(e.target.value))}
        />
        <span>{config.timesteps}</span>
      </div>

      <div className="field">
        <label>Neurons</label>
        <input
          type="range"
          min={24}
          max={120}
          step={4}
          value={config.neurons}
          onChange={(e) => update("neurons", Number(e.target.value))}
        />
        <span>{config.neurons}</span>
      </div>

      <div className="field">
        <label>Training samples</label>
        <input
          type="range"
          min={20}
          max={140}
          step={10}
          value={config.trainSamples}
          onChange={(e) => update("trainSamples", Number(e.target.value))}
        />
        <span>{config.trainSamples}</span>
      </div>

      <div className="field">
        <label>Epochs (training time)</label>
        <input
          type="range"
          min={8}
          max={80}
          step={4}
          value={config.epochs}
          onChange={(e) => update("epochs", Number(e.target.value))}
        />
        <span>{config.epochs}</span>
      </div>

      <div className="field">
        <label>Learning rate</label>
        <input
          type="range"
          min={0.005}
          max={0.08}
          step={0.005}
          value={config.learningRate}
          onChange={(e) => update("learningRate", Number(e.target.value))}
        />
        <span>{config.learningRate.toFixed(3)}</span>
      </div>

      <div className="field">
        <label>Connectivity</label>
        <input
          type="range"
          min={0.02}
          max={0.2}
          step={0.01}
          value={config.connectivity}
          onChange={(e) => update("connectivity", Number(e.target.value))}
        />
        <span>{config.connectivity.toFixed(2)}</span>
      </div>

      <div className="estimate">
        <div>
          <span className="metric-label">Compute estimate</span>
          <span className={`metric-value ${isHeavy ? "warn" : ""}`}>{formatOps(estimate)} ops</span>
        </div>
        <p className="hint">
          {isHeavy
            ? "Over the 10s target. Consider fewer neurons or epochs."
            : "Within the 10s target for live training."}
        </p>
      </div>

      <button className="run" type="button" onClick={onRun} disabled={disabled}>
        {disabled ? "Training…" : "Run Training"}
      </button>
    </aside>
  );
}

export default ControlPanel;
