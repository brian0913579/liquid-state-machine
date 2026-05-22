import { buildDataset, buildTheta, generateCurveSample } from "./formulas";
import { LiquidStateMachine } from "./lsm";
import { Rng } from "./rng";
import type { CurveSample, LsmConfig, TrainingConfig, TrainingResult } from "./types";

const DEFAULTS = {
  spectralRadius: 0.95,
  decay: 0.9,
  threshold: 1.0,
  resetValue: 0,
  inputScale: 0.8,
};

function makeSampleFromR(theta: number[], r: number[], params: CurveSample["params"]): CurveSample {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < theta.length; i += 1) {
    x.push(r[i] * Math.cos(theta[i]));
    y.push(r[i] * Math.sin(theta[i]));
  }
  return { theta, r, x, y, params };
}

function summarizeReadout(weights: Float32Array, bias: number): string {
  const pairs = Array.from(weights).map((value, idx) => ({ idx, value }));
  pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const top = pairs.slice(0, 6).map((item) => `w${item.idx}=${item.value.toFixed(3)}`).join(", ");
  return `r̂(θ) = Σ wᵢ·spikeᵢ + b, b=${bias.toFixed(3)} (top: ${top})`;
}

export function estimateOps(config: TrainingConfig): number {
  const neurons = config.neurons;
  const avgConnections = Math.max(1, Math.floor(neurons * config.connectivity));
  return config.timesteps * config.trainSamples * config.epochs * avgConnections * neurons;
}

export function trainModel(config: TrainingConfig): TrainingResult {
  const rng = new Rng(config.seed);
  const theta = buildTheta(config.timesteps);
  const dataset = buildDataset(config.formula, theta, config.trainSamples, rng);
  const testSample = generateCurveSample(config.formula, theta, rng);

  const numExc = Math.max(1, Math.floor(config.neurons * 0.7));
  const numInh = Math.max(1, config.neurons - numExc);
  const lsmConfig: LsmConfig = {
    inputDim: 2,
    numExcitatory: numExc,
    numInhibitory: numInh,
    connectivity: config.connectivity,
    spectralRadius: DEFAULTS.spectralRadius,
    decay: DEFAULTS.decay,
    threshold: DEFAULTS.threshold,
    resetValue: DEFAULTS.resetValue,
    inputScale: DEFAULTS.inputScale,
    seed: config.seed + 101,
  };

  const lsm = new LiquidStateMachine(lsmConfig);
  const weights = new Float32Array(lsm.numTotal);
  let bias = 0;
  const losses: number[] = [];

  for (let epoch = 0; epoch < config.epochs; epoch += 1) {
    let lossSum = 0;
    let count = 0;
    for (let s = 0; s < dataset.sequences.length; s += 1) {
      lsm.resetState();
      const sequence = dataset.sequences[s];
      const radial = dataset.radialTargets[s];
      for (let t = 0; t < sequence.length; t += 1) {
        const spikes = lsm.step([sequence[t][0], sequence[t][1]]);
        let pred = bias;
        for (let i = 0; i < spikes.length; i += 1) {
          pred += weights[i] * spikes[i];
        }
        const err = pred - radial[t];
        lossSum += err * err;
        count += 1;
        const grad = 2 * err;
        for (let i = 0; i < spikes.length; i += 1) {
          weights[i] -= config.learningRate * grad * spikes[i];
        }
        bias -= config.learningRate * grad;
      }
    }
    losses.push(lossSum / Math.max(count, 1));
  }

  lsm.resetState();
  const spikesTrace: number[][] = [];
  const reservoirTrace: number[][] = [];
  const predR: number[] = [];
  for (let t = 0; t < testSample.x.length; t += 1) {
    const spikes = lsm.step([testSample.x[t], testSample.y[t]]);
    spikesTrace.push(Array.from(spikes));
    reservoirTrace.push(Array.from(lsm.voltage));
    let pred = bias;
    for (let i = 0; i < spikes.length; i += 1) {
      pred += weights[i] * spikes[i];
    }
    predR.push(pred);
  }

  const prediction = makeSampleFromR(theta, predR, testSample.params);
  const errorByTime = predR.map((value, idx) => value - testSample.r[idx]);
  const mse = errorByTime.reduce((sum, value) => sum + value * value, 0) / Math.max(errorByTime.length, 1);
  const maxError = errorByTime.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

  return {
    config,
    losses,
    errorByTime,
    theta,
    groundTruth: testSample,
    prediction,
    spikes: spikesTrace,
    reservoir: reservoirTrace,
    weights: {
      recurrent: lsm.getRecurrentMatrix(),
      input: lsm.getInputMatrix(),
      readout: Array.from(weights),
      bias,
    },
    readoutFormula: summarizeReadout(weights, bias),
    metrics: { mse, maxError },
  };
}
