import type { LsmConfig } from "./types";
import { Rng } from "./rng";

interface SparseMatrix {
  indices: number[][];
  weights: number[][];
}

function randNormal(rng: Rng): number {
  const u = Math.max(rng.next(), 1e-6);
  const v = Math.max(rng.next(), 1e-6);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function powerIteration(matrix: Float32Array, size: number, iters = 20): number {
  let vec = new Float32Array(size).fill(1 / size);
  let norm = 0;
  for (let iter = 0; iter < iters; iter += 1) {
    const next = new Float32Array(size);
    for (let row = 0; row < size; row += 1) {
      let sum = 0;
      const offset = row * size;
      for (let col = 0; col < size; col += 1) {
        sum += matrix[offset + col] * vec[col];
      }
      next[row] = sum;
    }
    norm = 0;
    for (let i = 0; i < size; i += 1) {
      norm += next[i] * next[i];
    }
    norm = Math.sqrt(norm);
    if (norm < 1e-6) return 0;
    for (let i = 0; i < size; i += 1) {
      vec[i] = next[i] / norm;
    }
  }
  return norm;
}

export class LiquidStateMachine {
  readonly inputDim: number;
  readonly numExc: number;
  readonly numInh: number;
  readonly numTotal: number;
  readonly connectivity: number;
  readonly spectralRadius: number;
  readonly decay: number;
  readonly threshold: number;
  readonly resetValue: number;

  private rng: Rng;
  private recurrentDense: Float32Array;
  private recurrentSparse: SparseMatrix;
  private inputWeights: Float32Array;

  voltage: Float32Array;
  spikes: Float32Array;

  constructor(config: LsmConfig) {
    this.inputDim = config.inputDim;
    this.numExc = config.numExcitatory;
    this.numInh = config.numInhibitory;
    this.numTotal = this.numExc + this.numInh;
    this.connectivity = config.connectivity;
    this.spectralRadius = config.spectralRadius;
    this.decay = config.decay;
    this.threshold = config.threshold;
    this.resetValue = config.resetValue;
    this.rng = new Rng(config.seed);

    this.recurrentDense = this.initRecurrent();
    this.recurrentSparse = this.buildSparse(this.recurrentDense);
    this.inputWeights = this.initInput(config.inputScale);
    this.voltage = new Float32Array(this.numTotal);
    this.spikes = new Float32Array(this.numTotal);
  }

  resetState(): void {
    this.voltage.fill(0);
    this.spikes.fill(0);
  }

  step(inputVec: [number, number]): Float32Array {
    const nextSpikes = this.spikes;
    for (let neuron = 0; neuron < this.numTotal; neuron += 1) {
      let recurrentSum = 0;
      const idxs = this.recurrentSparse.indices[neuron];
      const wts = this.recurrentSparse.weights[neuron];
      for (let k = 0; k < idxs.length; k += 1) {
        recurrentSum += wts[k] * nextSpikes[idxs[k]];
      }
      const inputOffset = neuron * this.inputDim;
      const inputCurrent = this.inputWeights[inputOffset] * inputVec[0] + this.inputWeights[inputOffset + 1] * inputVec[1];
      const voltage = this.decay * this.voltage[neuron] + recurrentSum + inputCurrent;
      this.voltage[neuron] = voltage;
      if (voltage >= this.threshold) {
        nextSpikes[neuron] = 1;
        this.voltage[neuron] = this.resetValue;
      } else {
        nextSpikes[neuron] = 0;
      }
    }
    return nextSpikes;
  }

  getRecurrentMatrix(): number[][] {
    const matrix: number[][] = [];
    for (let row = 0; row < this.numTotal; row += 1) {
      const offset = row * this.numTotal;
      const rowData: number[] = [];
      for (let col = 0; col < this.numTotal; col += 1) {
        rowData.push(this.recurrentDense[offset + col]);
      }
      matrix.push(rowData);
    }
    return matrix;
  }

  getInputMatrix(): number[][] {
    const matrix: number[][] = [];
    for (let row = 0; row < this.numTotal; row += 1) {
      const offset = row * this.inputDim;
      matrix.push([this.inputWeights[offset], this.inputWeights[offset + 1]]);
    }
    return matrix;
  }

  private initInput(scale: number): Float32Array {
    const weights = new Float32Array(this.numTotal * this.inputDim);
    for (let i = 0; i < weights.length; i += 1) {
      weights[i] = randNormal(this.rng) * scale;
    }
    return weights;
  }

  private initRecurrent(): Float32Array {
    const size = this.numTotal * this.numTotal;
    const weights = new Float32Array(size);
    const baseScale = 1 / Math.sqrt(this.numTotal);
    for (let row = 0; row < this.numTotal; row += 1) {
      for (let col = 0; col < this.numTotal; col += 1) {
        if (row === col) continue;
        if (this.rng.next() > this.connectivity) continue;
        const w = randNormal(this.rng) * baseScale;
        const idx = row * this.numTotal + col;
        weights[idx] = w;
      }
    }
    for (let col = 0; col < this.numTotal; col += 1) {
      const isExc = col < this.numExc;
      for (let row = 0; row < this.numTotal; row += 1) {
        const idx = row * this.numTotal + col;
        if (weights[idx] === 0) continue;
        weights[idx] = isExc ? Math.abs(weights[idx]) : -Math.abs(weights[idx]);
      }
    }
    const radius = powerIteration(weights, this.numTotal);
    if (radius > 1e-6) {
      const scale = this.spectralRadius / radius;
      for (let i = 0; i < weights.length; i += 1) {
        weights[i] *= scale;
      }
    }
    return weights;
  }

  private buildSparse(matrix: Float32Array): SparseMatrix {
    const indices: number[][] = [];
    const weights: number[][] = [];
    for (let row = 0; row < this.numTotal; row += 1) {
      const rowIdx: number[] = [];
      const rowWt: number[] = [];
      const offset = row * this.numTotal;
      for (let col = 0; col < this.numTotal; col += 1) {
        const value = matrix[offset + col];
        if (value !== 0) {
          rowIdx.push(col);
          rowWt.push(value);
        }
      }
      indices.push(rowIdx);
      weights.push(rowWt);
    }
    return { indices, weights };
  }
}
