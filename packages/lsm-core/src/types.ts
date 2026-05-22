export type CurveFamily =
  | "limacon-inner-loop"
  | "cardioid"
  | "limacon-dimpled"
  | "limacon-convex"
  | "rose"
  | "circle"
  | "lemniscate"
  | "conic";

export interface CurveParams {
  a: number;
  b?: number;
  n?: number;
  e?: number;
  l?: number;
  form?: string;
  trig: "cos" | "sin";
}

export interface CurveSample {
  theta: number[];
  r: number[];
  x: number[];
  y: number[];
  params: CurveParams;
}

export interface Dataset {
  sequences: number[][][];
  radialTargets: number[][];
  params: CurveParams[];
}

export interface LsmConfig {
  inputDim: number;
  numExcitatory: number;
  numInhibitory: number;
  connectivity: number;
  spectralRadius: number;
  decay: number;
  threshold: number;
  resetValue: number;
  inputScale: number;
  seed: number;
}

export interface TrainingConfig {
  formula: CurveFamily;
  timesteps: number;
  trainSamples: number;
  epochs: number;
  learningRate: number;
  neurons: number;
  connectivity: number;
  seed: number;
}

export interface TrainingResult {
  config: TrainingConfig;
  losses: number[];
  errorByTime: number[];
  theta: number[];
  groundTruth: CurveSample;
  prediction: CurveSample;
  spikes: number[][];
  reservoir: number[][];
  weights: {
    recurrent: number[][];
    input: number[][];
    readout: number[];
    bias: number;
  };
  readoutFormula: string;
  metrics: {
    mse: number;
    maxError: number;
  };
}
