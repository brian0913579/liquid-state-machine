import type { CurveFamily, CurveParams, CurveSample, Dataset } from "./types";
import { Rng } from "./rng";

const TWO_PI = Math.PI * 2;
const SCALE_BANDS: Array<[number, number]> = [
  [0.8, 1.0],
  [1.2, 1.6],
];
const ECC_BANDS: Array<[number, number]> = [
  [0.2, 0.45],
  [0.65, 0.9],
];

export const FORMULA_LABELS: Record<CurveFamily, string> = {
  "limacon-inner-loop": "Limacon (inner loop)",
  cardioid: "Cardioid",
  "limacon-dimpled": "Limacon (dimpled)",
  "limacon-convex": "Limacon (convex)",
  rose: "Rose curve",
  circle: "Circle",
  lemniscate: "Lemniscate",
  conic: "Conic section",
};

export const FORMULA_ORDER: CurveFamily[] = [
  "limacon-inner-loop",
  "cardioid",
  "limacon-dimpled",
  "limacon-convex",
  "rose",
  "circle",
  "lemniscate",
  "conic",
];

const LIMACON_RANGES: Record<CurveFamily, [number, number]> = {
  "limacon-inner-loop": [0.2, 0.8],
  cardioid: [1.0, 1.0],
  "limacon-dimpled": [1.15, 1.9],
  "limacon-convex": [2.1, 3.2],
  rose: [0, 0],
  circle: [0, 0],
  lemniscate: [0, 0],
  conic: [0, 0],
};

function sampleFromBands(rng: Rng, bands: Array<[number, number]>): number {
  const [low, high] = rng.pick(bands);
  return rng.float(low, high);
}

function sampleScale(rng: Rng): number {
  return sampleFromBands(rng, SCALE_BANDS);
}

function sampleEccentricity(rng: Rng): number {
  return sampleFromBands(rng, ECC_BANDS);
}

export function buildTheta(steps: number): number[] {
  const theta: number[] = [];
  for (let i = 0; i < steps; i += 1) {
    theta.push((i / steps) * TWO_PI);
  }
  return theta;
}

export function normalizeCurve(theta: number[], r: number[]): CurveSample {
  const x: number[] = [];
  const y: number[] = [];
  let maxRadius = 0;
  for (let i = 0; i < theta.length; i += 1) {
    const xi = r[i] * Math.cos(theta[i]);
    const yi = r[i] * Math.sin(theta[i]);
    x.push(xi);
    y.push(yi);
    const rad = Math.sqrt(xi * xi + yi * yi);
    if (rad > maxRadius) maxRadius = rad;
  }
  const scale = maxRadius > 1e-6 ? maxRadius : 1;
  const xNorm = x.map((v) => v / scale);
  const yNorm = y.map((v) => v / scale);
  const rNorm = r.map((v) => v / scale);
  return {
    theta,
    r: rNorm,
    x: xNorm,
    y: yNorm,
    params: { a: 0, trig: "cos" },
  };
}

export function generateCurveSample(family: CurveFamily, theta: number[], rng: Rng): CurveSample {
  const trig = rng.boolean() ? "cos" : "sin";
  const trigFn = trig === "cos" ? Math.cos : Math.sin;
  const r: number[] = [];
  const params: CurveParams = { a: 0, trig };

  if (
    family === "limacon-inner-loop" ||
    family === "limacon-dimpled" ||
    family === "limacon-convex" ||
    family === "cardioid"
  ) {
    const scale = sampleScale(rng);
    const [minRatio, maxRatio] = LIMACON_RANGES[family];
    const ratio = minRatio === maxRatio ? minRatio : rng.float(minRatio, maxRatio);
    const a = ratio * scale;
    const b = scale;
    params.a = a;
    params.b = b;
    for (let i = 0; i < theta.length; i += 1) {
      const core = trigFn(theta[i]);
      if (family === "cardioid") {
        r.push(a * (1 + core));
      } else {
        r.push(a + b * core);
      }
    }
  } else if (family === "rose") {
    const a = sampleScale(rng);
    const n = rng.pick([2, 3, 4, 5, 6]);
    params.a = a;
    params.n = n;
    for (let i = 0; i < theta.length; i += 1) {
      r.push(a * trigFn(n * theta[i]));
    }
  } else if (family === "circle") {
    const a = sampleScale(rng);
    const form = rng.pick(["r = a", "r = a cos(θ)", "r = a sin(θ)"]);
    params.a = a;
    params.form = form;
    for (let i = 0; i < theta.length; i += 1) {
      if (form === "r = a") {
        r.push(a);
      } else if (form === "r = a cos(θ)") {
        r.push(a * Math.cos(theta[i]));
      } else {
        r.push(a * Math.sin(theta[i]));
      }
    }
  } else if (family === "lemniscate") {
    const a = sampleScale(rng);
    const coreFn = trigFn === Math.cos ? Math.cos : Math.sin;
    params.a = a;
    params.n = 2;
    for (let i = 0; i < theta.length; i += 1) {
      const core = coreFn(2 * theta[i]);
      const val = Math.sign(core) * a * Math.sqrt(Math.abs(core));
      r.push(val);
    }
  } else if (family === "conic") {
    const l = sampleScale(rng);
    const e = sampleEccentricity(rng);
    const sign = rng.boolean() ? 1 : -1;
    const trigLocal = trig === "cos" ? Math.cos : Math.sin;
    params.l = l;
    params.e = e;
    params.form = `${sign > 0 ? "+" : "-"}${trig === "cos" ? "cos" : "sin"}`;
    for (let i = 0; i < theta.length; i += 1) {
      const denom = 1 + sign * e * trigLocal(theta[i]);
      const safe = Math.abs(denom) < 0.15 ? Math.sign(denom || 1) * 0.15 : denom;
      r.push(l / safe);
    }
  }

  const normalized = normalizeCurve(theta, r);
  return {
    ...normalized,
    params,
  };
}

export function buildDataset(
  family: CurveFamily,
  theta: number[],
  numSamples: number,
  rng: Rng,
): Dataset {
  const sequences: number[][][] = [];
  const radialTargets: number[][] = [];
  const params: CurveParams[] = [];
  for (let i = 0; i < numSamples; i += 1) {
    const sample = generateCurveSample(family, theta, rng);
    sequences.push(sample.theta.map((_, idx) => [sample.x[idx], sample.y[idx]]));
    radialTargets.push(sample.r);
    params.push(sample.params);
  }
  return { sequences, radialTargets, params };
}
