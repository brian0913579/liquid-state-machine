import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { FORMULA_ORDER } from "../../../packages/lsm-core/src/formulas";
import { trainModel } from "../../../packages/lsm-core/src/training";
import { TrainingConfig } from "../../../packages/lsm-core/src/types";

const ROOT = process.cwd();
const OUTPUT_DIR = path.join(ROOT, "public", "assets", "precomputed");

const BASE_CONFIG: TrainingConfig = {
  formula: "cardioid",
  timesteps: 80,
  trainSamples: 50,
  epochs: 24,
  learningRate: 0.04,
  neurons: 48,
  connectivity: 0.08,
  seed: 77,
};

async function main() {
  await mkdir(OUTPUT_DIR, { recursive: true });
  for (const formula of FORMULA_ORDER) {
    const result = trainModel({ ...BASE_CONFIG, formula });
    const outPath = path.join(OUTPUT_DIR, `${formula}.json`);
    await writeFile(outPath, JSON.stringify(result));
    console.log(`Saved ${outPath}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
