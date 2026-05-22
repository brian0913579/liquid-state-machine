/// <reference lib="webworker" />

import { estimateOps, trainModel } from "@lsm-core";
import type { TrainingConfig } from "@lsm-core";

const MAX_OPS = 2.5e7;

self.onmessage = (event: MessageEvent<{ type: string; payload: TrainingConfig }>) => {
  const { type, payload } = event.data;
  if (type !== "train") return;
  try {
    const ops = estimateOps(payload);
    if (ops > MAX_OPS) {
      self.postMessage({ type: "tooHeavy", payload: ops });
      return;
    }
    const result = trainModel(payload);
    self.postMessage({ type: "result", payload: result });
  } catch (error) {
    self.postMessage({ type: "error", payload: (error as Error).message });
  }
};
