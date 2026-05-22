import type { CurveSample } from "@lsm-core";

interface CurvePlotProps {
  groundTruth: CurveSample;
  prediction: CurveSample;
  height?: number;
}

function CurvePlot({ groundTruth, prediction, height = 240 }: CurvePlotProps) {
  const width = 520;
  const padding = 20;
  const xs = groundTruth.x.concat(prediction.x);
  const ys = groundTruth.y.concat(prediction.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scaleX = (value: number) =>
    padding + ((value - minX) / Math.max(maxX - minX, 1e-6)) * (width - padding * 2);
  const scaleY = (value: number) =>
    padding + (1 - (value - minY) / Math.max(maxY - minY, 1e-6)) * (height - padding * 2);

  const groundPath = groundTruth.x
    .map((x, idx) => `${scaleX(x)},${scaleY(groundTruth.y[idx])}`)
    .join(" ");
  const predPath = prediction.x
    .map((x, idx) => `${scaleX(x)},${scaleY(prediction.y[idx])}`)
    .join(" ");

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        points={groundPath}
        fill="none"
        stroke="#e2e8f0"
        strokeWidth="2.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <polyline
        points={predPath}
        fill="none"
        stroke="#7c5cff"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default CurvePlot;
