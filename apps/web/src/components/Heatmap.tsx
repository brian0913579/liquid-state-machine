import { useEffect, useRef } from "react";

interface HeatmapProps {
  data: number[][];
  height?: number;
  mode: "weights" | "spikes" | "state";
}

const colorStops = {
  weights: [
    [0, 68, 1, 84],
    [58, 82, 139],
    [255, 255, 255],
    [253, 231, 37],
  ],
  spikes: [
    [8, 10, 15],
    [60, 82, 140],
    [199, 233, 255],
  ],
  state: [
    [10, 12, 17],
    [68, 98, 152],
    [201, 238, 255],
  ],
};

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function colorFor(value: number, min: number, max: number, mode: HeatmapProps["mode"]) {
  const range = max - min || 1;
  const t = (value - min) / range;
  const stops = colorStops[mode];
  if (stops.length === 4) {
    if (t < 0.5) {
      const local = t / 0.5;
      return stops[0].map((c, idx) => lerp(c, stops[1][idx], local));
    }
    const local = (t - 0.5) / 0.5;
    return stops[2].map((c, idx) => lerp(c, stops[3][idx], local));
  }
  const local = t;
  return stops[0].map((c, idx) => lerp(c, stops[2][idx], local));
}

function Heatmap({ data, height = 200, mode }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rows = data.length;
    const cols = data[0]?.length ?? 0;
    if (!rows || !cols) return;

    const min = Math.min(...data.flat());
    const max = Math.max(...data.flat());

    const width = canvas.clientWidth || 320;
    const heightPx = canvas.clientHeight || height;
    canvas.width = width;
    canvas.height = heightPx;

    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(cols, rows);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const idx = (y * cols + x) * 4;
        const [r, g, b] = colorFor(data[y][x], min, max, mode);
        image.data[idx] = r;
        image.data[idx + 1] = g;
        image.data[idx + 2] = b;
        image.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);

    const displayCtx = canvas.getContext("2d");
    if (!displayCtx) return;
    displayCtx.imageSmoothingEnabled = false;
    displayCtx.clearRect(0, 0, width, heightPx);
    displayCtx.drawImage(offscreen, 0, 0, width, heightPx);
  }, [data, mode, height]);

  return <canvas ref={canvasRef} className="heatmap" style={{ height }} />;
}

export default Heatmap;
