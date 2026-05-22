interface LineChartProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}

function LineChart({ data, width = 520, height = 180, color = "#7c5cff" }: LineChartProps) {
  if (!data.length) return null;
  const padding = 24;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((value, idx) => {
    const x = padding + (idx / Math.max(data.length - 1, 1)) * (width - padding * 2);
    const y = padding + (1 - (value - min) / range) * (height - padding * 2);
    return `${x},${y}`;
  });

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={points.join(" ")}
      />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#2c3140" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#2c3140" />
    </svg>
  );
}

export default LineChart;
