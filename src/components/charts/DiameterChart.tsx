import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot,
  ResponsiveContainer,
} from 'recharts';

interface DiameterChartProps {
  distancesMm: number[];
  diametersMm: number[];
  mldIndex?: number;
  proxRefIndex?: number;
  distRefIndex?: number;
}

export function DiameterChart({
  distancesMm,
  diametersMm,
  mldIndex,
  proxRefIndex,
  distRefIndex,
}: DiameterChartProps) {
  const data = useMemo(() => {
    return distancesMm.map((d, i) => ({
      distance: parseFloat(d.toFixed(2)),
      diameter: parseFloat(diametersMm[i].toFixed(3)),
    }));
  }, [distancesMm, diametersMm]);

  if (data.length === 0) return null;

  const mldPoint = mldIndex != null && data[mldIndex] ? data[mldIndex] : null;
  const proxPoint = proxRefIndex != null && data[proxRefIndex] ? data[proxRefIndex] : null;
  const distPoint = distRefIndex != null && data[distRefIndex] ? data[distRefIndex] : null;

  return (
    <div className="space-y-1">
      <label className="text-xs text-content-secondary">Diameter Profile</label>
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, #333)" />
            <XAxis
              dataKey="distance"
              tick={{ fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
              label={{ value: 'mm', position: 'insideBottomRight', offset: -2, fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
            />
            <YAxis
              tick={{ fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
              label={{ value: 'mm', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-surface-tertiary, #1a1a1a)',
                border: '1px solid var(--color-border, #333)',
                borderRadius: '4px',
                fontSize: 10,
              }}
              labelFormatter={(v) => `Distance: ${v} mm`}
              formatter={(v: number) => [`${v.toFixed(3)} mm`, 'Diameter']}
            />
            <Line
              type="monotone"
              dataKey="diameter"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3 }}
            />
            {mldPoint && (
              <ReferenceDot
                x={mldPoint.distance}
                y={mldPoint.diameter}
                r={4}
                fill="#ef4444"
                stroke="#ef4444"
              />
            )}
            {proxPoint && (
              <ReferenceDot
                x={proxPoint.distance}
                y={proxPoint.diameter}
                r={3}
                fill="#22c55e"
                stroke="#22c55e"
              />
            )}
            {distPoint && (
              <ReferenceDot
                x={distPoint.distance}
                y={distPoint.diameter}
                r={3}
                fill="#eab308"
                stroke="#eab308"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-3 text-[10px] text-content-muted">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> MLD
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-green-500" /> Prox Ref
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-yellow-500" /> Dist Ref
        </span>
      </div>
    </div>
  );
}
