import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { RWSResult, RWSInterpretation } from '@/types';

interface RWSChartProps {
  results: RWSResult[];
}

const INTERP_BAR_COLORS: Record<RWSInterpretation, string> = {
  normal: '#22c55e',
  intermediate: '#eab308',
  vulnerable: '#f97316',
  high_risk: '#ef4444',
};

export function RWSChart({ results }: RWSChartProps) {
  const data = useMemo(() => {
    return results.map((r) => ({
      beat: `Beat ${r.beatNumber}`,
      mldRws: parseFloat(r.mldRwsPct.toFixed(2)),
      avgRws: parseFloat(r.averageRwsPct.toFixed(2)),
      interpretation: r.interpretation,
    }));
  }, [results]);

  if (data.length === 0) return null;

  return (
    <div className="space-y-1">
      <label className="text-xs text-content-secondary">RWS per Beat (MLD)</label>
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, #333)" />
            <XAxis
              dataKey="beat"
              tick={{ fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
            />
            <YAxis
              tick={{ fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
              label={{ value: '%', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: 'var(--color-content-muted, #888)' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-surface-tertiary, #1a1a1a)',
                border: '1px solid var(--color-border, #333)',
                borderRadius: '4px',
                fontSize: 10,
              }}
              formatter={(v: number, name: string) => [
                `${v.toFixed(2)}%`,
                name === 'mldRws' ? 'MLD RWS' : 'Avg RWS',
              ]}
            />
            <ReferenceLine y={8} stroke="#eab308" strokeDasharray="3 3" />
            <ReferenceLine y={14} stroke="#ef4444" strokeDasharray="3 3" />
            <Bar dataKey="mldRws" radius={[2, 2, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={index} fill={INTERP_BAR_COLORS[entry.interpretation]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-3 text-[10px] text-content-muted flex-wrap">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-green-500" /> Normal (&lt;8%)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-yellow-500" /> Intermediate
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-orange-500" /> Vulnerable
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> High Risk (&gt;14%)
        </span>
      </div>
    </div>
  );
}
