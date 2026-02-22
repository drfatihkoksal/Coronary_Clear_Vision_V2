import { useRWSStore } from '@/stores/rwsStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useTimingStore } from '@/stores/timingStore';
import { RWSChart } from '@/components/charts/RWSChart';
import type { RWSInterpretation } from '@/types';

const OUTLIER_METHODS: { value: 'none' | 'hampel' | 'double_hampel'; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'hampel', label: 'Hampel' },
  { value: 'double_hampel', label: 'Double' },
];

const VESSELS = ['LAD', 'LCx', 'RCA', 'LM', 'other'] as const;

const INTERP_COLORS: Record<RWSInterpretation, string> = {
  normal: 'bg-green-500/20 text-green-400 border-green-500/30',
  intermediate: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  vulnerable: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  high_risk: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const INTERP_LABELS: Record<RWSInterpretation, string> = {
  normal: 'Normal',
  intermediate: 'Intermediate',
  vulnerable: 'Vulnerable',
  high_risk: 'High Risk',
};

export function RWSPanel() {
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const goToFrame = usePlayerStore((s) => s.goToFrame);
  const startFrame = useRWSStore((s) => s.startFrame);
  const endFrame = useRWSStore((s) => s.endFrame);
  const outlierMethod = useRWSStore((s) => s.outlierMethod);
  const vessel = useRWSStore((s) => s.vessel);
  const isCalculating = useRWSStore((s) => s.isCalculating);
  const error = useRWSStore((s) => s.error);
  const results = useRWSStore((s) => s.results);
  const summary = useRWSStore((s) => s.summary);
  const calculate = useRWSStore((s) => s.calculate);
  const setRange = useRWSStore((s) => s.setRange);
  const setOutlierMethod = useRWSStore((s) => s.setOutlierMethod);
  const setVessel = useRWSStore((s) => s.setVessel);
  const removeResult = useRWSStore((s) => s.removeResult);
  const clearResults = useRWSStore((s) => s.clearResults);

  const beatBoundaries = useTimingStore((s) => s.beatBoundaries);
  const selectedBeat = useRWSStore((s) => s.selectedBeat);

  const canCalculate = startFrame != null && endFrame != null && startFrame < endFrame && !isCalculating;

  const handleBeatClick = (beat: { beatNumber: number; startFrame: number; endFrame: number }) => {
    setRange(beat.startFrame, beat.endFrame, beat.beatNumber);
    goToFrame(beat.startFrame);
  };

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">RWS Analysis</h3>

      {/* Beat selector */}
      {beatBoundaries.length > 0 && (
        <div className="space-y-1">
          <label className="text-xs text-content-secondary">Beat</label>
          <div className="flex flex-wrap gap-1">
            {beatBoundaries.map((beat) => (
              <button
                key={beat.beatNumber}
                onClick={() => handleBeatClick(beat)}
                className={`px-2 py-1 text-xs rounded border transition-colors ${
                  selectedBeat === beat.beatNumber
                    ? 'border-brand bg-brand/10 text-brand'
                    : 'border-border text-content-secondary hover:border-content-muted'
                }`}
              >
                B{beat.beatNumber}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Frame range */}
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <label className="text-xs text-content-secondary">Start Frame</label>
          <input
            type="number"
            min={0}
            max={totalFrames - 1}
            value={startFrame ?? ''}
            onChange={(e) => setRange(Number(e.target.value), endFrame ?? 0, undefined)}
            className="w-full px-2 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-content-secondary">End Frame</label>
          <input
            type="number"
            min={0}
            max={totalFrames - 1}
            value={endFrame ?? ''}
            onChange={(e) => setRange(startFrame ?? 0, Number(e.target.value), undefined)}
            className="w-full px-2 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
          />
        </div>
      </div>

      {/* Outlier method */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Outlier Filter</label>
        <div className="grid grid-cols-3 gap-1">
          {OUTLIER_METHODS.map((m) => (
            <button
              key={m.value}
              onClick={() => setOutlierMethod(m.value)}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                outlierMethod === m.value
                  ? 'border-brand bg-brand/10 text-brand'
                  : 'border-border text-content-secondary hover:border-content-muted'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Vessel selector */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Vessel</label>
        <select
          value={vessel ?? ''}
          onChange={(e) => setVessel(e.target.value || null)}
          className="w-full px-2 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
        >
          <option value="">-- Select --</option>
          {VESSELS.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {/* Calculate button */}
      <button
        onClick={calculate}
        disabled={!canCalculate}
        className="w-full py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isCalculating ? 'Calculating...' : 'Calculate RWS'}
      </button>

      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Summary */}
      {summary && (
        <div className={`p-2 rounded border text-xs space-y-1 ${INTERP_COLORS[summary.interpretation]}`}>
          <div className="font-semibold">{INTERP_LABELS[summary.interpretation]}</div>
          <Row label="Mean RWS" value={`${summary.meanRwsPct.toFixed(2)}%`} />
          <Row label="Median RWS" value={`${summary.medianRwsPct.toFixed(2)}%`} />
          <Row label="Beats" value={String(summary.numBeats)} />
        </div>
      )}

      {/* Result cards */}
      {results.map((r, i) => (
        <div key={i} className={`p-2 rounded border text-xs space-y-1 ${INTERP_COLORS[r.interpretation]}`}>
          <div className="flex justify-between items-center">
            <span className="font-semibold">Beat {r.beatNumber}</span>
            <button
              onClick={() => removeResult(i)}
              className="text-content-muted hover:text-red-400 text-xs"
            >
              Remove
            </button>
          </div>
          <Row label="MLD RWS" value={`${r.mldRwsPct.toFixed(2)}%`} />
          <Row label="Proximal" value={`${r.proximalRwsPct.toFixed(2)}%`} />
          <Row label="Distal" value={`${r.distalRwsPct.toFixed(2)}%`} />
          <Row label="Average" value={`${r.averageRwsPct.toFixed(2)}%`} />
          <Row label="Frames" value={`${r.startFrame} - ${r.endFrame}`} />
          <Row label="Filter" value={r.outlierMethod} />
          {r.vessel && <Row label="Vessel" value={r.vessel} />}
        </div>
      ))}

      {/* Chart */}
      {results.length > 0 && <RWSChart results={results} />}

      {/* Clear button */}
      {results.length > 0 && (
        <button
          onClick={clearResults}
          className="w-full py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors"
        >
          Clear All Results
        </button>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="opacity-80">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
