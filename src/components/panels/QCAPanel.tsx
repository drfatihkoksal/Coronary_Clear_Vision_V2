import { useAnalysisStore } from '@/stores/analysisStore';
import { usePlayerStore } from '@/stores/playerStore';
import { DiameterChart } from '@/components/charts/DiameterChart';
import type { QCAMetrics } from '@/types';

const METHODS: { value: QCAMetrics['method']; label: string }[] = [
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'parabolic', label: 'Parabolic' },
  { value: 'threshold', label: 'Threshold' },
];

export function QCAPanel() {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const frameData = useAnalysisStore((s) => s.frameData);
  const qcaResults = useAnalysisStore((s) => s.qcaResults);
  const computeQCA = useAnalysisStore((s) => s.computeQCA);
  const isComputingQCA = useAnalysisStore((s) => s.isComputingQCA);
  const qcaError = useAnalysisStore((s) => s.qcaError);

  const hasSegmentation = frameData.has(currentFrame);
  const qca = qcaResults.get(currentFrame) as QCAMetrics | undefined;

  const handleMethodChange = (method: QCAMetrics['method']) => {
    if (hasSegmentation) {
      computeQCA(currentFrame, method);
    }
  };

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">QCA Analysis</h3>

      {/* Method selection */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Method</label>
        <div className="grid grid-cols-3 gap-1">
          {METHODS.map((m) => (
            <button
              key={m.value}
              onClick={() => handleMethodChange(m.value)}
              disabled={!hasSegmentation || isComputingQCA}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                qca?.method === m.value
                  ? 'border-brand bg-brand/10 text-brand'
                  : 'border-border text-content-secondary hover:border-content-muted'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {isComputingQCA && (
        <p className="text-xs text-content-muted">Computing QCA...</p>
      )}

      {!hasSegmentation && (
        <p className="text-xs text-content-muted">Segment the frame first to see QCA results.</p>
      )}

      {qcaError && <p className="text-xs text-red-400">{qcaError}</p>}

      {/* Results */}
      {qca && (
        <>
          <div className="p-2 rounded bg-surface-tertiary text-xs space-y-1">
            <Row label="MLD" value={`${qca.mldMm.toFixed(2)} mm`} />
            <Row label="DS%" value={`${qca.diameterStenosisPct.toFixed(1)}%`} highlight={qca.diameterStenosisPct > 50} />
            <Row label="Proximal Ref" value={`${(qca.proximalRefMm ?? 0).toFixed(2)} mm`} />
            <Row label="Distal Ref" value={`${(qca.distalRefMm ?? 0).toFixed(2)} mm`} />
            <Row label="Interp. Ref" value={`${(qca.interpolatedRefMm ?? 0).toFixed(2)} mm`} />
            {qca.lesionLengthMm != null && (
              <Row label="Lesion Length" value={`${qca.lesionLengthMm.toFixed(2)} mm`} />
            )}
            <Row label="Vessel Length" value={`${(qca.vesselLengthMm ?? 0).toFixed(2)} mm`} />
            <Row label="Spacing" value={`${qca.pixelSpacingMm.toFixed(3)} mm/px`} />
            <Row label="Points" value={`${qca.numPoints}`} />
          </div>

          {/* Diameter chart */}
          <DiameterChart
            distancesMm={qca.distancesMm}
            diametersMm={qca.diameterProfileMm}
            mldIndex={qca.mldIndex}
            proxRefIndex={qca.proximalRefIndex}
            distRefIndex={qca.distalRefIndex}
          />
        </>
      )}
    </div>
  );
}

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-content-secondary">{label}</span>
      <span className={highlight ? 'text-red-400 font-medium' : 'text-content-primary'}>{value}</span>
    </div>
  );
}
