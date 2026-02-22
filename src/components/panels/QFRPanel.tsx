import { useQFRStore, type QFRMode } from '@/stores/qfrStore';

const QFR_MODES: { value: QFRMode; label: string; desc: string }[] = [
  { value: 'fQFR', label: 'fQFR', desc: 'Fixed flow' },
  { value: 'cQFR', label: 'cQFR', desc: 'Contrast flow' },
  { value: 'aQFR', label: 'aQFR', desc: 'Adenosine' },
];

export function QFRPanel() {
  const isQfrMode = useQFRStore((s) => s.isQfrMode);
  const enableQfrMode = useQFRStore((s) => s.enableQfrMode);
  const disableQfrMode = useQFRStore((s) => s.disableQfrMode);
  const mode = useQFRStore((s) => s.mode);
  const kt = useQFRStore((s) => s.kt);
  const error = useQFRStore((s) => s.error);
  const qfrResult = useQFRStore((s) => s.qfrResult);
  const angularSeparation = useQFRStore((s) => s.angularSeparation);
  const reconstructionInfo = useQFRStore((s) => s.reconstructionInfo);
  const projection1 = useQFRStore((s) => s.projection1);
  const projection2 = useQFRStore((s) => s.projection2);
  const setMode = useQFRStore((s) => s.setMode);
  const setKt = useQFRStore((s) => s.setKt);
  const clearQfr = useQFRStore((s) => s.clearQfr);

  const qfrColor = (val: number) => {
    if (val >= 0.9) return 'text-green-400';
    if (val >= 0.8) return 'text-yellow-400';
    if (val >= 0.75) return 'text-orange-400';
    return 'text-red-400';
  };

  const angularWarning = angularSeparation !== null && (angularSeparation < 25 || angularSeparation > 40);

  // TIMI frame counts
  const timiP1Count =
    projection1.timiStart !== null && projection1.timiEnd !== null
      ? projection1.timiEnd - projection1.timiStart
      : null;
  const timiP2Count =
    projection2.timiStart !== null && projection2.timiEnd !== null
      ? projection2.timiEnd - projection2.timiStart
      : null;
  const timiP1Set = timiP1Count !== null;
  const timiP2Set = timiP2Count !== null;

  // Mode-specific warnings
  const timiRequired = mode === 'cQFR' || mode === 'aQFR';
  const timiMissing = timiRequired && (!timiP1Set || !timiP2Set);

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">QFR Analysis</h3>

      {/* QFR Mode Toggle */}
      <button
        onClick={isQfrMode ? disableQfrMode : enableQfrMode}
        className={`w-full py-2 text-sm font-medium rounded transition-colors ${
          isQfrMode
            ? 'bg-red-600/90 text-white hover:bg-red-700'
            : 'bg-brand text-white hover:bg-blue-600'
        }`}
      >
        {isQfrMode ? 'Exit QFR Mode' : 'Enter QFR Mode'}
      </button>

      {isQfrMode && (
        <p className="text-xs text-content-muted">
          The dual projection viewer is active. Upload and manage projections in the main viewer area.
        </p>
      )}

      {/* Projection status summary */}
      {(projection1.loaded || projection2.loaded) && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-content-secondary">Projection Status</div>
          <div className="p-2 rounded border border-border text-xs space-y-1">
            <div className="flex justify-between">
              <span className="opacity-80">Proj 1</span>
              <span className={`font-medium ${projection1.segmented ? 'text-green-400' : projection1.loaded ? 'text-yellow-400' : 'text-content-muted'}`}>
                {projection1.segmented
                  ? `Segmented (${projection1.numCenterlinePoints} pts)`
                  : projection1.loaded
                    ? 'Loaded'
                    : 'Empty'}
              </span>
            </div>
            {/* P1 TIMI info */}
            {projection1.loaded && timiP1Set && (
              <div className="flex justify-between pl-2">
                <span className="opacity-60">TIMI</span>
                <span className="text-blue-400 font-medium">
                  T₀={projection1.timiStart} T₁={projection1.timiEnd} ({timiP1Count} frames)
                </span>
              </div>
            )}
            {/* P1 calibration status */}
            {projection1.loaded && (
              <div className="flex justify-between pl-2">
                <span className="opacity-60">Calibration</span>
                <span className={projection1.pixelSpacing === 0.3 ? 'text-orange-400' : 'text-green-400'}>
                  {projection1.pixelSpacing.toFixed(3)} mm/px
                  {projection1.pixelSpacing === 0.3 && ' (default)'}
                </span>
              </div>
            )}

            <div className="flex justify-between">
              <span className="opacity-80">Proj 2</span>
              <span className={`font-medium ${projection2.segmented ? 'text-green-400' : projection2.loaded ? 'text-yellow-400' : 'text-content-muted'}`}>
                {projection2.segmented
                  ? `Segmented (${projection2.numCenterlinePoints} pts)`
                  : projection2.loaded
                    ? 'Loaded'
                    : 'Empty'}
              </span>
            </div>
            {/* P2 TIMI info */}
            {projection2.loaded && timiP2Set && (
              <div className="flex justify-between pl-2">
                <span className="opacity-60">TIMI</span>
                <span className="text-blue-400 font-medium">
                  T₀={projection2.timiStart} T₁={projection2.timiEnd} ({timiP2Count} frames)
                </span>
              </div>
            )}
            {/* P2 calibration status */}
            {projection2.loaded && (
              <div className="flex justify-between pl-2">
                <span className="opacity-60">Calibration</span>
                <span className={projection2.pixelSpacing === 0.3 ? 'text-orange-400' : 'text-green-400'}>
                  {projection2.pixelSpacing.toFixed(3)} mm/px
                  {projection2.pixelSpacing === 0.3 && ' (default)'}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Angular separation info */}
      {angularSeparation !== null && (
        <div className="text-xs text-content-secondary">
          Angular sep:{' '}
          <span className={angularWarning ? 'text-yellow-400 font-medium' : ''}>
            {angularSeparation.toFixed(1)}&deg;
          </span>
          {angularWarning && (
            <span className="text-yellow-400 ml-1">(recommended: 25-40&deg;)</span>
          )}
        </div>
      )}
      {/* Inline angular sep from projection angles when reconstruction hasn't happened yet */}
      {angularSeparation === null && projection1.loaded && projection2.loaded && (
        <div className="text-xs text-content-secondary">
          Angular sep (est.): {Math.abs(projection1.angleDeg - projection2.angleDeg).toFixed(1)}&deg;
        </div>
      )}

      {/* Mode selector */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">QFR Mode</label>
        <div className="grid grid-cols-3 gap-1">
          {QFR_MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => setMode(m.value)}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                mode === m.value
                  ? 'border-brand bg-brand/10 text-brand'
                  : 'border-border text-content-secondary hover:border-content-muted'
              }`}
              title={m.desc}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Mode-specific warnings */}
      {timiRequired && timiMissing && (
        <div className="p-2 rounded border border-yellow-500/30 bg-yellow-500/5 text-xs text-yellow-400">
          {mode === 'cQFR' && 'cQFR requires TIMI frame selection (T₀/T₁) on both projections for contrast flow estimation.'}
          {mode === 'aQFR' && 'aQFR requires TIMI frame selection for hyperemic flow estimation.'}
          {!timiP1Set && ' Proj 1: TIMI not set.'}
          {!timiP2Set && ' Proj 2: TIMI not set.'}
        </div>
      )}

      {/* Kt input */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Turbulent coeff. (Kt)</label>
        <input
          type="number"
          step={0.01}
          min={0.5}
          max={5}
          value={kt}
          onChange={(e) => setKt(Number(e.target.value))}
          className="w-full px-2 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
        />
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* QFR Result */}
      {qfrResult && (
        <div className="space-y-2">
          <div className="p-3 rounded border border-border bg-surface-primary text-center">
            <div className="text-xs text-content-secondary mb-1">QFR ({qfrResult.mode})</div>
            <div className={`text-3xl font-bold ${qfrColor(qfrResult.qfr)}`}>
              {qfrResult.qfr.toFixed(3)}
            </div>
            {qfrResult.qfr < 0.8 && (
              <div className="text-xs text-red-400 mt-1">Functionally significant (&lt;0.80)</div>
            )}
          </div>

          <div className="p-2 rounded border border-border text-xs space-y-1">
            <Row label="Reference diameter" value={`${qfrResult.reference_diameter_mm.toFixed(2)} mm`} />
            <Row label="MLD" value={`${qfrResult.mld_mm.toFixed(2)} mm`} />
            <Row label="Vessel length" value={`${qfrResult.vessel_length_mm.toFixed(1)} mm`} />
            <Row label="Flow rate" value={`${qfrResult.flow_rate_ml_s.toFixed(2)} ml/s`} />
            <Row label="Kt" value={qfrResult.kt.toFixed(2)} />
          </div>

          {qfrResult.qfr_per_view && (
            <div className="p-2 rounded border border-border text-xs space-y-1">
              <div className="text-content-secondary mb-1 font-medium">Per-view QFR</div>
              {qfrResult.qfr_combined != null && (
                <Row label="Combined" value={qfrResult.qfr_combined.toFixed(3)} />
              )}
              {qfrResult.qfr_per_view.view1 && (
                <Row label="View 1" value={qfrResult.qfr_per_view.view1.qfr.toFixed(3)} />
              )}
              {qfrResult.qfr_per_view.view2 && (
                <Row label="View 2" value={qfrResult.qfr_per_view.view2.qfr.toFixed(3)} />
              )}
            </div>
          )}

          {angularSeparation !== null && reconstructionInfo && (
            <div className="p-2 rounded border border-border text-xs space-y-1">
              <Row label="Angular separation" value={`${angularSeparation.toFixed(1)}\u00B0`} />
              <Row label="3D points" value={String(reconstructionInfo.numPoints)} />
              <Row label="3D vessel length" value={`${reconstructionInfo.vesselLengthMm.toFixed(1)} mm`} />
            </div>
          )}

          <p className="text-xs text-yellow-500/80 italic leading-relaxed">
            {qfrResult.disclaimer}
          </p>
        </div>
      )}

      {/* Clear */}
      {(projection1.loaded || projection2.loaded) && (
        <button
          onClick={clearQfr}
          className="w-full py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors"
        >
          Clear QFR
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
