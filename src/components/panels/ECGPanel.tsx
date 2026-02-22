import { useTimingStore } from '@/stores/timingStore';
import { useStudyStore } from '@/stores/studyStore';

export function ECGPanel() {
  const metadata = useStudyStore((s) => s.metadata);
  const {
    ecgSignal, heartRate, beatBoundaries, rPeaks,
    isLoadingEcg, ecgError, isEcgEditMode,
    motionSignal, motionPeaks, isCalculatingMotion, motionError,
    loadECG, calculateMotionSignal, toggleEcgEditMode,
  } = useTimingStore();

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">ECG / Timing</h3>

      {/* Load ECG */}
      <button
        onClick={loadECG}
        disabled={isLoadingEcg || !metadata}
        className="w-full py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isLoadingEcg ? 'Loading ECG...' : 'Load ECG'}
      </button>

      {ecgError && (
        <p className="text-xs text-red-400">{ecgError}</p>
      )}

      {/* ECG Info */}
      {ecgSignal && (
        <div className="p-2 rounded bg-surface-tertiary text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-content-secondary">Heart Rate</span>
            <span className="text-content-primary">
              {heartRate != null ? `${heartRate} bpm` : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">R-Peaks</span>
            <span className="text-content-primary">{rPeaks.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Beats</span>
            <span className="text-content-primary">{beatBoundaries.length}</span>
          </div>
        </div>
      )}

      {/* R-Peak Edit Toggle */}
      {ecgSignal && rPeaks.length > 0 && (
        <button
          onClick={toggleEcgEditMode}
          className={`w-full py-1.5 text-xs font-medium rounded border transition-colors ${
            isEcgEditMode
              ? 'border-blue-500 bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
              : 'border-border bg-surface-secondary text-content-secondary hover:bg-surface-tertiary'
          }`}
        >
          {isEcgEditMode ? 'Done Editing R-Peaks' : 'Edit R-Peaks'}
        </button>
      )}

      {/* Beat Boundaries */}
      {beatBoundaries.length > 0 && (
        <div className="space-y-1">
          <label className="text-xs text-content-secondary">Beat Boundaries</label>
          <div className="max-h-32 overflow-y-auto space-y-0.5">
            {beatBoundaries.map((b) => (
              <div
                key={b.beatNumber}
                className="flex justify-between text-xs px-2 py-1 rounded bg-surface-primary"
              >
                <span className="text-content-secondary">Beat {b.beatNumber}</span>
                <span className="text-content-primary">
                  Frames {b.startFrame}-{b.endFrame}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Calculate Motion */}
      <div className="pt-2 border-t border-border">
        <button
          onClick={calculateMotionSignal}
          disabled={isCalculatingMotion || !metadata}
          className="w-full py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isCalculatingMotion ? 'Calculating...' : 'Calculate Motion'}
        </button>
      </div>

      {motionError && (
        <p className="text-xs text-red-400">{motionError}</p>
      )}

      {motionSignal && (
        <div className="p-2 rounded bg-surface-tertiary text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-content-secondary">Frames</span>
            <span className="text-content-primary">{motionSignal.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Motion Peaks</span>
            <span className="text-content-primary">
              {motionPeaks.length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
