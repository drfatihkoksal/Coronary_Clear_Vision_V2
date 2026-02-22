import { useState } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { usePlayerStore } from '@/stores/playerStore';

const CATHETER_SIZES = [4, 5, 6, 7, 8] as const;

export function CalibrationPanel() {
  const calibration = useAnalysisStore((s) => s.calibration);
  const frameData = useAnalysisStore((s) => s.frameData);
  const calibrateCatheterFromSeg = useAnalysisStore((s) => s.calibrateCatheterFromSeg);
  const isCalibrating = useAnalysisStore((s) => s.isCalibrating);
  const calibrationError = useAnalysisStore((s) => s.calibrationError);
  const currentFrame = usePlayerStore((s) => s.currentFrame);

  const [catheterSizeFr, setCatheterSizeFr] = useState<number>(6);

  const hasSegmentation = frameData.has(currentFrame);

  const handleCatheterFromSeg = () => {
    calibrateCatheterFromSeg(currentFrame, catheterSizeFr);
  };

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">Calibration</h3>

      {/* Current calibration */}
      {calibration && (
        <div className="p-2 rounded bg-surface-tertiary text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-content-secondary">Pixel Spacing</span>
            <span className="text-content-primary">{calibration.rowSpacing?.toFixed(4) ?? '—'} mm/px</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Source</span>
            <span className="text-content-primary">{calibration.source}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Confidence</span>
            <span className="text-content-primary">{(calibration.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Catheter calibration from segmentation */}
      <div className="space-y-1.5">
        <label className="text-xs text-content-secondary font-medium">Catheter Calibration</label>
        <p className="text-xs text-content-muted">
          Segment the catheter first, then select its French size and calibrate.
        </p>
        <div className="grid grid-cols-5 gap-1">
          {CATHETER_SIZES.map((size) => (
            <button
              key={size}
              onClick={() => setCatheterSizeFr(size)}
              className={`px-1 py-1.5 text-xs rounded border transition-colors ${
                catheterSizeFr === size
                  ? 'border-brand bg-brand/10 text-brand'
                  : 'border-border text-content-secondary hover:border-content-muted'
              }`}
            >
              {size}Fr
            </button>
          ))}
        </div>
        <button
          onClick={handleCatheterFromSeg}
          disabled={isCalibrating || !hasSegmentation}
          className="w-full py-1.5 text-xs font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isCalibrating ? 'Calibrating...' : 'Calibrate from Segmentation'}
        </button>
        {!hasSegmentation && (
          <p className="text-xs text-content-muted">No segmentation on current frame.</p>
        )}
      </div>

      {calibrationError && <p className="text-xs text-red-400">{calibrationError}</p>}
    </div>
  );
}
