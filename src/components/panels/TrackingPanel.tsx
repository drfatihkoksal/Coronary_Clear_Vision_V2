import { useState } from 'react';
import { useTrackingStore } from '@/stores/trackingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useAnalysisStore } from '@/stores/analysisStore';

export function TrackingPanel() {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);

  const currentRoi = useAnalysisStore((s) => s.rois.get(currentFrame));

  const isInitialized = useTrackingStore((s) => s.isInitialized);
  const isTrackMode = useTrackingStore((s) => s.isTrackMode);
  const isPropagating = useTrackingStore((s) => s.isPropagating);
  const confidence = useTrackingStore((s) => s.confidence);
  const trackedFrames = useTrackingStore((s) => s.trackedFrames);
  const startFrame = useTrackingStore((s) => s.startFrame);
  const error = useTrackingStore((s) => s.error);

  const initializeTracking = useTrackingStore((s) => s.initializeTracking);
  const propagateForward = useTrackingStore((s) => s.propagateForward);
  const propagateBackward = useTrackingStore((s) => s.propagateBackward);
  const enableTrackMode = useTrackingStore((s) => s.enableTrackMode);
  const disableTrackMode = useTrackingStore((s) => s.disableTrackMode);
  const clearTracking = useTrackingStore((s) => s.clearTracking);

  // Manual ROI input fallback
  const [manualRoi, setManualRoi] = useState<{ x: string; y: string; w: string; h: string }>({
    x: '0', y: '0', w: '100', h: '100',
  });
  const [maxFrames, setMaxFrames] = useState<string>('');

  const resolvedRoi: [number, number, number, number] | null = currentRoi
    ? [currentRoi.x, currentRoi.y, currentRoi.width, currentRoi.height]
    : (Number(manualRoi.w) > 0 && Number(manualRoi.h) > 0)
      ? [Number(manualRoi.x), Number(manualRoi.y), Number(manualRoi.w), Number(manualRoi.h)]
      : null;

  const canInitialize = resolvedRoi !== null && totalFrames > 0;
  const canPropagate = isInitialized && !isPropagating;
  const parsedMaxFrames = maxFrames ? Number(maxFrames) : undefined;

  const handleInitialize = () => {
    if (resolvedRoi) {
      initializeTracking(currentFrame, resolvedRoi);
    }
  };

  const currentFrameData = trackedFrames.get(currentFrame);
  const trackedCount = trackedFrames.size;

  // Collect confidence values for display
  const sortedTrackedFrames = Array.from(trackedFrames.entries()).sort((a, b) => a[0] - b[0]);

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">Object Tracking</h3>

      {/* Track mode toggle */}
      <div className="flex items-center gap-2">
        <button
          onClick={isTrackMode ? disableTrackMode : enableTrackMode}
          className={`flex-1 py-2 text-sm font-medium rounded transition-colors ${
            isTrackMode
              ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
              : 'bg-surface-primary text-content-secondary border border-border hover:border-content-muted'
          }`}
        >
          {isTrackMode ? 'Track Mode ON' : 'Enable Track Mode'}
        </button>
        {isTrackMode && (
          <span className="w-2.5 h-2.5 rounded-full bg-orange-400 animate-pulse shrink-0" />
        )}
      </div>

      {/* Status */}
      <div className="p-2 rounded border border-border text-xs space-y-1">
        <Row label="Status" value={isInitialized ? 'Initialized' : 'Not initialized'} />
        <Row label="Current Frame" value={String(currentFrame)} />
        {startFrame !== null && <Row label="Start Frame" value={String(startFrame)} />}
        <Row label="Tracked Frames" value={String(trackedCount)} />
        {confidence > 0 && <Row label="Init Confidence" value={`${(confidence * 100).toFixed(1)}%`} />}
        {currentFrameData && (
          <Row label="Current Conf." value={`${(currentFrameData.confidence * 100).toFixed(1)}%`} />
        )}
      </div>

      {/* ROI input: show manual inputs only when no ROI from analysisStore */}
      {!currentRoi && (
        <div className="space-y-1">
          <label className="text-xs text-content-secondary">Manual ROI (x, y, w, h)</label>
          <div className="grid grid-cols-4 gap-1">
            <input
              type="number"
              value={manualRoi.x}
              onChange={(e) => setManualRoi((r) => ({ ...r, x: e.target.value }))}
              placeholder="x"
              className="w-full px-1.5 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
            />
            <input
              type="number"
              value={manualRoi.y}
              onChange={(e) => setManualRoi((r) => ({ ...r, y: e.target.value }))}
              placeholder="y"
              className="w-full px-1.5 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
            />
            <input
              type="number"
              value={manualRoi.w}
              onChange={(e) => setManualRoi((r) => ({ ...r, w: e.target.value }))}
              placeholder="w"
              className="w-full px-1.5 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
            />
            <input
              type="number"
              value={manualRoi.h}
              onChange={(e) => setManualRoi((r) => ({ ...r, h: e.target.value }))}
              placeholder="h"
              className="w-full px-1.5 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
            />
          </div>
        </div>
      )}

      {currentRoi && (
        <div className="text-xs text-content-secondary">
          Using ROI from viewer: [{currentRoi.x}, {currentRoi.y}, {currentRoi.width}, {currentRoi.height}]
        </div>
      )}

      {/* Initialize button */}
      <button
        onClick={handleInitialize}
        disabled={!canInitialize}
        className="w-full py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Initialize at Frame {currentFrame}
      </button>

      {/* Max frames for propagation */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Max Frames (optional)</label>
        <input
          type="number"
          min={1}
          value={maxFrames}
          onChange={(e) => setMaxFrames(e.target.value)}
          placeholder="All"
          className="w-full px-2 py-1.5 text-xs rounded border border-border bg-surface-primary text-content-primary"
        />
      </div>

      {/* Propagate buttons */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => propagateBackward(parsedMaxFrames)}
          disabled={!canPropagate}
          className="py-2 text-sm font-medium rounded bg-surface-primary text-content-primary border border-border hover:border-brand disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isPropagating ? '...' : 'Backward'}
        </button>
        <button
          onClick={() => propagateForward(parsedMaxFrames)}
          disabled={!canPropagate}
          className="py-2 text-sm font-medium rounded bg-surface-primary text-content-primary border border-border hover:border-brand disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isPropagating ? '...' : 'Forward'}
        </button>
      </div>

      {/* Error display */}
      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Tracked frames list */}
      {sortedTrackedFrames.length > 0 && (
        <div className="space-y-1">
          <label className="text-xs text-content-secondary font-medium">
            Tracked Frames ({trackedCount})
          </label>
          <div className="max-h-48 overflow-y-auto space-y-0.5">
            {sortedTrackedFrames.map(([frame, data]) => (
              <div
                key={frame}
                className={`flex justify-between text-xs px-2 py-1 rounded ${
                  frame === currentFrame
                    ? 'bg-brand/10 text-brand border border-brand/30'
                    : 'bg-surface-primary text-content-secondary'
                }`}
              >
                <span>Frame {frame}</span>
                <span
                  className={
                    data.confidence >= 0.8
                      ? 'text-green-400'
                      : data.confidence >= 0.5
                        ? 'text-yellow-400'
                        : 'text-red-400'
                  }
                >
                  {(data.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clear button */}
      {isInitialized && (
        <button
          onClick={clearTracking}
          className="w-full py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors"
        >
          Clear Tracking
        </button>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-content-secondary">{label}</span>
      <span className="text-content-primary font-medium">{value}</span>
    </div>
  );
}
