import { useAnalysisStore } from '@/stores/analysisStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useStudyStore } from '@/stores/studyStore';
import { useToolStore } from '@/stores/toolStore';
import type { SegmentationEngine } from '@/types';

const ENGINES: { value: SegmentationEngine; label: string; desc: string }[] = [
  { value: 'nnunet', label: 'nnU-Net (ROI)', desc: 'Best for ROI-cropped regions' },
  { value: 'nnunet_wide', label: 'nnU-Net (Wide)', desc: 'Wider context window' },
  { value: 'nnunet_fullframe', label: 'nnU-Net (Full)', desc: 'Full frame processing' },
  { value: 'angiopy', label: 'AngioPy', desc: 'Seed-based segmentation' },
  { value: 'seedmodel', label: 'SeedModel', desc: 'Seed-guided nnU-Net segmentation' },
];

export function SegmentationPanel() {
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const metadata = useStudyStore((s) => s.metadata);
  const {
    selectedEngine, isSegmenting, segmentationError, frameData,
    seedPoints, rois,
    setEngine, segmentAndExtract, clearSeedPoints, setRoi,
  } = useAnalysisStore();
  const { activeTool, setActiveTool, overlays, toggleOverlay } = useToolStore();

  const currentSeeds = seedPoints.get(currentFrame) || [];
  const currentRoi = rois.get(currentFrame);
  const currentSegData = frameData.get(currentFrame);

  const handleSegment = () => {
    if (!metadata) return;
    segmentAndExtract(
      currentFrame,
      currentRoi || undefined,
      currentSeeds.length > 0 ? currentSeeds : undefined,
    );
  };

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">Segmentation</h3>

      {/* Engine selection */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Engine</label>
        <div className="grid grid-cols-2 gap-1">
          {ENGINES.map((e) => (
            <button
              key={e.value}
              onClick={() => setEngine(e.value)}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                selectedEngine === e.value
                  ? 'border-brand bg-brand/10 text-brand'
                  : 'border-border text-content-secondary hover:border-content-muted'
              }`}
              title={e.desc}
            >
              {e.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tool buttons */}
      <div className="flex gap-1">
        <button
          onClick={() => setActiveTool(activeTool === 'seed' ? 'select' : 'seed')}
          className={`flex-1 px-2 py-1.5 text-xs rounded border transition-colors ${
            activeTool === 'seed' ? 'border-brand bg-brand/10 text-brand' : 'border-border text-content-secondary hover:border-content-muted'
          }`}
        >
          Seed (S)
        </button>
        <button
          onClick={() => setActiveTool(activeTool === 'roi' ? 'select' : 'roi')}
          className={`flex-1 px-2 py-1.5 text-xs rounded border transition-colors ${
            activeTool === 'roi' ? 'border-brand bg-brand/10 text-brand' : 'border-border text-content-secondary hover:border-content-muted'
          }`}
        >
          ROI (B)
        </button>
      </div>

      {/* Seeds list */}
      {currentSeeds.length > 0 && (
        <div className="text-xs text-content-muted">
          Seeds: {currentSeeds.length}
          <button
            onClick={() => clearSeedPoints(currentFrame)}
            className="ml-2 text-red-400 hover:text-red-300"
          >
            Clear
          </button>
        </div>
      )}

      {/* ROI info */}
      {currentRoi && (
        <div className="text-xs text-content-muted">
          ROI: [{currentRoi.x}, {currentRoi.y}, {currentRoi.width}x{currentRoi.height}]
          <button
            onClick={() => setRoi(currentFrame, null)}
            className="ml-2 text-red-400 hover:text-red-300"
          >
            Clear
          </button>
        </div>
      )}

      {/* Segment button */}
      <button
        onClick={handleSegment}
        disabled={isSegmenting || !metadata}
        className="w-full py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isSegmenting ? 'Segmenting...' : 'Segment & Extract'}
      </button>

      {segmentationError && (
        <p className="text-xs text-red-400">{segmentationError}</p>
      )}

      {/* Result info */}
      {currentSegData && (
        <div className="p-2 rounded bg-surface-tertiary text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-content-secondary">Confidence</span>
            <span className="text-content-primary">{(currentSegData.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Centerline pts</span>
            <span className="text-content-primary">{currentSegData.centerline.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-content-secondary">Time</span>
            <span className="text-content-primary">{currentSegData.inferenceTimeMs.toFixed(0)}ms</span>
          </div>
        </div>
      )}

      {/* Overlay toggles */}
      <div className="space-y-1">
        <label className="text-xs text-content-secondary">Overlays</label>
        {(Object.keys(overlays) as (keyof typeof overlays)[]).map((key) => (
          <label key={key} className="flex items-center gap-2 text-xs text-content-secondary">
            <input
              type="checkbox"
              checked={overlays[key]}
              onChange={() => toggleOverlay(key)}
              className="rounded border-border"
            />
            {key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase())}
          </label>
        ))}
      </div>
    </div>
  );
}
