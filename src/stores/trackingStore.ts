import { create } from 'zustand';
import {
  initializeTracking as apiInitialize,
  propagateTracking as apiPropagate,
  getTrackingState as apiGetState,
  clearTracking as apiClear,
} from '@/lib/api/tracking';
import type { TrackingFrameResult } from '@/lib/api/tracking';
import { useAnalysisStore } from '@/stores/analysisStore';
import { eventBus } from '@/lib/eventBus';
import type { BoundingBox } from '@/types';

export interface TrackedFrameData {
  bbox: [number, number, number, number];
  confidence: number;
}

interface TrackingState {
  isInitialized: boolean;
  isTrackMode: boolean;
  isTracking: boolean;
  isPropagating: boolean;
  autoSegQcaEnabled: boolean;
  confidenceThreshold: number;
  confidence: number;
  trackedFrames: Map<number, TrackedFrameData>;
  startFrame: number | null;
  error: string | null;

  initializeTracking: (frameIndex: number, roi: [number, number, number, number]) => Promise<void>;
  trackSingleFrame: (targetFrame: number) => Promise<boolean>;
  propagateForward: (maxFrames?: number) => Promise<void>;
  propagateBackward: (maxFrames?: number) => Promise<void>;
  enableTrackMode: () => void;
  disableTrackMode: () => void;
  setAutoSegQcaEnabled: (enabled: boolean) => void;
  setConfidenceThreshold: (threshold: number) => void;
  refreshState: () => Promise<void>;
  clearTracking: () => Promise<void>;
}

function parseResultsArray(
  results: TrackingFrameResult[],
): Map<number, TrackedFrameData> {
  const map = new Map<number, TrackedFrameData>();
  for (const r of results) {
    map.set(r.frame_index, { bbox: r.bbox, confidence: r.confidence });
  }
  return map;
}

function roiToTuple(roi: BoundingBox): [number, number, number, number] {
  return [Math.round(roi.x), Math.round(roi.y), Math.round(roi.width), Math.round(roi.height)];
}

export const useTrackingStore = create<TrackingState>((set, get) => ({
  isInitialized: false,
  isTrackMode: false,
  isTracking: false,
  isPropagating: false,
  autoSegQcaEnabled: false,
  confidenceThreshold: 0.5,
  confidence: 0,
  trackedFrames: new Map(),
  startFrame: null,
  error: null,

  initializeTracking: async (frameIndex: number, roi: [number, number, number, number]) => {
    set({ isTracking: true, error: null });
    try {
      const result = await apiInitialize(frameIndex, roi);
      const trackedFrames = new Map<number, TrackedFrameData>();
      // Backend init returns {frame_index, roi, status} — confidence is always 1.0 for init
      trackedFrames.set(frameIndex, { bbox: result.roi, confidence: 1.0 });
      set({
        isInitialized: true,
        isTracking: false,
        confidence: 1.0,
        startFrame: frameIndex,
        trackedFrames,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Tracking initialization failed';
      set({ isTracking: false, error: msg });
    }
  },

  trackSingleFrame: async (targetFrame: number) => {
    // Guard against concurrent calls (seg+QCA takes time, user may press → again)
    if (get().isTracking) return false;

    const { autoSegQcaEnabled } = get();
    // Always read fresh state for ROI lookup
    const analysisStore = useAnalysisStore.getState();
    const prevRoi = analysisStore.rois.get(targetFrame - 1);

    if (!prevRoi) {
      set({ error: 'No reference ROI on previous frame' });
      return false;
    }

    set({ isTracking: true, error: null });
    try {
      // Re-initialize on previous frame (multi-replica safety)
      await apiInitialize(targetFrame - 1, roiToTuple(prevRoi));

      // Propagate forward one frame
      const result = await apiPropagate('forward', 1, false);

      // Backend returns { direction, tracked_frames (count), total_frames, results (array) }
      const results = result.results ?? [];
      if (results.length === 0) {
        set({ isTracking: false, error: 'No tracking result returned' });
        return false;
      }

      // Store ALL tracked frames' ROIs (backend may return more than requested)
      const freshAnalysis = useAnalysisStore.getState();
      const merged = new Map(get().trackedFrames);
      let targetConfidence = 0;

      for (const frameResult of results) {
        const frameIdx = frameResult.frame_index;
        const [x, y, w, h] = frameResult.bbox;
        const roi: BoundingBox = { x, y, width: w, height: h };
        freshAnalysis.setRoi(frameIdx, roi);
        merged.set(frameIdx, { bbox: frameResult.bbox, confidence: frameResult.confidence });
        if (frameIdx === targetFrame) {
          targetConfidence = frameResult.confidence;
        }
      }

      // If exact target wasn't in response, use last result's confidence
      if (targetConfidence === 0 && results.length > 0) {
        targetConfidence = results[results.length - 1].confidence;
      }

      set({
        isTracking: false,
        isInitialized: true,
        confidence: targetConfidence,
        trackedFrames: merged,
      });

      eventBus.emit('tracking:updated', { frameIndex: targetFrame });

      // Fire-and-forget seg+QCA so it doesn't block frame advancement.
      // goToFrame must happen immediately after trackSingleFrame returns.
      if (autoSegQcaEnabled) {
        const targetRoi = useAnalysisStore.getState().rois.get(targetFrame);
        if (targetRoi) {
          useAnalysisStore.getState().segmentAndExtract(targetFrame, targetRoi).catch(() => {});
        }
      }

      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Tracking failed';
      set({ isTracking: false, error: message });
      return false;
    }
  },

  propagateForward: async (maxFrames?: number) => {
    set({ isPropagating: true, error: null });
    try {
      const result = await apiPropagate('forward', maxFrames);
      const newEntries = parseResultsArray(result.results ?? []);
      set((state) => {
        const merged = new Map(state.trackedFrames);
        for (const [k, v] of newEntries) {
          merged.set(k, v);
        }
        return { trackedFrames: merged, isPropagating: false };
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Forward propagation failed';
      set({ isPropagating: false, error: msg });
    }
  },

  propagateBackward: async (maxFrames?: number) => {
    set({ isPropagating: true, error: null });
    try {
      const result = await apiPropagate('backward', maxFrames);
      const newEntries = parseResultsArray(result.results ?? []);
      set((state) => {
        const merged = new Map(state.trackedFrames);
        for (const [k, v] of newEntries) {
          merged.set(k, v);
        }
        return { trackedFrames: merged, isPropagating: false };
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Backward propagation failed';
      set({ isPropagating: false, error: msg });
    }
  },

  enableTrackMode: () => set({ isTrackMode: true }),
  disableTrackMode: () => set({ isTrackMode: false }),
  setAutoSegQcaEnabled: (enabled) => set({ autoSegQcaEnabled: enabled }),
  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),

  refreshState: async () => {
    try {
      const result = await apiGetState();
      const trackedFrames = parseResultsArray(result.results ?? []);
      set({
        isInitialized: result.status === 'initialized',
        startFrame: result.start_frame ?? null,
        trackedFrames,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to fetch tracking state';
      set({ error: msg });
    }
  },

  clearTracking: async () => {
    try {
      await apiClear();
    } catch {
      // ignore server error on clear; reset locally anyway
    }
    const { autoSegQcaEnabled, confidenceThreshold } = get();
    set({
      isInitialized: false,
      isTrackMode: false,
      isTracking: false,
      isPropagating: false,
      autoSegQcaEnabled,
      confidenceThreshold,
      confidence: 0,
      trackedFrames: new Map(),
      startFrame: null,
      error: null,
    });
  },
}));

// Reset tracking when study is cleared
eventBus.on('study:cleared', () => {
  useTrackingStore.getState().clearTracking();
});
