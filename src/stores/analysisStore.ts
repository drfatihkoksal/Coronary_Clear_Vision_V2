import { create } from 'zustand';
import { segmentAndExtract, fetchMask, getEngines } from '@/lib/api/segmentation';
import { calculateQCA } from '@/lib/api/qca';
import { calibrateCatheter as apiCalibrateCatheter, calibrateCatheterFromSegmentation as apiCalibrateCatheterFromSeg, calibrateManual as apiCalibrateManual, getCurrentCalibration } from '@/lib/api/calibration';
import { eventBus } from '@/lib/eventBus';
import type { SegmentationEngine, Point, BoundingBox, QCAMetrics, PixelSpacing } from '@/types';

interface FrameSegmentationData {
  maskBitmap: ImageBitmap | null;
  centerline: Point[];
  diameters_px: number[];
  engine: SegmentationEngine;
  confidence: number;
  inferenceTimeMs: number;
}

interface AnalysisState {
  // Segmentation data per frame
  frameData: Map<number, FrameSegmentationData>;
  selectedEngine: SegmentationEngine;
  isSegmenting: boolean;
  segmentationError: string | null;

  // Available engines
  availableEngines: Record<string, { available: boolean }>;

  // Annotations per frame
  seedPoints: Map<number, Point[]>;
  rois: Map<number, BoundingBox | null>;

  // QCA
  qcaResults: Map<number, QCAMetrics>;
  isComputingQCA: boolean;
  qcaError: string | null;

  // Calibration
  calibration: PixelSpacing | null;
  isCalibrating: boolean;
  calibrationError: string | null;

  // Actions
  segmentAndExtract: (frameIndex: number, roi?: BoundingBox, seeds?: Point[]) => Promise<void>;
  setEngine: (engine: SegmentationEngine) => void;
  addSeedPoint: (frameIndex: number, point: Point) => void;
  removeSeedPoint: (frameIndex: number, index: number) => void;
  clearSeedPoints: (frameIndex: number) => void;
  setRoi: (frameIndex: number, roi: BoundingBox | null) => void;
  computeQCA: (frameIndex: number, method?: string) => Promise<void>;
  calibrateCatheter: (catheterDiameterPx: number, catheterSizeFr: number) => Promise<void>;
  calibrateCatheterFromSeg: (frameIndex: number, catheterSizeFr: number) => Promise<void>;
  calibrateManual: (knownDistanceMm: number, measuredDistancePx: number) => Promise<void>;
  loadCalibration: () => Promise<void>;
  setPixelSpacing: (ps: PixelSpacing) => void;
  loadAvailableEngines: () => Promise<void>;
  clearFrameData: (frameIndex: number) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  frameData: new Map(),
  selectedEngine: 'nnunet' as SegmentationEngine,
  isSegmenting: false,
  segmentationError: null,
  availableEngines: {},
  seedPoints: new Map(),
  rois: new Map(),
  qcaResults: new Map(),
  isComputingQCA: false,
  qcaError: null,
  calibration: null,
  isCalibrating: false,
  calibrationError: null,

  segmentAndExtract: async (frameIndex: number, roi?: BoundingBox, seeds?: Point[]) => {
    set({ isSegmenting: true, segmentationError: null });
    try {
      const { selectedEngine } = get();
      const roiArray = roi ? [roi.x, roi.y, roi.width, roi.height] as [number, number, number, number] : undefined;
      const seedArray = seeds?.map(p => [p.x, p.y] as [number, number]);

      const result = await segmentAndExtract({
        frame_index: frameIndex,
        engine: selectedEngine,
        roi: roiArray,
        seed_points: seedArray,
      });

      // Fetch the mask bitmap
      let maskBitmap: ImageBitmap | null = null;
      try {
        maskBitmap = await fetchMask(frameIndex);
      } catch { /* mask might not be available */ }

      const data: FrameSegmentationData = {
        maskBitmap,
        centerline: result.centerline || [],
        diameters_px: result.diameters_px || [],
        engine: selectedEngine,
        confidence: result.confidence,
        inferenceTimeMs: result.inference_time_ms,
      };

      set((state) => {
        const newMap = new Map(state.frameData);
        newMap.set(frameIndex, data);
        return { frameData: newMap, isSegmenting: false };
      });

      eventBus.emit('segmentation:completed', { frameIndex });

      // Auto-compute QCA after successful segmentation
      if (data.centerline.length >= 2) {
        get().computeQCA(frameIndex);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Segmentation failed';
      set({ isSegmenting: false, segmentationError: message });
    }
  },

  setEngine: (engine) => set({ selectedEngine: engine }),

  addSeedPoint: (frameIndex, point) => {
    set((state) => {
      const newMap = new Map(state.seedPoints);
      const existing = newMap.get(frameIndex) || [];
      newMap.set(frameIndex, [...existing, point]);
      return { seedPoints: newMap };
    });
  },

  removeSeedPoint: (frameIndex, index) => {
    set((state) => {
      const newMap = new Map(state.seedPoints);
      const existing = newMap.get(frameIndex) || [];
      newMap.set(frameIndex, existing.filter((_, i) => i !== index));
      return { seedPoints: newMap };
    });
  },

  clearSeedPoints: (frameIndex) => {
    set((state) => {
      const newMap = new Map(state.seedPoints);
      newMap.delete(frameIndex);
      return { seedPoints: newMap };
    });
  },

  setRoi: (frameIndex, roi) => {
    set((state) => {
      const newMap = new Map(state.rois);
      newMap.set(frameIndex, roi);
      return { rois: newMap };
    });
  },

  computeQCA: async (frameIndex: number, method = 'gaussian') => {
    set({ isComputingQCA: true, qcaError: null });
    try {
      const result = await calculateQCA(frameIndex, method);
      set((state) => {
        const newMap = new Map(state.qcaResults);
        newMap.set(frameIndex, result);
        return { qcaResults: newMap, isComputingQCA: false };
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'QCA computation failed';
      set({ isComputingQCA: false, qcaError: message });
    }
  },

  calibrateCatheter: async (catheterDiameterPx: number, catheterSizeFr: number) => {
    set({ isCalibrating: true, calibrationError: null });
    try {
      const result = await apiCalibrateCatheter(catheterDiameterPx, catheterSizeFr);
      const calibration: PixelSpacing = {
        rowSpacing: result.pixel_spacing_mm,
        colSpacing: result.pixel_spacing_mm,
        source: 'catheter',
        confidence: 0.9,
      };
      set({ calibration, isCalibrating: false });
      eventBus.emit('calibration:changed', { pixelSpacing: result.pixel_spacing_mm });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Catheter calibration failed';
      set({ isCalibrating: false, calibrationError: message });
    }
  },

  calibrateCatheterFromSeg: async (frameIndex: number, catheterSizeFr: number) => {
    set({ isCalibrating: true, calibrationError: null });
    try {
      const result = await apiCalibrateCatheterFromSeg(frameIndex, catheterSizeFr);
      const calibration: PixelSpacing = {
        rowSpacing: result.pixel_spacing_mm,
        colSpacing: result.pixel_spacing_mm,
        source: 'catheter',
        confidence: 0.9,
      };
      set({ calibration, isCalibrating: false });
      eventBus.emit('calibration:changed', { pixelSpacing: result.pixel_spacing_mm });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Catheter calibration failed';
      set({ isCalibrating: false, calibrationError: message });
    }
  },

  calibrateManual: async (knownDistanceMm: number, measuredDistancePx: number) => {
    set({ isCalibrating: true, calibrationError: null });
    try {
      const result = await apiCalibrateManual(knownDistanceMm, measuredDistancePx);
      const calibration: PixelSpacing = {
        rowSpacing: result.pixel_spacing_mm,
        colSpacing: result.pixel_spacing_mm,
        source: 'manual',
        confidence: 0.7,
      };
      set({ calibration, isCalibrating: false });
      eventBus.emit('calibration:changed', { pixelSpacing: result.pixel_spacing_mm });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Manual calibration failed';
      set({ isCalibrating: false, calibrationError: message });
    }
  },

  loadCalibration: async () => {
    try {
      const calibration = await getCurrentCalibration();
      set({ calibration });
    } catch { /* ignore */ }
  },

  setPixelSpacing: (ps) => {
    set({ calibration: ps });
  },

  loadAvailableEngines: async () => {
    try {
      const result = await getEngines();
      set({ availableEngines: result.engines });
    } catch { /* ignore */ }
  },

  clearFrameData: (frameIndex) => {
    set((state) => {
      const newMap = new Map(state.frameData);
      const data = newMap.get(frameIndex);
      if (data?.maskBitmap) data.maskBitmap.close();
      newMap.delete(frameIndex);
      return { frameData: newMap };
    });
  },

  reset: () => {
    const { frameData } = get();
    frameData.forEach((d) => d.maskBitmap?.close());
    set({
      frameData: new Map(),
      selectedEngine: 'nnunet',
      isSegmenting: false,
      segmentationError: null,
      availableEngines: {},
      seedPoints: new Map(),
      rois: new Map(),
      qcaResults: new Map(),
      isComputingQCA: false,
      qcaError: null,
      calibration: null,
      isCalibrating: false,
      calibrationError: null,
    });
  },
}));

// Auto-calibrate from DICOM pixel spacing on study load
eventBus.on('study:loaded', (data) => {
  if (data.metadata.pixelSpacing) {
    useAnalysisStore.getState().setPixelSpacing(data.metadata.pixelSpacing);
  }
});

// Clear all analysis data when study is cleared
eventBus.on('study:cleared', () => {
  useAnalysisStore.getState().reset();
});
