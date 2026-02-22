import { create } from 'zustand';
import {
  uploadProjection,
  segmentProjection,
  calibrateProjection,
  reconstructQFR,
  recalculateQFR,
  setTimiFrames as apiSetTimiFrames,
  calibrateFromSegmentation as apiCalibrateFromSeg,
  fetchProjectionMask,
  type QFRResultData,
} from '@/lib/api/qfr';
import { useSessionStore } from '@/stores/sessionStore';
import type { SegmentationEngine } from '@/types';

export type QFRViewMode = 'side-by-side' | 'overlay' | 'single' | '3d';
export type QFRMode = 'fQFR' | 'cQFR' | 'aQFR';
export type QFRActiveTool = 'select' | 'seed';
export type QFRSegEngine = Extract<SegmentationEngine, 'angiopy' | 'seedmodel'>;

export interface Point {
  x: number;
  y: number;
}

interface QFRProjectionState {
  loaded: boolean;
  numFrames: number;
  imageWidth: number;
  imageHeight: number;
  angleDeg: number;
  secondaryAngleDeg: number;
  pixelSpacing: number;
  sod: number;
  sid: number;
  frameRate: number;
  segmented: boolean;
  segmentedFrameIndex: number | null;
  numCenterlinePoints: number;
  // TIMI (per-projection)
  timiStart: number | null;
  timiEnd: number | null;
  // Seed points
  seedPoints: Point[];
  // Mask / overlays
  maskBitmap: ImageBitmap | null;
  centerline: Point[];
  diametersMm: number[];
  diametersPx: number[];
  showMask: boolean;
  showCenterline: boolean;
  showSeedPoints: boolean;
  showDiameters: boolean;
}

const emptyProjection: QFRProjectionState = {
  loaded: false,
  numFrames: 0,
  imageWidth: 0,
  imageHeight: 0,
  angleDeg: 0,
  secondaryAngleDeg: 0,
  pixelSpacing: 0.3,
  sod: 1000,
  sid: 1000,
  frameRate: 15,
  segmented: false,
  segmentedFrameIndex: null,
  numCenterlinePoints: 0,
  timiStart: null,
  timiEnd: null,
  seedPoints: [],
  maskBitmap: null,
  centerline: [],
  diametersMm: [],
  diametersPx: [],
  showMask: true,
  showCenterline: true,
  showSeedPoints: true,
  showDiameters: true,
};

export interface Mesh3DData {
  positions: number[];
  normals: number[];
  indices: number[];
  colors: number[];
  num_vertices: number;
  num_triangles: number;
}

interface QFRState {
  isQfrMode: boolean;
  projection1: QFRProjectionState;
  projection2: QFRProjectionState;
  qfrResult: QFRResultData | null;
  angularSeparation: number | null;
  reconstructionInfo: { numPoints: number; vesselLengthMm: number } | null;
  mesh3D: Mesh3DData | null;
  viewMode: QFRViewMode;
  mode: QFRMode;
  kt: number;
  activeTool: QFRActiveTool;
  segEngine: QFRSegEngine;
  mldOverrideIndex: number | null;
  isUploading: boolean;
  isSegmenting: boolean;
  isReconstructing: boolean;
  isRecalculating: boolean;
  error: string | null;

  enableQfrMode: () => void;
  disableQfrMode: () => void;
  setViewMode: (mode: QFRViewMode) => void;
  setMode: (mode: QFRMode) => void;
  setKt: (kt: number) => void;
  setActiveTool: (tool: QFRActiveTool) => void;
  setSegEngine: (engine: QFRSegEngine) => void;
  setTimiFrame: (projectionId: number, type: 'start' | 'end', frame: number) => void;
  setTimiFramesApi: (projectionId: number, start: number, end: number) => Promise<void>;
  addSeedPoint: (projectionId: number, point: Point) => void;
  clearSeedPoints: (projectionId: number) => void;
  setOverlayToggle: (projectionId: number, overlay: 'showMask' | 'showCenterline' | 'showSeedPoints' | 'showDiameters', value: boolean) => void;
  uploadProjection: (projectionId: number, file: File) => Promise<void>;
  segmentProjection: (projectionId: number, frameIndex: number) => Promise<void>;
  calibrateProjection: (projectionId: number, catheterSizeFr: number, catheterDiameterPx: number) => Promise<void>;
  calibrateFromSegmentation: (projectionId: number, catheterSizeFr: number) => Promise<void>;
  loadMask: (projectionId: number, frameIndex: number) => Promise<void>;
  reconstruct: () => Promise<void>;
  setMldOverride: (index: number | null) => void;
  recalculateQfr: (stenosisIndex: number) => Promise<void>;
  clearQfr: () => void;
}

function projKey(id: number): 'projection1' | 'projection2' {
  return id === 1 ? 'projection1' : 'projection2';
}

export const useQFRStore = create<QFRState>((set, get) => ({
  isQfrMode: false,
  projection1: { ...emptyProjection },
  projection2: { ...emptyProjection },
  qfrResult: null,
  angularSeparation: null,
  reconstructionInfo: null,
  mesh3D: null,
  viewMode: 'side-by-side',
  mode: 'fQFR',
  kt: 1.52,
  activeTool: 'select',
  segEngine: 'seedmodel',
  mldOverrideIndex: null,
  isUploading: false,
  isSegmenting: false,
  isReconstructing: false,
  isRecalculating: false,
  error: null,

  enableQfrMode: () => set({ isQfrMode: true }),
  disableQfrMode: () => set({ isQfrMode: false }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setMode: (mode) => set({ mode }),
  setKt: (kt) => set({ kt }),
  setActiveTool: (tool) => set({ activeTool: tool }),
  setSegEngine: (engine) => set({ segEngine: engine }),

  setTimiFrame: (projectionId, type, frame) => {
    const key = projKey(projectionId);
    const current = get()[key];
    if (type === 'start') {
      set({ [key]: { ...current, timiStart: frame } } as Partial<QFRState>);
    } else {
      set({ [key]: { ...current, timiEnd: frame } } as Partial<QFRState>);
    }
  },

  setTimiFramesApi: async (projectionId, start, end) => {
    try {
      await apiSetTimiFrames(projectionId, start, end);
      const key = projKey(projectionId);
      const current = get()[key];
      set({
        [key]: { ...current, timiStart: start, timiEnd: end },
        qfrResult: null,
      } as Partial<QFRState>);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to set TIMI frames';
      set({ error: msg });
    }
  },

  addSeedPoint: (projectionId, point) => {
    const key = projKey(projectionId);
    const current = get()[key];
    set({ [key]: { ...current, seedPoints: [...current.seedPoints, point] } } as Partial<QFRState>);
  },

  clearSeedPoints: (projectionId) => {
    const key = projKey(projectionId);
    const current = get()[key];
    set({ [key]: { ...current, seedPoints: [] } } as Partial<QFRState>);
  },

  setOverlayToggle: (projectionId, overlay, value) => {
    const key = projKey(projectionId);
    const current = get()[key];
    set({ [key]: { ...current, [overlay]: value } } as Partial<QFRState>);
  },

  uploadProjection: async (projectionId: number, file: File) => {
    set({ isUploading: true, error: null });
    try {
      const data = await uploadProjection(projectionId, file);

      // Save session if backend created one (QFR can start without prior DICOM upload)
      if (data.session_id) {
        const sessionStore = useSessionStore.getState();
        if (!sessionStore.sessionId) {
          sessionStore.setSession(data.session_id);
        }
      }

      const proj: QFRProjectionState = {
        ...emptyProjection,
        loaded: true,
        numFrames: data.num_frames,
        imageWidth: data.image_width,
        imageHeight: data.image_height,
        angleDeg: data.angle_deg,
        secondaryAngleDeg: data.secondary_angle_deg,
        pixelSpacing: data.pixel_spacing,
        sod: data.sod,
        sid: data.sid,
        frameRate: data.frame_rate,
      };
      if (projectionId === 1) {
        set({ projection1: proj, isUploading: false, qfrResult: null, angularSeparation: null });
      } else {
        set({ projection2: proj, isUploading: false, qfrResult: null, angularSeparation: null });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed';
      set({ isUploading: false, error: msg });
    }
  },

  segmentProjection: async (projectionId: number, frameIndex: number) => {
    set({ isSegmenting: true, error: null });
    try {
      const key = projKey(projectionId);
      const seedPoints = get()[key].seedPoints;
      const engine = get().segEngine;
      const data = await segmentProjection(
        projectionId,
        frameIndex,
        engine,
        undefined,
        seedPoints.map((p) => [p.x, p.y] as [number, number]),
      );

      // Compute pixel diameters from mm diameters for overlay drawing
      const ps = get()[key].pixelSpacing;
      const diametersPx = data.diameters_mm.map((d) => d / ps);

      // Get fresh state after async call to avoid overwriting concurrent updates
      const fresh = get()[key];
      set({
        [key]: {
          ...fresh,
          segmented: true,
          segmentedFrameIndex: frameIndex,
          numCenterlinePoints: data.num_points,
          centerline: data.centerline,
          diametersMm: data.diameters_mm,
          diametersPx,
        },
        isSegmenting: false,
      } as Partial<QFRState>);

      // Auto-load mask after segmentation for the segmented frame
      get().loadMask(projectionId, frameIndex);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Segmentation failed';
      set({ isSegmenting: false, error: msg });
    }
  },

  calibrateProjection: async (projectionId: number, catheterSizeFr: number, catheterDiameterPx: number) => {
    set({ error: null });
    try {
      const data = await calibrateProjection(projectionId, catheterSizeFr, catheterDiameterPx);
      const key = projKey(projectionId);
      const current = get()[key];
      set({
        [key]: {
          ...current,
          pixelSpacing: data.pixel_spacing,
          segmented: false,
          segmentedFrameIndex: null,
          numCenterlinePoints: 0,
          seedPoints: [],
          maskBitmap: null,
          centerline: [],
          diametersMm: [],
          diametersPx: [],
        },
      } as Partial<QFRState>);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Calibration failed';
      set({ error: msg });
    }
  },

  calibrateFromSegmentation: async (projectionId: number, catheterSizeFr: number) => {
    set({ error: null });
    try {
      const data = await apiCalibrateFromSeg(projectionId, catheterSizeFr);
      const key = projKey(projectionId);
      const current = get()[key];
      set({
        [key]: {
          ...current,
          pixelSpacing: data.pixel_spacing,
          segmented: false,
          segmentedFrameIndex: null,
          numCenterlinePoints: 0,
          seedPoints: [],
          maskBitmap: null,
          centerline: [],
          diametersMm: [],
          diametersPx: [],
        },
      } as Partial<QFRState>);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Calibration from segmentation failed';
      set({ error: msg });
    }
  },

  loadMask: async (projectionId: number, frameIndex: number) => {
    try {
      const bitmap = await fetchProjectionMask(projectionId, frameIndex);
      const key = projKey(projectionId);
      const current = get()[key];
      set({ [key]: { ...current, maskBitmap: bitmap } } as Partial<QFRState>);
    } catch {
      // Mask loading is non-critical
    }
  },

  reconstruct: async () => {
    const { mode, kt, projection1, projection2 } = get();
    set({ isReconstructing: true, error: null, mldOverrideIndex: null });
    try {
      const data = await reconstructQFR(
        mode,
        kt,
        projection1.timiStart,
        projection1.timiEnd,
        projection2.timiStart,
        projection2.timiEnd,
      );
      set({
        qfrResult: data.qfr,
        angularSeparation: data.angular_separation,
        reconstructionInfo: {
          numPoints: data.reconstruction.num_points,
          vesselLengthMm: data.reconstruction.vessel_length_mm,
        },
        mesh3D: data.mesh ?? null,
        isReconstructing: false,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Reconstruction failed';
      set({ isReconstructing: false, error: msg });
    }
  },

  setMldOverride: (index) => set({ mldOverrideIndex: index }),

  recalculateQfr: async (stenosisIndex) => {
    const { mode, kt } = get();
    set({ isRecalculating: true, error: null, mldOverrideIndex: stenosisIndex });
    try {
      const data = await recalculateQFR([stenosisIndex], mode, kt);
      set({
        qfrResult: data.qfr,
        mesh3D: data.mesh ?? null,
        isRecalculating: false,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Recalculation failed';
      set({ isRecalculating: false, error: msg });
    }
  },

  clearQfr: () =>
    set({
      isQfrMode: false,
      projection1: { ...emptyProjection },
      projection2: { ...emptyProjection },
      qfrResult: null,
      angularSeparation: null,
      reconstructionInfo: null,
      mesh3D: null,
      mldOverrideIndex: null,
      viewMode: 'side-by-side',
      activeTool: 'select',
      segEngine: 'seedmodel',
      error: null,
    }),
}));
