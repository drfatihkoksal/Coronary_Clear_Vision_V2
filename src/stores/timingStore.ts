import { create } from 'zustand';
import { fetchECGSignal, updateRPeaks, addRPeak, removeRPeak, moveRPeak } from '@/lib/api/ecg';
import { calculateMotion } from '@/lib/api/motion';
import { eventBus } from '@/lib/eventBus';

interface BeatBoundary {
  beatNumber: number;
  startFrame: number;
  endFrame: number;
}

interface TimingState {
  // ECG
  ecgSignal: number[] | null;
  ecgSampleRate: number;
  rPeaks: number[];
  heartRate: number | null;
  beatBoundaries: BeatBoundary[];
  isLoadingEcg: boolean;
  ecgError: string | null;
  isEcgEditMode: boolean;

  // Visibility
  ecgVisible: boolean;
  motionVisible: boolean;

  // Motion
  motionSignal: number[] | null;
  motionPeaks: number[];
  isCalculatingMotion: boolean;
  motionError: string | null;

  // Actions
  loadECG: () => Promise<void>;
  setRPeaks: (peaks: number[]) => Promise<void>;
  addPeak: (sampleIndex: number) => Promise<void>;
  removePeak: (sampleIndex: number) => Promise<void>;
  movePeak: (fromIndex: number, toIndex: number) => Promise<void>;
  toggleEcgEditMode: () => void;
  toggleEcgVisible: () => void;
  toggleMotionVisible: () => void;
  calculateMotionSignal: () => Promise<void>;
  reset: () => void;
}

function mapBeatBoundaries(data: Record<string, number>[]): BeatBoundary[] {
  return (data || []).map((b) => ({
    beatNumber: b.beat_number,
    startFrame: b.start_frame,
    endFrame: b.end_frame,
  }));
}

export const useTimingStore = create<TimingState>((set, get) => ({
  ecgSignal: null,
  ecgSampleRate: 0,
  rPeaks: [],
  heartRate: null,
  beatBoundaries: [],
  isLoadingEcg: false,
  ecgError: null,
  isEcgEditMode: false,
  ecgVisible: true,
  motionVisible: true,
  motionSignal: null,
  motionPeaks: [],
  isCalculatingMotion: false,
  motionError: null,

  loadECG: async () => {
    set({ isLoadingEcg: true, ecgError: null });
    try {
      const data = await fetchECGSignal();
      set({
        ecgSignal: data.signal,
        ecgSampleRate: data.sample_rate,
        rPeaks: data.r_peaks,
        heartRate: data.heart_rate,
        beatBoundaries: mapBeatBoundaries(data.beat_boundaries),
        isLoadingEcg: false,
      });
      // Auto-calculate motion after ECG loads
      get().calculateMotionSignal();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load ECG';
      set({ isLoadingEcg: false, ecgError: message });
    }
  },

  setRPeaks: async (peaks) => {
    try {
      const data = await updateRPeaks(peaks);
      set({
        rPeaks: data.r_peaks,
        heartRate: data.heart_rate,
        beatBoundaries: mapBeatBoundaries(data.beat_boundaries),
      });
    } catch {
      /* ignore */
    }
  },

  addPeak: async (sampleIndex) => {
    try {
      const data = await addRPeak(sampleIndex);
      set({
        rPeaks: data.r_peaks,
        heartRate: data.heart_rate,
        beatBoundaries: mapBeatBoundaries(data.beat_boundaries),
      });
    } catch {
      /* ignore */
    }
  },

  removePeak: async (sampleIndex) => {
    try {
      const data = await removeRPeak(sampleIndex);
      set({
        rPeaks: data.r_peaks,
        heartRate: data.heart_rate,
        beatBoundaries: mapBeatBoundaries(data.beat_boundaries),
      });
    } catch {
      /* ignore */
    }
  },

  movePeak: async (fromIndex, toIndex) => {
    try {
      const data = await moveRPeak(fromIndex, toIndex);
      set({
        rPeaks: data.r_peaks,
        heartRate: data.heart_rate,
        beatBoundaries: mapBeatBoundaries(data.beat_boundaries),
      });
    } catch {
      /* ignore */
    }
  },

  toggleEcgEditMode: () => set((s) => ({ isEcgEditMode: !s.isEcgEditMode })),
  toggleEcgVisible: () => set((s) => ({ ecgVisible: !s.ecgVisible })),
  toggleMotionVisible: () => set((s) => ({ motionVisible: !s.motionVisible })),

  calculateMotionSignal: async () => {
    set({ isCalculatingMotion: true, motionError: null });
    try {
      const data = await calculateMotion();
      set({
        motionSignal: data.signal,
        motionPeaks: data.peaks,
        isCalculatingMotion: false,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to calculate motion';
      set({ isCalculatingMotion: false, motionError: message });
    }
  },

  reset: () => set({
    ecgSignal: null, ecgSampleRate: 0, rPeaks: [], heartRate: null,
    beatBoundaries: [], isLoadingEcg: false, ecgError: null, isEcgEditMode: false,
    ecgVisible: true, motionVisible: true,
    motionSignal: null, motionPeaks: [], isCalculatingMotion: false, motionError: null,
  }),
}));

// Auto-load ECG when a study is loaded; reset on study clear
eventBus.on('study:loaded', () => {
  useTimingStore.getState().loadECG();
});
eventBus.on('study:cleared', () => {
  useTimingStore.getState().reset();
});
