import { create } from 'zustand';
import { uploadDicom, fetchFrame, clearStudy as apiClearStudy } from '@/lib/api/dicom';
import { eventBus } from '@/lib/eventBus';
import { useSessionStore } from '@/stores/sessionStore';
import type { StudyMetadata } from '@/types';

interface StudyState {
  metadata: StudyMetadata | null;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  frameCache: Map<number, ImageBitmap>;

  loadFile: (file: File, anonymize: boolean) => Promise<void>;
  getFrame: (index: number) => Promise<ImageBitmap>;
  clearStudy: () => Promise<void>;
  reset: () => void;
}

export const useStudyStore = create<StudyState>((set, get) => ({
  metadata: null,
  isLoaded: false,
  isLoading: false,
  error: null,
  frameCache: new Map(),

  loadFile: async (file: File, anonymize: boolean) => {
    // Clear previous study state before loading a new one
    if (get().isLoaded) {
      get().reset();
      eventBus.emit('study:cleared');
    }
    set({ isLoading: true, error: null });
    try {
      const res = await uploadDicom(file, anonymize);
      useSessionStore.getState().setSession(res.session_id);

      const rawPs = res.pixel_spacing;
      const metadata: StudyMetadata = {
        sessionId: res.session_id,
        patient: res.patient,
        studyInfo: res.study_info,
        numFrames: res.num_frames,
        frameRate: res.frame_rate,
        imageWidth: res.image_width,
        imageHeight: res.image_height,
        pixelSpacing: rawPs
          ? {
              rowSpacing: rawPs.row_spacing,
              colSpacing: rawPs.col_spacing,
              source: rawPs.source as 'dicom' | 'catheter' | 'manual' | 'from_mask',
              confidence: rawPs.confidence,
            }
          : null,
      };

      set({ metadata, isLoaded: true, isLoading: false, frameCache: new Map() });
      eventBus.emit('study:loaded', { sessionId: res.session_id, metadata });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load DICOM';
      set({ isLoading: false, error: message });
    }
  },

  getFrame: async (index: number) => {
    const { frameCache } = get();
    const cached = frameCache.get(index);
    if (cached) return cached;

    const bitmap = await fetchFrame(index);
    set((state) => {
      const newCache = new Map(state.frameCache);
      newCache.set(index, bitmap);
      return { frameCache: newCache };
    });
    return bitmap;
  },

  clearStudy: async () => {
    try {
      await apiClearStudy();
    } catch {
      /* ignore */
    }
    get().reset();
    eventBus.emit('study:cleared');
  },

  reset: () => {
    const { frameCache } = get();
    frameCache.forEach((bmp) => bmp.close());
    set({ metadata: null, isLoaded: false, isLoading: false, error: null, frameCache: new Map() });
  },
}));
