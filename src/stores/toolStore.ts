import { create } from 'zustand';

export type AnnotationTool = 'select' | 'roi' | 'seed' | 'pan';

interface ToolState {
  activeTool: AnnotationTool;
  overlays: {
    mask: boolean;
    centerline: boolean;
    seedPoints: boolean;
    roi: boolean;
    diameterMarkers: boolean;
    ecgOverlay: boolean;
  };

  setActiveTool: (tool: AnnotationTool) => void;
  setOverlayVisibility: (layer: keyof ToolState['overlays'], visible: boolean) => void;
  toggleOverlay: (layer: keyof ToolState['overlays']) => void;
  reset: () => void;
}

const defaultOverlays = {
  mask: true,
  centerline: true,
  seedPoints: true,
  roi: true,
  diameterMarkers: false,
  ecgOverlay: false,
};

export const useToolStore = create<ToolState>((set) => ({
  activeTool: 'select',
  overlays: { ...defaultOverlays },

  setActiveTool: (tool) => set({ activeTool: tool }),

  setOverlayVisibility: (layer, visible) =>
    set((state) => ({
      overlays: { ...state.overlays, [layer]: visible },
    })),

  toggleOverlay: (layer) =>
    set((state) => ({
      overlays: { ...state.overlays, [layer]: !state.overlays[layer] },
    })),

  reset: () => set({ activeTool: 'select', overlays: { ...defaultOverlays } }),
}));
