import { create } from 'zustand';
import {
  applyBrush,
  applySmartBrush,
  applyFloodFill,
  applyMorphological,
} from '@/lib/api/mask-edit';
import type { MorphologicalRequest } from '@/lib/api/mask-edit';

export type MaskEditTool = 'brush' | 'eraser' | 'smartBrush' | 'floodFill';

const MAX_UNDO_STACK = 20;

interface MaskEditState {
  isEditMode: boolean;
  frameIndex: number;
  workingMask: ImageBitmap | null;
  activeTool: MaskEditTool;
  brushSize: number;
  tolerance: number;
  undoStack: ImageBitmap[];
  redoStack: ImageBitmap[];
  isSaving: boolean;

  enterEditMode: (frameIndex: number, mask: ImageBitmap) => void;
  exitEditMode: () => void;
  setTool: (tool: MaskEditTool) => void;
  setBrushSize: (size: number) => void;
  setTolerance: (tolerance: number) => void;
  applyEdit: (newMask: ImageBitmap) => void;
  undo: () => void;
  redo: () => void;

  brushStroke: (points: [number, number][], isErasing: boolean) => Promise<void>;
  smartBrushStroke: (points: [number, number][]) => Promise<void>;
  floodFillAt: (seedPoint: [number, number]) => Promise<void>;
  morphologicalOp: (operation: MorphologicalRequest['operation'], kernelSize?: number, iterations?: number) => Promise<void>;
}

export const useMaskEditStore = create<MaskEditState>((set, get) => ({
  isEditMode: false,
  frameIndex: -1,
  workingMask: null,
  activeTool: 'brush',
  brushSize: 5,
  tolerance: 30,
  undoStack: [],
  redoStack: [],
  isSaving: false,

  enterEditMode: (frameIndex, mask) =>
    set({ isEditMode: true, frameIndex, workingMask: mask, undoStack: [], redoStack: [] }),

  exitEditMode: () => {
    const { undoStack, redoStack } = get();
    undoStack.forEach((bmp) => bmp.close());
    redoStack.forEach((bmp) => bmp.close());
    set({
      isEditMode: false,
      frameIndex: -1,
      workingMask: null,
      undoStack: [],
      redoStack: [],
    });
  },

  setTool: (tool) => set({ activeTool: tool }),
  setBrushSize: (size) => set({ brushSize: Math.max(1, Math.min(50, size)) }),
  setTolerance: (tolerance) => set({ tolerance: Math.max(0, Math.min(255, tolerance)) }),

  applyEdit: (newMask) => {
    const { workingMask, redoStack } = get();
    if (!workingMask) return;

    redoStack.forEach((bmp) => bmp.close());

    set((state) => {
      const newUndo = [...state.undoStack, workingMask];
      // Evict oldest entries if stack exceeds max
      while (newUndo.length > MAX_UNDO_STACK) {
        const evicted = newUndo.shift();
        evicted?.close();
      }
      return {
        workingMask: newMask,
        undoStack: newUndo,
        redoStack: [],
      };
    });
  },

  undo: () => {
    const { undoStack, workingMask } = get();
    if (undoStack.length === 0 || !workingMask) return;
    const previous = undoStack[undoStack.length - 1];
    set((state) => ({
      workingMask: previous,
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [workingMask, ...state.redoStack],
    }));
  },

  redo: () => {
    const { redoStack, workingMask } = get();
    if (redoStack.length === 0 || !workingMask) return;
    const next = redoStack[0];
    set((state) => ({
      workingMask: next,
      redoStack: state.redoStack.slice(1),
      undoStack: [...state.undoStack, workingMask],
    }));
  },

  brushStroke: async (points, isErasing) => {
    const { frameIndex, brushSize, applyEdit } = get();
    const newMask = await applyBrush({
      frame_index: frameIndex,
      points,
      radius: brushSize,
      is_erasing: isErasing,
    });
    applyEdit(newMask);
  },

  smartBrushStroke: async (points) => {
    const { frameIndex, brushSize, tolerance, applyEdit } = get();
    const newMask = await applySmartBrush({
      frame_index: frameIndex,
      points,
      radius: brushSize,
      tolerance,
    });
    applyEdit(newMask);
  },

  floodFillAt: async (seedPoint) => {
    const { frameIndex, tolerance, applyEdit } = get();
    const newMask = await applyFloodFill({
      frame_index: frameIndex,
      seed_point: seedPoint,
      tolerance,
    });
    applyEdit(newMask);
  },

  morphologicalOp: async (operation, kernelSize = 3, iterations = 1) => {
    const { frameIndex, applyEdit } = get();
    const newMask = await applyMorphological({
      frame_index: frameIndex,
      operation,
      kernel_size: kernelSize,
      iterations,
    });
    applyEdit(newMask);
  },
}));
