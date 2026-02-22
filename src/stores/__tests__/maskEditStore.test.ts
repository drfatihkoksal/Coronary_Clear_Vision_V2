import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useMaskEditStore } from '@/stores/maskEditStore';

// Mock the API module
vi.mock('@/lib/api/mask-edit', () => ({
  applyBrush: vi.fn(),
  applySmartBrush: vi.fn(),
  applyFloodFill: vi.fn(),
  applyMorphological: vi.fn(),
}));

function makeMockBitmap(id = 0): ImageBitmap {
  return { close: vi.fn(), width: 64, height: 64, _id: id } as unknown as ImageBitmap;
}

describe('maskEditStore', () => {
  beforeEach(() => {
    useMaskEditStore.getState().exitEditMode();
  });

  it('should have correct initial state', () => {
    const state = useMaskEditStore.getState();
    expect(state.isEditMode).toBe(false);
    expect(state.workingMask).toBeNull();
    expect(state.activeTool).toBe('brush');
    expect(state.brushSize).toBe(5);
    expect(state.undoStack).toHaveLength(0);
    expect(state.redoStack).toHaveLength(0);
  });

  it('should enter edit mode', () => {
    const mask = makeMockBitmap();
    useMaskEditStore.getState().enterEditMode(3, mask);

    const state = useMaskEditStore.getState();
    expect(state.isEditMode).toBe(true);
    expect(state.frameIndex).toBe(3);
    expect(state.workingMask).toBe(mask);
  });

  it('should exit edit mode and clear state', () => {
    const mask = makeMockBitmap();
    useMaskEditStore.getState().enterEditMode(0, mask);
    useMaskEditStore.getState().exitEditMode();

    const state = useMaskEditStore.getState();
    expect(state.isEditMode).toBe(false);
    expect(state.workingMask).toBeNull();
    expect(state.undoStack).toHaveLength(0);
    expect(state.redoStack).toHaveLength(0);
  });

  it('should set tool', () => {
    useMaskEditStore.getState().setTool('eraser');
    expect(useMaskEditStore.getState().activeTool).toBe('eraser');
  });

  it('should clamp brush size', () => {
    useMaskEditStore.getState().setBrushSize(100);
    expect(useMaskEditStore.getState().brushSize).toBe(50);

    useMaskEditStore.getState().setBrushSize(-5);
    expect(useMaskEditStore.getState().brushSize).toBe(1);
  });

  it('should apply edit with undo stack', () => {
    const mask1 = makeMockBitmap(1);
    const mask2 = makeMockBitmap(2);
    useMaskEditStore.getState().enterEditMode(0, mask1);

    useMaskEditStore.getState().applyEdit(mask2);

    const state = useMaskEditStore.getState();
    expect(state.workingMask).toBe(mask2);
    expect(state.undoStack).toHaveLength(1);
    expect(state.undoStack[0]).toBe(mask1);
    expect(state.redoStack).toHaveLength(0);
  });

  it('should undo and redo', () => {
    const mask1 = makeMockBitmap(1);
    const mask2 = makeMockBitmap(2);
    useMaskEditStore.getState().enterEditMode(0, mask1);
    useMaskEditStore.getState().applyEdit(mask2);

    // Undo
    useMaskEditStore.getState().undo();
    expect(useMaskEditStore.getState().workingMask).toBe(mask1);
    expect(useMaskEditStore.getState().redoStack).toHaveLength(1);

    // Redo
    useMaskEditStore.getState().redo();
    expect(useMaskEditStore.getState().workingMask).toBe(mask2);
    expect(useMaskEditStore.getState().undoStack).toHaveLength(1);
  });

  it('should not undo when stack is empty', () => {
    const mask1 = makeMockBitmap(1);
    useMaskEditStore.getState().enterEditMode(0, mask1);
    useMaskEditStore.getState().undo();
    expect(useMaskEditStore.getState().workingMask).toBe(mask1);
  });

  it('should limit undo stack to 20 entries', () => {
    const initial = makeMockBitmap(0);
    useMaskEditStore.getState().enterEditMode(0, initial);

    for (let i = 1; i <= 25; i++) {
      useMaskEditStore.getState().applyEdit(makeMockBitmap(i));
    }

    expect(useMaskEditStore.getState().undoStack.length).toBeLessThanOrEqual(20);
  });
});
