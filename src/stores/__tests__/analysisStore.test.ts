import { describe, it, expect, beforeEach } from 'vitest';
import { useAnalysisStore } from '@/stores/analysisStore';

describe('analysisStore', () => {
  beforeEach(() => {
    useAnalysisStore.getState().reset();
  });

  it('should have empty initial state', () => {
    const state = useAnalysisStore.getState();
    expect(state.frameData.size).toBe(0);
    expect(state.isSegmenting).toBe(false);
    expect(state.selectedEngine).toBe('nnunet');
  });

  it('should set engine', () => {
    useAnalysisStore.getState().setEngine('angiopy');
    expect(useAnalysisStore.getState().selectedEngine).toBe('angiopy');
  });

  it('should add and remove seed points', () => {
    const store = useAnalysisStore.getState();
    store.addSeedPoint(0, { x: 100, y: 200 });
    store.addSeedPoint(0, { x: 150, y: 250 });

    expect(useAnalysisStore.getState().seedPoints.get(0)?.length).toBe(2);

    useAnalysisStore.getState().removeSeedPoint(0, 0);
    expect(useAnalysisStore.getState().seedPoints.get(0)?.length).toBe(1);
  });

  it('should clear seed points', () => {
    useAnalysisStore.getState().addSeedPoint(0, { x: 100, y: 200 });
    useAnalysisStore.getState().clearSeedPoints(0);
    expect(useAnalysisStore.getState().seedPoints.get(0)).toBeUndefined();
  });

  it('should reset state', () => {
    useAnalysisStore.getState().setEngine('angiopy');
    useAnalysisStore.getState().addSeedPoint(0, { x: 1, y: 2 });
    useAnalysisStore.getState().reset();

    expect(useAnalysisStore.getState().selectedEngine).toBe('nnunet');
    expect(useAnalysisStore.getState().seedPoints.size).toBe(0);
  });
});
