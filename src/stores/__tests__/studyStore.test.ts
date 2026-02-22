import { describe, it, expect, beforeEach } from 'vitest';
import { useStudyStore } from '../studyStore';

describe('studyStore', () => {
  beforeEach(() => {
    useStudyStore.setState({
      metadata: null,
      isLoaded: false,
      isLoading: false,
      error: null,
      frameCache: new Map(),
    });
  });

  it('should have no metadata initially', () => {
    const state = useStudyStore.getState();
    expect(state.metadata).toBeNull();
    expect(state.isLoaded).toBe(false);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.frameCache.size).toBe(0);
  });

  it('should clear state on reset', () => {
    useStudyStore.setState({
      metadata: {
        sessionId: 'test',
        patient: { patientId: null, name: null, birthDate: null, sex: null, age: null },
        studyInfo: {
          studyInstanceUid: null,
          seriesInstanceUid: null,
          studyDate: null,
          studyTime: null,
          description: null,
          institution: null,
          modality: 'XA',
        },
        numFrames: 10,
        frameRate: 15,
        imageWidth: 512,
        imageHeight: 512,
        pixelSpacing: null,
      },
      isLoaded: true,
      isLoading: false,
      error: null,
      frameCache: new Map(),
    });

    useStudyStore.getState().reset();
    const state = useStudyStore.getState();
    expect(state.metadata).toBeNull();
    expect(state.isLoaded).toBe(false);
    expect(state.frameCache.size).toBe(0);
  });
});
