import { create } from 'zustand';
import { eventBus } from '@/lib/eventBus';

export type PlaybackState = 'stopped' | 'playing' | 'paused';

interface PlayerState {
  currentFrame: number;
  totalFrames: number;
  frameRate: number;
  playbackState: PlaybackState;
  playbackSpeed: number;
  isLooping: boolean;
  rangeStart: number | null;
  rangeEnd: number | null;

  play: () => void;
  pause: () => void;
  stop: () => void;
  togglePlayPause: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  goToFrame: (frame: number) => void;
  setTotalFrames: (total: number) => void;
  setFrameRate: (rate: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  toggleLoop: () => void;
  setPlaybackRange: (start: number, end: number) => void;
  clearPlaybackRange: () => void;
}

export const usePlayerStore = create<PlayerState>((set, get) => ({
  currentFrame: 0,
  totalFrames: 0,
  frameRate: 15,
  playbackState: 'stopped',
  playbackSpeed: 1.0,
  isLooping: true,
  rangeStart: null,
  rangeEnd: null,

  play: () => set({ playbackState: 'playing' }),
  pause: () => set({ playbackState: 'paused' }),
  stop: () => {
    const { rangeStart } = get();
    set({ playbackState: 'stopped', currentFrame: rangeStart ?? 0 });
  },

  togglePlayPause: () => {
    const { playbackState } = get();
    if (playbackState === 'playing') {
      set({ playbackState: 'paused' });
    } else {
      set({ playbackState: 'playing' });
    }
  },

  stepForward: () => {
    const { currentFrame, totalFrames, isLooping, rangeStart, rangeEnd } = get();
    if (totalFrames === 0) return;
    const lo = rangeStart ?? 0;
    const hi = rangeEnd ?? totalFrames - 1;
    const next = currentFrame + 1;
    if (next > hi) {
      set({ currentFrame: isLooping ? lo : hi });
    } else {
      set({ currentFrame: next });
    }
  },

  stepBackward: () => {
    const { currentFrame, totalFrames, isLooping, rangeStart, rangeEnd } = get();
    if (totalFrames === 0) return;
    const lo = rangeStart ?? 0;
    const hi = rangeEnd ?? totalFrames - 1;
    const prev = currentFrame - 1;
    if (prev < lo) {
      set({ currentFrame: isLooping ? hi : lo });
    } else {
      set({ currentFrame: prev });
    }
  },

  goToFrame: (frame: number) => {
    const { totalFrames } = get();
    if (totalFrames === 0) return;
    const clamped = Math.max(0, Math.min(frame, totalFrames - 1));
    set({ currentFrame: clamped });
  },

  setTotalFrames: (total: number) => set({ totalFrames: total, currentFrame: 0, rangeStart: null, rangeEnd: null }),
  setFrameRate: (rate: number) => set({ frameRate: rate }),
  setPlaybackSpeed: (speed: number) => set({ playbackSpeed: speed }),
  toggleLoop: () => set((state) => ({ isLooping: !state.isLooping })),
  setPlaybackRange: (start: number, end: number) => set({ rangeStart: start, rangeEnd: end, currentFrame: start }),
  clearPlaybackRange: () => set({ rangeStart: null, rangeEnd: null }),
}));

// Auto-set player state when a study is loaded or cleared
eventBus.on('study:loaded', (data) => {
  const store = usePlayerStore.getState();
  store.setTotalFrames(data.metadata.numFrames);
  store.setFrameRate(data.metadata.frameRate);
});
eventBus.on('study:cleared', () => {
  usePlayerStore.getState().stop();
  usePlayerStore.getState().setTotalFrames(0);
});
