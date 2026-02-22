import { describe, it, expect, beforeEach } from 'vitest';
import { usePlayerStore } from '../playerStore';

describe('playerStore', () => {
  beforeEach(() => {
    usePlayerStore.setState({
      currentFrame: 0,
      totalFrames: 0,
      frameRate: 15,
      playbackState: 'stopped',
      playbackSpeed: 1.0,
      isLooping: true,
    });
  });

  it('should have correct initial state', () => {
    const state = usePlayerStore.getState();
    expect(state.currentFrame).toBe(0);
    expect(state.totalFrames).toBe(0);
    expect(state.frameRate).toBe(15);
    expect(state.playbackState).toBe('stopped');
    expect(state.playbackSpeed).toBe(1.0);
    expect(state.isLooping).toBe(true);
  });

  it('should toggle play/pause', () => {
    usePlayerStore.getState().togglePlayPause();
    expect(usePlayerStore.getState().playbackState).toBe('playing');

    usePlayerStore.getState().togglePlayPause();
    expect(usePlayerStore.getState().playbackState).toBe('paused');
  });

  it('should step forward', () => {
    usePlayerStore.getState().setTotalFrames(10);
    usePlayerStore.getState().stepForward();
    expect(usePlayerStore.getState().currentFrame).toBe(1);
  });

  it('should step backward', () => {
    usePlayerStore.getState().setTotalFrames(10);
    usePlayerStore.getState().goToFrame(5);
    usePlayerStore.getState().stepBackward();
    expect(usePlayerStore.getState().currentFrame).toBe(4);
  });

  it('should wrap around when looping and stepping forward past end', () => {
    usePlayerStore.getState().setTotalFrames(10);
    usePlayerStore.getState().goToFrame(9);
    usePlayerStore.getState().stepForward();
    expect(usePlayerStore.getState().currentFrame).toBe(0);
  });

  it('should wrap around when looping and stepping backward past start', () => {
    usePlayerStore.getState().setTotalFrames(10);
    usePlayerStore.getState().goToFrame(0);
    usePlayerStore.getState().stepBackward();
    expect(usePlayerStore.getState().currentFrame).toBe(9);
  });

  it('should clamp at boundaries when not looping', () => {
    usePlayerStore.setState({ isLooping: false });
    usePlayerStore.getState().setTotalFrames(10);
    usePlayerStore.getState().goToFrame(9);
    usePlayerStore.getState().stepForward();
    expect(usePlayerStore.getState().currentFrame).toBe(9);
  });

  it('should not step when totalFrames is 0', () => {
    usePlayerStore.getState().stepForward();
    expect(usePlayerStore.getState().currentFrame).toBe(0);
    usePlayerStore.getState().stepBackward();
    expect(usePlayerStore.getState().currentFrame).toBe(0);
  });
});
