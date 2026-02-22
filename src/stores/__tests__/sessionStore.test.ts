import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { useSessionStore } from '../sessionStore';

describe('sessionStore', () => {
  beforeEach(() => {
    useSessionStore.setState({
      sessionId: null,
      isConnected: false,
      isChecking: false,
      backendVersion: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should have no session initially', () => {
    const state = useSessionStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.isConnected).toBe(false);
    expect(state.backendVersion).toBeNull();
  });

  it('should update sessionId on setSession', () => {
    useSessionStore.getState().setSession('test-session-123');
    const state = useSessionStore.getState();
    expect(state.sessionId).toBe('test-session-123');
  });

  it('should clear sessionId on clearSession', () => {
    useSessionStore.getState().setSession('test-session-123');
    useSessionStore.getState().clearSession();
    const state = useSessionStore.getState();
    expect(state.sessionId).toBeNull();
  });

  it('should return a cleanup function from startHealthPolling', () => {
    vi.useFakeTimers();
    const cleanup = useSessionStore.getState().startHealthPolling();
    expect(typeof cleanup).toBe('function');
    cleanup();
    vi.useRealTimers();
  });
});
