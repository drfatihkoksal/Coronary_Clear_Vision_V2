import { create } from 'zustand';
import { apiClient, setSessionId } from '@/lib/api/client';

interface SessionState {
  sessionId: string | null;
  isConnected: boolean;
  isChecking: boolean;
  backendVersion: string | null;

  checkConnection: () => Promise<void>;
  startHealthPolling: () => () => void;
  setSession: (id: string) => void;
  clearSession: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  isConnected: false,
  isChecking: false,
  backendVersion: null,

  checkConnection: async () => {
    set({ isChecking: true });
    try {
      const res = await apiClient.get('/health');
      set({ isConnected: true, isChecking: false, backendVersion: res.data.version });
    } catch {
      set({ isConnected: false, isChecking: false, backendVersion: null });
    }
  },

  startHealthPolling: () => {
    get().checkConnection();
    const interval = setInterval(() => get().checkConnection(), 30000);
    return () => clearInterval(interval);
  },

  setSession: (id: string) => {
    setSessionId(id);
    set({ sessionId: id });
  },

  clearSession: () => {
    setSessionId(null);
    set({ sessionId: null });
  },
}));
