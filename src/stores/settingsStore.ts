import { create } from 'zustand';
import type { ThemeMode } from '@/types';

interface SettingsState {
  theme: ThemeMode;

  setTheme: (theme: ThemeMode) => void;
  applyTheme: () => void;
}

function getStoredTheme(): ThemeMode {
  try {
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      return stored;
    }
  } catch {
    // localStorage not available
  }
  return 'dark';
}

function applyThemeToDOM(theme: ThemeMode) {
  const isDark =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  document.documentElement.classList.toggle('dark', isDark);
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  theme: getStoredTheme(),

  setTheme: (theme: ThemeMode) => {
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // localStorage not available
    }
    set({ theme });
    applyThemeToDOM(theme);
  },

  applyTheme: () => {
    applyThemeToDOM(get().theme);
  },
}));
