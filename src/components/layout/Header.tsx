import { useStudyStore } from '@/stores/studyStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useQFRStore } from '@/stores/qfrStore';
import type { ThemeMode } from '@/types';

export function Header() {
  const metadata = useStudyStore((s) => s.metadata);
  const theme = useSettingsStore((s) => s.theme);
  const setTheme = useSettingsStore((s) => s.setTheme);
  const isQfrMode = useQFRStore((s) => s.isQfrMode);
  const disableQfrMode = useQFRStore((s) => s.disableQfrMode);

  const cycleTheme = () => {
    const order: ThemeMode[] = ['dark', 'light', 'system'];
    const idx = order.indexOf(theme);
    setTheme(order[(idx + 1) % order.length]);
  };

  return (
    <header className="h-header flex items-center px-4 bg-surface-secondary border-b border-border shrink-0">
      <h1 className="text-lg font-semibold text-content-primary">Coronary RWS Analyser</h1>
      <span className="ml-2 text-xs text-content-muted">v2.0.0</span>

      {metadata && (
        <div className="ml-6 flex items-center gap-3 text-xs text-content-secondary">
          <span>{metadata.imageWidth}x{metadata.imageHeight}</span>
          <span>{metadata.numFrames} frames</span>
          <span>{metadata.frameRate.toFixed(1)} fps</span>
        </div>
      )}

      <div className="ml-auto flex items-center gap-2">
        {isQfrMode && (
          <button
            onClick={disableQfrMode}
            className="px-3 py-1 text-xs font-medium rounded bg-red-600/90 text-white hover:bg-red-700 transition-colors"
          >
            Exit QFR Mode
          </button>
        )}
        <button
          onClick={cycleTheme}
          className="px-2 py-1 text-xs text-content-secondary hover:text-content-primary rounded hover:bg-surface-tertiary transition-colors"
          title={`Theme: ${theme}`}
        >
          {theme === 'dark' ? '\u{1F319}' : theme === 'light' ? '\u{2600}\u{FE0F}' : '\u{1F5A5}\u{FE0F}'}
        </button>
      </div>
    </header>
  );
}
