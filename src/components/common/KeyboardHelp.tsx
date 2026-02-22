import { useEffect, useCallback } from 'react';

interface KeyboardHelpProps {
  onClose: () => void;
}

const shortcuts: { key: string; description: string }[] = [
  { key: 'Space', description: 'Play / Pause' },
  { key: '\u2190 / \u2192', description: 'Step backward / forward' },
  { key: 'Home / End', description: 'First / Last frame' },
  { key: '+ / \u2013', description: 'Speed up / slow down' },
  { key: 'R', description: 'Toggle loop' },
  { key: 'H', description: 'Toggle header' },
  { key: 'B', description: 'ROI tool' },
  { key: 'S', description: 'Seed tool' },
  { key: 'Escape', description: 'Deselect tool' },
  { key: 'Ctrl+Z / Y', description: 'Undo / Redo (mask edit)' },
  { key: '?', description: 'Show this help' },
];

export function KeyboardHelp({ onClose }: KeyboardHelpProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [handleKeyDown]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      <div className="bg-surface-secondary rounded-lg shadow-xl max-w-lg w-full mx-4 p-6">
        <h2 className="text-lg font-semibold text-content-primary mb-4">
          Keyboard Shortcuts
        </h2>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center gap-2 py-1">
              <kbd className="inline-block min-w-[4rem] px-2 py-0.5 text-xs font-mono text-content-primary bg-surface-tertiary border border-border rounded text-center">
                {s.key}
              </kbd>
              <span className="text-sm text-content-secondary">{s.description}</span>
            </div>
          ))}
        </div>
        <div className="mt-5 text-right">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-content-secondary hover:text-content-primary transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
