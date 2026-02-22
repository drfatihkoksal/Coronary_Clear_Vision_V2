import { useEffect, useRef } from 'react';

interface ViewerContextMenuProps {
  x: number;
  y: number;
  currentFrame: number;
  onClose: () => void;
  onSetStartFrame: (frame: number) => void;
  onSetEndFrame: (frame: number) => void;
}

export function ViewerContextMenu({
  x,
  y,
  currentFrame,
  onClose,
  onSetStartFrame,
  onSetEndFrame,
}: ViewerContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('mousedown', handleClickOutside);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('mousedown', handleClickOutside);
    };
  }, [onClose]);

  const items = [
    { label: `Set as RWS Start Frame (${currentFrame})`, action: () => onSetStartFrame(currentFrame) },
    { label: `Set as RWS End Frame (${currentFrame})`, action: () => onSetEndFrame(currentFrame) },
  ];

  return (
    <div
      ref={menuRef}
      className="absolute z-50 min-w-[200px] py-1 rounded border border-border bg-surface-secondary shadow-lg"
      style={{ left: x, top: y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {items.map((item) => (
        <button
          key={item.label}
          onClick={() => {
            item.action();
            onClose();
          }}
          className="w-full px-3 py-1.5 text-left text-xs text-content-primary hover:bg-brand/10 hover:text-brand transition-colors"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
