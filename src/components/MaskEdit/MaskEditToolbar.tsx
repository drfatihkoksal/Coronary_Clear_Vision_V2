import { useMaskEditStore } from '@/stores/maskEditStore';
import type { MaskEditTool } from '@/stores/maskEditStore';

const TOOLS: { value: MaskEditTool; label: string; shortcut: string }[] = [
  { value: 'brush', label: 'Brush', shortcut: 'B' },
  { value: 'eraser', label: 'Eraser', shortcut: 'E' },
  { value: 'smartBrush', label: 'Smart', shortcut: 'S' },
  { value: 'floodFill', label: 'Fill', shortcut: 'F' },
];

interface MaskEditToolbarProps {
  onSave: () => void;
  onCancel: () => void;
}

export function MaskEditToolbar({ onSave, onCancel }: MaskEditToolbarProps) {
  const {
    activeTool, brushSize, tolerance, undoStack, redoStack,
    setTool, setBrushSize, setTolerance, undo, redo,
    morphologicalOp,
  } = useMaskEditStore();

  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-secondary border border-border shadow-lg">
      {/* Tool buttons */}
      {TOOLS.map((t) => (
        <button
          key={t.value}
          onClick={() => setTool(t.value)}
          className={`px-2 py-1.5 text-xs rounded border transition-colors ${
            activeTool === t.value
              ? 'border-brand bg-brand/10 text-brand'
              : 'border-border text-content-secondary hover:border-content-muted'
          }`}
          title={`${t.label} (${t.shortcut})`}
        >
          {t.label}
        </button>
      ))}

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* Brush size */}
      <div className="flex items-center gap-1">
        <label className="text-xs text-content-secondary">Size</label>
        <input
          type="range"
          min={1}
          max={50}
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
          className="w-16 h-1 accent-brand"
        />
        <span className="text-xs text-content-muted w-5 text-right">{brushSize}</span>
      </div>

      {/* Tolerance (for smart brush / flood fill) */}
      {(activeTool === 'smartBrush' || activeTool === 'floodFill') && (
        <div className="flex items-center gap-1">
          <label className="text-xs text-content-secondary">Tol</label>
          <input
            type="range"
            min={0}
            max={255}
            value={tolerance}
            onChange={(e) => setTolerance(Number(e.target.value))}
            className="w-16 h-1 accent-brand"
          />
          <span className="text-xs text-content-muted w-6 text-right">{tolerance}</span>
        </div>
      )}

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* Morphological ops */}
      <button
        onClick={() => morphologicalOp('dilate')}
        className="px-2 py-1.5 text-xs rounded border border-border text-content-secondary hover:border-content-muted transition-colors"
        title="Dilate mask"
      >
        Dilate
      </button>
      <button
        onClick={() => morphologicalOp('erode')}
        className="px-2 py-1.5 text-xs rounded border border-border text-content-secondary hover:border-content-muted transition-colors"
        title="Erode mask"
      >
        Erode
      </button>

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* Undo / Redo */}
      <button
        onClick={undo}
        disabled={undoStack.length === 0}
        className="px-2 py-1.5 text-xs rounded border border-border text-content-secondary hover:border-content-muted disabled:opacity-30 transition-colors"
        title="Undo (Ctrl+Z)"
      >
        Undo
      </button>
      <button
        onClick={redo}
        disabled={redoStack.length === 0}
        className="px-2 py-1.5 text-xs rounded border border-border text-content-secondary hover:border-content-muted disabled:opacity-30 transition-colors"
        title="Redo (Ctrl+Y)"
      >
        Redo
      </button>

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* Save / Cancel */}
      <button
        onClick={onCancel}
        className="px-3 py-1.5 text-xs rounded border border-border text-content-secondary hover:border-red-400 hover:text-red-400 transition-colors"
      >
        Cancel
      </button>
      <button
        onClick={onSave}
        className="px-3 py-1.5 text-xs rounded bg-brand text-white hover:bg-blue-600 transition-colors"
      >
        Save
      </button>
    </div>
  );
}
