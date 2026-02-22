import { useStudyStore } from '@/stores/studyStore';
import { useToolStore } from '@/stores/toolStore';
import { useQFRStore } from '@/stores/qfrStore';
import { useRef, useState, type ReactNode } from 'react';
import { MousePointer, Crosshair, Square, Columns2, FolderOpen } from 'lucide-react';
import { PrivacyDialog } from '@/components/common/PrivacyDialog';

export function Toolbar() {
  const loadFile = useStudyStore((s) => s.loadFile);
  const isLoading = useStudyStore((s) => s.isLoading);
  const activeTool = useToolStore((s) => s.activeTool);
  const setActiveTool = useToolStore((s) => s.setActiveTool);
  const isQfrMode = useQFRStore((s) => s.isQfrMode);
  const enableQfrMode = useQFRStore((s) => s.enableQfrMode);
  const disableQfrMode = useQFRStore((s) => s.disableQfrMode);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  const handleFileOpen = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPendingFile(file);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handlePrivacyConfirm = async (anonymize: boolean) => {
    if (pendingFile) {
      await loadFile(pendingFile, anonymize);
      setPendingFile(null);
    }
  };

  const handlePrivacyCancel = () => {
    setPendingFile(null);
  };

  return (
    <>
      <aside className="w-toolbar bg-surface-secondary border-r border-border flex flex-col items-center py-2 gap-1 shrink-0">
        <input
          ref={fileInputRef}
          type="file"

          onChange={handleFileChange}
          className="hidden"
        />

        {/* File operations */}
        <ToolButton
          icon={<FolderOpen size={18} />}
          label="Open File"
          onClick={handleFileOpen}
          disabled={isLoading}
        />

        {/* Separator */}
        <div className="w-7 h-px bg-border my-1" />

        {/* Annotation tools */}
        <ToolButton
          icon={<MousePointer size={18} />}
          label="Pan [H]"
          onClick={() => setActiveTool('pan')}
          active={activeTool === 'pan'}
        />
        <ToolButton
          icon={<Crosshair size={18} />}
          label="Seed [S]"
          onClick={() => setActiveTool('seed')}
          active={activeTool === 'seed'}
        />
        <ToolButton
          icon={<Square size={18} />}
          label="ROI [B]"
          onClick={() => setActiveTool('roi')}
          active={activeTool === 'roi'}
        />

        {/* Separator */}
        <div className="w-7 h-px bg-border my-1" />

        {/* Mode toggles */}
        <ToolButton
          icon={<Columns2 size={18} />}
          label="QFR Mode"
          onClick={() => (isQfrMode ? disableQfrMode() : enableQfrMode())}
          active={isQfrMode}
        />
      </aside>

      {pendingFile && (
        <PrivacyDialog
          onConfirm={handlePrivacyConfirm}
          onCancel={handlePrivacyCancel}
        />
      )}
    </>
  );
}

function ToolButton({ icon, label, onClick, disabled = false, active = false }: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      className={`w-10 h-10 flex items-center justify-center rounded text-lg
        ${active ? 'bg-brand text-white' : 'text-content-secondary hover:bg-surface-tertiary hover:text-content-primary'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        transition-colors`}
    >
      {icon}
    </button>
  );
}
