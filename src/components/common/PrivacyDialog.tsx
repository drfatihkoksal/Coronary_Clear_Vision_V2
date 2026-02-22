import { useCallback, useEffect, useRef } from 'react';
import { ShieldCheck } from 'lucide-react';

interface PrivacyDialogProps {
  onConfirm: (anonymize: boolean) => void;
  onCancel: () => void;
}

export function PrivacyDialog({ onConfirm, onCancel }: PrivacyDialogProps) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  // Focus the confirm button on mount
  useEffect(() => {
    confirmRef.current?.focus();
  }, []);

  // Allow Escape to cancel
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    },
    [onCancel],
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onKeyDown={handleKeyDown}
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-label="HIPAA Privacy Notice"
    >
      <div
        className="bg-surface-primary rounded-lg shadow-xl max-w-md w-full mx-4 p-6 border border-border"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 mb-4">
          <ShieldCheck size={24} className="text-brand shrink-0" />
          <h2 className="text-lg font-semibold text-content-primary">
            HIPAA Privacy Notice
          </h2>
        </div>

        <p className="text-sm text-content-secondary mb-6 leading-relaxed">
          You can anonymize patient data (name, ID, date of birth) before
          processing, or keep the original DICOM metadata intact.
        </p>

        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm rounded border border-border text-content-secondary
              hover:bg-surface-tertiary transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(false)}
            className="px-4 py-2 text-sm rounded border border-border text-content-secondary
              hover:bg-surface-tertiary transition-colors cursor-pointer"
          >
            Keep Original
          </button>
          <button
            ref={confirmRef}
            onClick={() => onConfirm(true)}
            className="px-4 py-2 text-sm rounded bg-brand text-white
              hover:bg-brand/90 transition-colors cursor-pointer"
          >
            Anonymize &amp; Upload
          </button>
        </div>
      </div>
    </div>
  );
}
