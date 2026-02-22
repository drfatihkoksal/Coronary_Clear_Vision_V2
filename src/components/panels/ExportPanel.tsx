import { useState } from 'react';
import { exportCSV, exportJSON } from '@/lib/api/export';

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function ExportPanel() {
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [anonymize, setAnonymize] = useState(true);

  const handleExport = async (format: 'csv' | 'json') => {
    setIsExporting(true);
    setError(null);
    try {
      const blob = format === 'csv' ? await exportCSV() : await exportJSON();
      const ext = format;
      downloadBlob(blob, `rws_analysis.${ext}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setError(msg);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="p-3 space-y-3">
      <h3 className="text-sm font-semibold text-content-primary">Export Results</h3>

      <p className="text-xs text-content-secondary">
        Download QCA and RWS analysis results. Exported files include a research-use-only disclaimer.
      </p>

      {/* Anonymize toggle */}
      <label className="flex items-center gap-2 text-xs text-content-secondary cursor-pointer">
        <input
          type="checkbox"
          checked={anonymize}
          onChange={(e) => setAnonymize(e.target.checked)}
          className="rounded border-border"
        />
        Anonymize export
      </label>

      {/* Export buttons */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => handleExport('csv')}
          disabled={isExporting}
          className="py-2 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isExporting ? 'Exporting...' : 'Export CSV'}
        </button>
        <button
          onClick={() => handleExport('json')}
          disabled={isExporting}
          className="py-2 text-sm font-medium rounded border border-brand text-brand hover:bg-brand/10 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isExporting ? 'Exporting...' : 'Export JSON'}
        </button>
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      <div className="text-[10px] text-content-muted leading-tight">
        For Research Use Only. Not validated for clinical decision-making.
      </div>
    </div>
  );
}
