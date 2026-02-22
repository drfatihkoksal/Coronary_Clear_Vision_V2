import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect, useState, useCallback } from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { Toolbar } from '@/components/layout/Toolbar';
import { ViewerContainer } from '@/components/viewer/ViewerContainer';
import { QFRDualViewer } from '@/components/viewer/QFRDualViewer';
import { PlaybackControls } from '@/components/controls/PlaybackControls';
import { SegmentationPanel } from '@/components/panels/SegmentationPanel';
import { QCAPanel } from '@/components/panels/QCAPanel';
import { ECGChart } from '@/components/charts/ECGChart';
import { RWSPanel } from '@/components/panels/RWSPanel';
import { TrackingPanel } from '@/components/panels/TrackingPanel';
import { QFRPanel } from '@/components/panels/QFRPanel';
import { CalibrationPanel } from '@/components/panels/CalibrationPanel';
import { ExportPanel } from '@/components/panels/ExportPanel';
import { MaskEditToolbar } from '@/components/MaskEdit/MaskEditToolbar';
import { KeyboardHelp } from '@/components/common/KeyboardHelp';
import { useSessionStore } from '@/stores/sessionStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useMaskEditStore } from '@/stores/maskEditStore';
import { useQFRStore } from '@/stores/qfrStore';
import '@/stores/timingStore'; // side-effect: registers eventBus listeners

type RightPanelTab = 'segmentation' | 'qca' | 'calibration' | 'rws' | 'tracking' | 'qfr' | 'export';

function AnalysisApp() {
  const startHealthPolling = useSessionStore((s) => s.startHealthPolling);
  const applyTheme = useSettingsStore((s) => s.applyTheme);
  const [activeTab, setActiveTab] = useState<RightPanelTab>('segmentation');
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const isEditMode = useMaskEditStore((s) => s.isEditMode);
  const exitEditMode = useMaskEditStore((s) => s.exitEditMode);
  const isQfrMode = useQFRStore((s) => s.isQfrMode);

  const handleMaskSave = () => {
    // Mask is already persisted server-side on each edit; just exit edit mode
    exitEditMode();
  };

  const handleMaskCancel = () => {
    exitEditMode();
  };

  const toggleKeyboardHelp = useCallback(() => {
    setShowKeyboardHelp((prev) => !prev);
  }, []);

  useEffect(() => {
    applyTheme();
  }, [applyTheme]);

  useEffect(() => {
    const cleanup = startHealthPolling();
    return cleanup;
  }, [startHealthPolling]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === '?') {
        toggleKeyboardHelp();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [toggleKeyboardHelp]);

  const tabClass = (tab: RightPanelTab) =>
    `flex-1 px-3 py-2 text-xs font-medium transition-colors ${
      activeTab === tab
        ? 'text-brand border-b-2 border-brand bg-surface-primary'
        : 'text-content-secondary hover:text-content-primary'
    }`;

  return (
    <div className="flex flex-col h-screen bg-surface-primary">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Toolbar />
        <main className="flex-1 flex flex-col overflow-hidden relative">
          {isEditMode && (
            <MaskEditToolbar onSave={handleMaskSave} onCancel={handleMaskCancel} />
          )}
          {isQfrMode ? (
            <QFRDualViewer />
          ) : (
            <>
              <ViewerContainer />
              {/* Chart area below viewer */}
              <div className="border-t border-border bg-surface-secondary">
                <ECGChart />
              </div>
              <PlaybackControls />
            </>
          )}
        </main>
        <aside className="w-right-panel bg-surface-secondary border-l border-border overflow-y-auto shrink-0">
          <div className="flex border-b border-border" role="tablist">
            <button className={tabClass('segmentation')} onClick={() => setActiveTab('segmentation')} role="tab" aria-selected={activeTab === 'segmentation'}>
              Segment
            </button>
            <button className={tabClass('qca')} onClick={() => setActiveTab('qca')} role="tab" aria-selected={activeTab === 'qca'}>
              QCA
            </button>
            <button className={tabClass('calibration')} onClick={() => setActiveTab('calibration')} role="tab" aria-selected={activeTab === 'calibration'}>
              Calib
            </button>
            <button className={tabClass('rws')} onClick={() => setActiveTab('rws')} role="tab" aria-selected={activeTab === 'rws'}>
              RWS
            </button>
            <button className={tabClass('tracking')} onClick={() => setActiveTab('tracking')} role="tab" aria-selected={activeTab === 'tracking'}>
              Track
            </button>
            <button className={tabClass('qfr')} onClick={() => setActiveTab('qfr')} role="tab" aria-selected={activeTab === 'qfr'}>
              QFR
            </button>
            <button className={tabClass('export')} onClick={() => setActiveTab('export')} role="tab" aria-selected={activeTab === 'export'}>
              Export
            </button>
          </div>
          {activeTab === 'segmentation' && <SegmentationPanel />}
          {activeTab === 'qca' && <QCAPanel />}
          {activeTab === 'calibration' && <CalibrationPanel />}
          {activeTab === 'rws' && <RWSPanel />}
          {activeTab === 'tracking' && <TrackingPanel />}
          {activeTab === 'qfr' && <QFRPanel />}
          {activeTab === 'export' && <ExportPanel />}
        </aside>
      </div>
      <StatusBar />
      {showKeyboardHelp && <KeyboardHelp onClose={toggleKeyboardHelp} />}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/app" replace />} />
        <Route path="/app" element={<AnalysisApp />} />
        <Route path="*" element={<Navigate to="/app" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
