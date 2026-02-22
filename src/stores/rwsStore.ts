import { create } from 'zustand';
import type { RWSResult, RWSInterpretation } from '@/types';
import { calculateRWS, deleteRWSResult } from '@/lib/api/rws';
import { eventBus } from '@/lib/eventBus';

interface RWSSummary {
  meanRwsPct: number;
  medianRwsPct: number;
  interpretation: RWSInterpretation;
  numBeats: number;
}

interface RWSState {
  results: RWSResult[];
  summary: RWSSummary | null;
  isCalculating: boolean;
  error: string | null;
  startFrame: number | null;
  endFrame: number | null;
  selectedBeat: number | null;
  outlierMethod: 'none' | 'hampel' | 'double_hampel';
  vessel: string | null;

  calculate: () => Promise<void>;
  setRange: (start: number, end: number, beatNumber?: number) => void;
  setSelectedBeat: (beat: number | null) => void;
  setOutlierMethod: (method: 'none' | 'hampel' | 'double_hampel') => void;
  setVessel: (vessel: string | null) => void;
  removeResult: (index: number) => Promise<void>;
  clearResults: () => void;
}

function computeSummary(results: RWSResult[]): RWSSummary | null {
  if (results.length === 0) return null;
  const values = results.map((r) => r.mldRwsPct);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];

  let interpretation: RWSInterpretation;
  if (median < 8) interpretation = 'normal';
  else if (median < 12) interpretation = 'intermediate';
  else if (median < 14) interpretation = 'vulnerable';
  else interpretation = 'high_risk';

  return { meanRwsPct: mean, medianRwsPct: median, interpretation, numBeats: results.length };
}

export const useRWSStore = create<RWSState>((set, get) => ({
  results: [],
  summary: null,
  isCalculating: false,
  error: null,
  startFrame: null,
  endFrame: null,
  selectedBeat: null,
  outlierMethod: 'hampel',
  vessel: null,

  calculate: async () => {
    const { startFrame, endFrame, selectedBeat, outlierMethod, vessel } = get();
    if (startFrame == null || endFrame == null) return;
    set({ isCalculating: true, error: null });
    try {
      const data = await calculateRWS(startFrame, endFrame, outlierMethod, vessel ?? undefined);
      const result: RWSResult = {
        beatNumber: selectedBeat ?? get().results.length + 1,
        startFrame: data.start_frame,
        endFrame: data.end_frame,
        mldRwsPct: data.mld_rws_pct,
        proximalRwsPct: data.proximal_rws_pct,
        distalRwsPct: data.distal_rws_pct,
        averageRwsPct: data.average_rws_pct,
        interpretation: data.interpretation,
        outlierMethod: data.outlier_method,
        vessel: data.vessel,
      };
      const newResults = [...get().results, result];
      set({ results: newResults, summary: computeSummary(newResults), isCalculating: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'RWS calculation failed';
      set({ isCalculating: false, error: msg });
    }
  },

  setRange: (start: number, end: number, beatNumber?: number) =>
    set({ startFrame: start, endFrame: end, selectedBeat: beatNumber ?? null }),
  setSelectedBeat: (beat: number | null) => set({ selectedBeat: beat }),
  setOutlierMethod: (method) => set({ outlierMethod: method }),
  setVessel: (vessel) => set({ vessel }),

  removeResult: async (index: number) => {
    try {
      await deleteRWSResult(index);
    } catch {
      // ignore server error on delete; remove locally anyway
    }
    const newResults = get().results.filter((_, i) => i !== index);
    set({ results: newResults, summary: computeSummary(newResults) });
  },

  clearResults: () => set({ results: [], summary: null, error: null }),
}));

// Listen for beat:selected events to auto-set range
eventBus.on('beat:selected', ({ startFrame, endFrame }) => {
  useRWSStore.getState().setRange(startFrame, endFrame);
});

// Clear all RWS state when study is cleared
eventBus.on('study:cleared', () => {
  const store = useRWSStore.getState();
  store.clearResults();
  store.setRange(0, 0);
  // Reset to defaults
  useRWSStore.setState({ startFrame: null, endFrame: null, selectedBeat: null, vessel: null });
});
