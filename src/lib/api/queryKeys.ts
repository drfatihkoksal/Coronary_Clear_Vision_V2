export const queryKeys = {
  health: () => ['health'] as const,
  frame: (sessionId: string, index: number) => ['frame', sessionId, index] as const,
  metadata: (sessionId: string) => ['metadata', sessionId] as const,
  segmentation: (sessionId: string, frame: number) => ['segmentation', sessionId, frame] as const,
  qca: (sessionId: string, frame: number) => ['qca', sessionId, frame] as const,
  rws: (sessionId: string) => ['rws', sessionId] as const,
  ecg: (sessionId: string) => ['ecg', sessionId] as const,
  motion: (sessionId: string) => ['motion', sessionId] as const,
  engines: () => ['engines'] as const,
};
