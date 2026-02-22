import { apiClient } from '@/lib/api/client';

export async function fetchECGSignal() {
  const res = await apiClient.get('/ecg/signal');
  return res.data;
}

export async function updateRPeaks(rPeaks: number[]) {
  const res = await apiClient.post('/ecg/r-peaks', { r_peaks: rPeaks });
  return res.data;
}

export async function addRPeak(sampleIndex: number) {
  const res = await apiClient.post('/ecg/r-peaks/add', { sample_index: sampleIndex });
  return res.data;
}

export async function removeRPeak(sampleIndex: number) {
  const res = await apiClient.post('/ecg/r-peaks/remove', { sample_index: sampleIndex });
  return res.data;
}

export async function moveRPeak(fromIndex: number, toIndex: number) {
  const res = await apiClient.post('/ecg/r-peaks/move', { from_index: fromIndex, to_index: toIndex });
  return res.data;
}
