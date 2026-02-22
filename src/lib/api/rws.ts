import { apiClient } from '@/lib/api/client';

export async function calculateRWS(startFrame: number, endFrame: number, outlierMethod = 'hampel', vessel?: string) {
  const res = await apiClient.post('/rws/calculate', {
    start_frame: startFrame, end_frame: endFrame, outlier_method: outlierMethod, vessel,
  });
  return res.data;
}

export async function getRWSResults() {
  const res = await apiClient.get('/rws/results');
  return res.data;
}

export async function deleteRWSResult(index: number) {
  await apiClient.delete(`/rws/results/${index}`);
}
