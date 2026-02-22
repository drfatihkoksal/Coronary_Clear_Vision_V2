import { apiClient } from '@/lib/api/client';

export async function exportCSV(): Promise<Blob> {
  const res = await apiClient.post('/export/csv', null, { responseType: 'blob' });
  return res.data;
}

export async function exportJSON(): Promise<Blob> {
  const res = await apiClient.post('/export/json', null, { responseType: 'blob' });
  return res.data;
}
