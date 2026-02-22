import { apiClient } from '@/lib/api/client';

export async function calculateMotion() {
  const res = await apiClient.post('/motion/calculate');
  return res.data;
}

export async function fetchMotionSignal() {
  const res = await apiClient.get('/motion/signal');
  return res.data;
}
