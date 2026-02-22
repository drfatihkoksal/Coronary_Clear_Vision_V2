import { apiClient } from '@/lib/api/client';

export interface TrackingInitResponse {
  frame_index: number;
  roi: [number, number, number, number];
  status: string;
}

export interface TrackingFrameResult {
  frame_index: number;
  bbox: [number, number, number, number];
  confidence: number;
  success: boolean;
}

export interface TrackingPropagateResponse {
  direction: 'forward' | 'backward';
  tracked_frames: number;
  total_frames: number;
  results: TrackingFrameResult[];
}

export interface TrackingStateResponse {
  status: string;
  start_frame?: number | null;
  roi?: [number, number, number, number] | null;
  results: TrackingFrameResult[];
  num_tracked?: number;
}

export async function initializeTracking(
  frameIndex: number,
  roi: [number, number, number, number],
): Promise<TrackingInitResponse> {
  const response = await apiClient.post<TrackingInitResponse>('/tracking/initialize', {
    frame_index: frameIndex,
    roi,
  });
  return response.data;
}

export async function propagateTracking(
  direction: 'forward' | 'backward',
  maxFrames?: number,
  autoSegment?: boolean,
): Promise<TrackingPropagateResponse> {
  const response = await apiClient.post<TrackingPropagateResponse>('/tracking/propagate', {
    direction,
    max_frames: maxFrames,
    auto_segment: autoSegment,
  });
  return response.data;
}

export async function getTrackingState(): Promise<TrackingStateResponse> {
  const response = await apiClient.get<TrackingStateResponse>('/tracking/state');
  return response.data;
}

export async function clearTracking(): Promise<void> {
  await apiClient.post('/tracking/clear');
}
