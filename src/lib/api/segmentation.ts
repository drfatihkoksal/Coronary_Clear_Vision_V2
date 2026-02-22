import { apiClient } from '@/lib/api/client';
import type { SegmentationEngine, Point } from '@/types';

export interface SegmentRequest {
  frame_index: number;
  engine: SegmentationEngine;
  roi?: [number, number, number, number]; // [x, y, w, h]
  seed_points?: [number, number][]; // [[x, y], ...]
}

export interface SegmentResponse {
  frame_index: number;
  engine: string;
  confidence: number;
  inference_time_ms: number;
  centerline?: Point[];
  diameters_px?: number[];
  has_centerline?: boolean;
}

export interface EnginesResponse {
  engines: Record<string, { available: boolean }>;
}

export async function segmentFrame(request: SegmentRequest): Promise<SegmentResponse> {
  const response = await apiClient.post<SegmentResponse>('/segmentation/segment', request);
  return response.data;
}

export async function segmentAndExtract(request: SegmentRequest): Promise<SegmentResponse> {
  const response = await apiClient.post<SegmentResponse>('/segmentation/segment-and-extract', request);
  return response.data;
}

export async function fetchMask(frameIndex: number): Promise<ImageBitmap> {
  const response = await apiClient.get(`/segmentation/mask/${frameIndex}`, {
    responseType: 'arraybuffer',
  });
  const blob = new Blob([response.data], { type: 'image/png' });
  return createImageBitmap(blob);
}

export async function getEngines(): Promise<EnginesResponse> {
  const response = await apiClient.get<EnginesResponse>('/segmentation/engines');
  return response.data;
}
