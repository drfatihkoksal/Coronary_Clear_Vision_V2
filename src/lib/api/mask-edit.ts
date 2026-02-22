import { apiClient } from '@/lib/api/client';

export interface BrushRequest {
  frame_index: number;
  points: [number, number][];
  radius?: number;
  is_erasing?: boolean;
}

export interface SmartBrushRequest {
  frame_index: number;
  points: [number, number][];
  radius?: number;
  tolerance?: number;
  is_erasing?: boolean;
}

export interface FloodFillRequest {
  frame_index: number;
  seed_point: [number, number];
  tolerance?: number;
}

export interface MorphologicalRequest {
  frame_index: number;
  operation: 'dilate' | 'erode' | 'open' | 'close';
  kernel_size?: number;
  iterations?: number;
}

async function postMaskEdit(endpoint: string, data: unknown): Promise<ImageBitmap> {
  const response = await apiClient.post(endpoint, data, {
    responseType: 'arraybuffer',
  });
  const blob = new Blob([response.data], { type: 'image/png' });
  return createImageBitmap(blob);
}

export async function applyBrush(request: BrushRequest): Promise<ImageBitmap> {
  return postMaskEdit('/mask-edit/brush', request);
}

export async function applySmartBrush(request: SmartBrushRequest): Promise<ImageBitmap> {
  return postMaskEdit('/mask-edit/smart-brush', request);
}

export async function applyFloodFill(request: FloodFillRequest): Promise<ImageBitmap> {
  return postMaskEdit('/mask-edit/flood-fill', request);
}

export async function applyMorphological(request: MorphologicalRequest): Promise<ImageBitmap> {
  return postMaskEdit('/mask-edit/morphological', request);
}
