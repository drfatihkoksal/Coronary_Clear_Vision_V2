import { apiClient } from '@/lib/api/client';
import type { QCAMetrics } from '@/types';

export interface QCARequest {
  frame_index: number;
  method?: 'gaussian' | 'parabolic' | 'threshold';
  num_points?: number;
}

interface QCAResponse {
  centerline: { x: number; y: number }[];
  diameter_profile_mm: number[];
  diameter_profile_px: number[];
  distances_mm: number[];
  mld_mm: number;
  mld_px: number;
  mld_index: number;
  diameter_stenosis_pct: number;
  proximal_ref_mm: number;
  distal_ref_mm: number;
  proximal_ref_index: number;
  distal_ref_index: number;
  interpolated_ref_mm: number;
  lesion_length_mm: number | null;
  pixel_spacing_mm: number;
  num_points: number;
  method: string;
  vessel_length_mm: number;
}

function toQCAMetrics(r: QCAResponse, frameIndex: number): QCAMetrics {
  return {
    frameIndex,
    centerline: r.centerline,
    diameterProfileMm: r.diameter_profile_mm,
    diameterProfilePx: r.diameter_profile_px,
    distancesMm: r.distances_mm,
    mldMm: r.mld_mm,
    mldPx: r.mld_px,
    mldIndex: r.mld_index,
    diameterStenosisPct: r.diameter_stenosis_pct,
    proximalRefMm: r.proximal_ref_mm,
    distalRefMm: r.distal_ref_mm,
    proximalRefIndex: r.proximal_ref_index,
    distalRefIndex: r.distal_ref_index,
    interpolatedRefMm: r.interpolated_ref_mm,
    lesionLengthMm: r.lesion_length_mm,
    vesselLengthMm: r.vessel_length_mm,
    pixelSpacingMm: r.pixel_spacing_mm,
    numPoints: r.num_points,
    method: r.method as QCAMetrics['method'],
  };
}

export async function calculateQCA(
  frameIndex: number,
  method: string = 'gaussian',
): Promise<QCAMetrics> {
  const res = await apiClient.post<QCAResponse>('/qca/calculate', {
    frame_index: frameIndex,
    method,
  });
  return toQCAMetrics(res.data, frameIndex);
}

export async function getQCAMeasurements(frameIndex: number): Promise<QCAMetrics> {
  const res = await apiClient.get<QCAResponse>(`/qca/measurements/${frameIndex}`);
  return toQCAMetrics(res.data, frameIndex);
}
