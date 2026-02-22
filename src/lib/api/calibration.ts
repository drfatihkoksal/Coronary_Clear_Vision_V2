import { apiClient } from '@/lib/api/client';
import type { PixelSpacing } from '@/types';

export interface CatheterCalibrationRequest {
  catheter_diameter_px: number;
  catheter_size_fr: number;
}

export interface ManualCalibrationRequest {
  known_distance_mm: number;
  measured_distance_px: number;
}

export interface CalibrationResponse {
  pixel_spacing_mm: number;
  source: string;
}

export async function calibrateCatheter(
  catheterDiameterPx: number,
  catheterSizeFr: number,
): Promise<CalibrationResponse> {
  const res = await apiClient.post<CalibrationResponse>('/calibration/catheter', {
    catheter_diameter_px: catheterDiameterPx,
    catheter_size_fr: catheterSizeFr,
  });
  return res.data;
}

export async function calibrateCatheterFromSegmentation(
  frameIndex: number,
  catheterSizeFr: number,
): Promise<CalibrationResponse> {
  const res = await apiClient.post<CalibrationResponse>('/calibration/catheter-from-segmentation', {
    frame_index: frameIndex,
    catheter_size_fr: catheterSizeFr,
  });
  return res.data;
}

export async function calibrateManual(
  knownDistanceMm: number,
  measuredDistancePx: number,
): Promise<CalibrationResponse> {
  const res = await apiClient.post<CalibrationResponse>('/calibration/manual', {
    known_distance_mm: knownDistanceMm,
    measured_distance_px: measuredDistancePx,
  });
  return res.data;
}

export async function getCurrentCalibration(): Promise<PixelSpacing> {
  const res = await apiClient.get<{
    row_spacing: number;
    col_spacing: number;
    source: string;
    confidence: number;
  }>('/calibration/current');
  return {
    rowSpacing: res.data.row_spacing,
    colSpacing: res.data.col_spacing,
    source: res.data.source as PixelSpacing['source'],
    confidence: res.data.confidence,
  };
}
