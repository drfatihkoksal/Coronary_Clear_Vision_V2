import { apiClient } from '@/lib/api/client';

export interface ProjectionUploadResponse {
  session_id: string;
  projection_id: number;
  num_frames: number;
  image_width: number;
  image_height: number;
  angle_deg: number;
  secondary_angle_deg: number;
  pixel_spacing: number;
  sod: number;
  sid: number;
  frame_rate: number;
}

export interface SegmentProjectionResponse {
  projection_id: number;
  confidence: number;
  centerline: { x: number; y: number }[];
  diameters_mm: number[];
  num_points: number;
}

export interface CalibrateProjectionResponse {
  projection_id: number;
  pixel_spacing: number;
  catheter_size_fr: number;
  catheter_mm: number;
}

export interface SetTimiFramesResponse {
  projection_id: number;
  timi_start: number;
  timi_end: number;
  timi_frame_count: number;
}

export interface CalibrateFromSegmentationResponse {
  projection_id: number;
  pixel_spacing: number;
  catheter_size_fr: number;
  measured_diameter_px: number;
  quality_score: number;
}

export interface QFRPerViewResult {
  qfr: number;
  mld_mm: number;
  reference_diameter_mm: number;
}

export interface QFRResultData {
  qfr: number;
  qfr_combined?: number;
  qfr_per_view?: Record<string, QFRPerViewResult>;
  mode: string;
  pressure_profile: number[];
  diameters_mm: number[];
  distances_mm: number[];
  reference_diameter_mm: number;
  mld_mm: number;
  kt: number;
  flow_rate_ml_s: number;
  vessel_length_mm: number;
  disclaimer: string;
}

export interface ReconstructResponse {
  qfr: QFRResultData;
  mesh: {
    positions: number[];
    normals: number[];
    indices: number[];
    colors: number[];
    num_vertices: number;
    num_triangles: number;
  };
  angular_separation: number;
  reconstruction: {
    num_points: number;
    vessel_length_mm: number;
  };
}

export async function uploadProjection(projectionId: number, file: File): Promise<ProjectionUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<ProjectionUploadResponse>(
    `/qfr/projection/upload?projection_id=${projectionId}`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return response.data;
}

export async function segmentProjection(
  projectionId: number,
  frameIndex: number,
  engine = 'nnunet',
  roi?: [number, number, number, number],
  seedPoints?: [number, number][],
): Promise<SegmentProjectionResponse> {
  const response = await apiClient.post<SegmentProjectionResponse>('/qfr/projection/segment', {
    projection_id: projectionId,
    frame_index: frameIndex,
    engine,
    roi: roi ?? null,
    seed_points: seedPoints ?? null,
  });
  return response.data;
}

export async function calibrateProjection(
  projectionId: number,
  catheterSizeFr: number,
  catheterDiameterPx: number,
): Promise<CalibrateProjectionResponse> {
  const response = await apiClient.post<CalibrateProjectionResponse>('/qfr/projection/calibrate', {
    projection_id: projectionId,
    catheter_size_fr: catheterSizeFr,
    catheter_diameter_px: catheterDiameterPx,
  });
  return response.data;
}

export async function reconstructQFR(
  mode = 'fQFR',
  kt = 1.52,
  timiStartP1?: number | null,
  timiEndP1?: number | null,
  timiStartP2?: number | null,
  timiEndP2?: number | null,
): Promise<ReconstructResponse> {
  const response = await apiClient.post<ReconstructResponse>('/qfr/reconstruct', {
    mode,
    kt,
    timi_start_p1: timiStartP1 ?? null,
    timi_end_p1: timiEndP1 ?? null,
    timi_start_p2: timiStartP2 ?? null,
    timi_end_p2: timiEndP2 ?? null,
  });
  return response.data;
}

export async function recalculateQFR(
  stenosisIndices: number[],
  mode: string,
  kt: number,
): Promise<ReconstructResponse> {
  const response = await apiClient.post<ReconstructResponse>('/qfr/recalculate', {
    stenosis_indices: stenosisIndices,
    mode,
    kt,
  });
  return response.data;
}

export async function getQFRResult(): Promise<ReconstructResponse | { qfr: null; message: string }> {
  const response = await apiClient.get('/qfr/result');
  return response.data;
}

export async function fetchProjectionFrame(projectionId: number, frameIndex: number): Promise<ImageBitmap> {
  const response = await apiClient.get(`/qfr/projection/frame/${frameIndex}?projection_id=${projectionId}`, {
    responseType: 'arraybuffer',
  });
  const blob = new Blob([response.data], { type: 'image/png' });
  return createImageBitmap(blob);
}

export async function setTimiFrames(
  projectionId: number,
  timiStart: number,
  timiEnd: number,
): Promise<SetTimiFramesResponse> {
  const response = await apiClient.post<SetTimiFramesResponse>('/qfr/projection/set-timi-frames', {
    projection_id: projectionId,
    timi_start: timiStart,
    timi_end: timiEnd,
  });
  return response.data;
}

export async function calibrateFromSegmentation(
  projectionId: number,
  catheterSizeFr: number,
): Promise<CalibrateFromSegmentationResponse> {
  const response = await apiClient.post<CalibrateFromSegmentationResponse>(
    '/qfr/projection/calibrate-from-segmentation',
    {
      projection_id: projectionId,
      catheter_size_fr: catheterSizeFr,
    },
  );
  return response.data;
}

export async function fetchProjectionMask(projectionId: number, frameIndex: number): Promise<ImageBitmap> {
  const response = await apiClient.get(`/qfr/projection/mask/${frameIndex}?projection_id=${projectionId}`, {
    responseType: 'arraybuffer',
  });
  const blob = new Blob([response.data], { type: 'image/png' });
  return createImageBitmap(blob);
}
