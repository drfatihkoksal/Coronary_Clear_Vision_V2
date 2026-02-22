import { apiClient, getSessionId } from '@/lib/api/client';
import type { StudyMetadata } from '@/types';

export interface UploadResponse {
  session_id: string;
  patient: StudyMetadata['patient'];
  study_info: StudyMetadata['studyInfo'];
  num_frames: number;
  frame_rate: number;
  image_width: number;
  image_height: number;
  pixel_spacing: { row_spacing: number; col_spacing: number; source: string; confidence: number } | null;
}

export async function uploadDicom(file: File, anonymize: boolean = true): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<UploadResponse>(
    `/dicom/upload?anonymize=${anonymize}`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return response.data;
}

export async function fetchFrame(frameIndex: number): Promise<ImageBitmap> {
  const response = await apiClient.get(`/dicom/frame/${frameIndex}`, {
    responseType: 'arraybuffer',
    params: { sid: getSessionId() ?? '' },
  });
  const blob = new Blob([response.data], { type: 'image/png' });
  return createImageBitmap(blob);
}

export async function fetchMetadata(): Promise<UploadResponse> {
  const response = await apiClient.get<UploadResponse>('/dicom/metadata');
  return response.data;
}

export async function clearStudy(): Promise<void> {
  await apiClient.post('/dicom/clear');
}
