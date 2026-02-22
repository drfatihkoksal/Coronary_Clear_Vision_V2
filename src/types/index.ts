// Study types
export interface PatientInfo {
  patientId: string | null;
  name: string | null;
  birthDate: string | null;
  sex: 'M' | 'F' | 'O' | null;
  age: number | null;
}

export interface StudyInfo {
  studyInstanceUid: string | null;
  seriesInstanceUid: string | null;
  studyDate: string | null;
  studyTime: string | null;
  description: string | null;
  institution: string | null;
  modality: string;
}

export interface PixelSpacing {
  rowSpacing: number;
  colSpacing: number;
  source: 'dicom' | 'catheter' | 'manual' | 'from_mask';
  confidence: number;
}

export interface StudyMetadata {
  sessionId: string;
  patient: PatientInfo;
  studyInfo: StudyInfo;
  numFrames: number;
  frameRate: number;
  imageWidth: number;
  imageHeight: number;
  pixelSpacing: PixelSpacing | null;
}

export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// QCA types
export interface QCAMetrics {
  frameIndex: number;
  centerline: Point[];
  diameterProfileMm: number[];
  diameterProfilePx: number[];
  distancesMm: number[];
  mldMm: number;
  mldPx: number;
  mldIndex?: number;
  diameterStenosisPct: number;
  proximalRefMm?: number;
  distalRefMm?: number;
  proximalRefIndex?: number;
  distalRefIndex?: number;
  interpolatedRefMm?: number;
  lesionLengthMm: number | null;
  vesselLengthMm?: number;
  pixelSpacingMm: number;
  numPoints: number;
  method: 'gaussian' | 'parabolic' | 'threshold';
}

// RWS types
export type RWSInterpretation = 'normal' | 'intermediate' | 'vulnerable' | 'high_risk';

export interface RWSResult {
  beatNumber: number;
  startFrame: number;
  endFrame: number;
  mldRwsPct: number;
  proximalRwsPct: number;
  distalRwsPct: number;
  averageRwsPct: number;
  interpretation: RWSInterpretation;
  outlierMethod: 'none' | 'hampel' | 'double_hampel';
  vessel: string | null;
}

// Segmentation
export type SegmentationEngine = 'nnunet' | 'nnunet_wide' | 'nnunet_fullframe' | 'angiopy' | 'roi_angiopy' | 'seedmodel';

// ECG types
export interface ECGData {
  signal: number[];
  sampleRate: number;
  numSamples: number;
  rPeaks: number[];
}

// Settings
export type ThemeMode = 'light' | 'dark' | 'system';
