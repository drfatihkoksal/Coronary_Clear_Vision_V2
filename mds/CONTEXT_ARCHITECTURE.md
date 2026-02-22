# Coronary RWS Analyser v1.1 — Architecture Deep-Dive

> This document provides a detailed architectural analysis of every layer, data flow, dependency, state machine, error path, caching strategy, concurrency model, security posture, API design, module boundary, scalability constraint, testing approach, and deployment pipeline in the Coronary RWS Analyser application. It is intended as the architectural blueprint for a complete rewrite.

---

## Table of Contents

1. [Data Flow Diagrams](#1-data-flow-diagrams)
2. [Dependency Analysis](#2-dependency-analysis)
3. [State Machine Definitions](#3-state-machine-definitions)
4. [Error Propagation](#4-error-propagation)
5. [Caching Strategy](#5-caching-strategy)
6. [Concurrency Model](#6-concurrency-model)
7. [Security Architecture](#7-security-architecture)
8. [API Design Analysis](#8-api-design-analysis)
9. [Module Boundaries](#9-module-boundaries)
10. [Scalability Analysis](#10-scalability-analysis)
11. [Testing Architecture](#11-testing-architecture)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Recommendations for Rewrite](#13-recommendations-for-rewrite)

---

## 1. Data Flow Diagrams

### 1.1 Request Lifecycle (General)

```
Frontend Action (e.g. button click)
  │
  ▼
Zustand Store Action (e.g. segmentationStore.segmentFrame)
  │
  ▼
api.ts Function (e.g. segmentationApi.segment)
  │  ─ Adds headers: Authorization (JWT), X-Session-ID
  │  ─ Serializes body (JSON or FormData)
  │
  ▼
HTTP Request (fetch/axios)
  │  ─ http://127.0.0.1:8000/segmentation/segment
  │
  ▼
FastAPI Middleware Chain
  │  1. LoggingMiddleware (logs method, path, client, timing)
  │  2. CORSMiddleware (validates origin against cors_origins_list)
  │  3. Exception Handlers (DomainException, ValidationError, RequestValidationError, catch-all)
  │
  ▼
Route Handler (presentation/routes/segmentation_routes.py)
  │  ─ Pydantic request validation
  │  ─ Dependency injection (Depends(...))
  │  ─ Access module-level state (_current_study via _get_study())
  │
  ▼
Use Case (application/use_cases/) [sometimes bypassed]
  │  ─ Orchestrates domain services
  │  ─ Coordinates infrastructure calls
  │
  ▼
Domain Service (domain/services/)
  │  ─ Pure computation (e.g. QCAEngine, RWSCalculator)
  │  ─ No I/O, no external dependencies
  │
  ▼
Infrastructure (infrastructure/ml_engines/, file_handlers/)
  │  ─ ML model inference (nnU-Net, AngioPy)
  │  ─ DICOM I/O (pydicom)
  │  ─ Image processing (OpenCV, Pillow)
  │
  ▼
Response (JSON)
  │  ─ Images: base64-encoded PNG strings
  │  ─ Masks: "data:image/png;base64,..." (segmentation routes)
  │  ─ Masks: raw base64 without prefix (QFR routes)
  │
  ▼
Zustand Store Update
  │  ─ Stores result in Map<number, Data> (per-frame)
  │  ─ Triggers React re-render
  │
  ▼
Canvas Layer Re-render
  ─ Only dirty layers repaint
```

### 1.2 DICOM Loading Pipeline

```
User: Drag-drop or file-open dialog
  │
  ▼
dicomStore.loadFile(file, anonymize)
  │  ─ Calls resetAllAnalysis() first (clears all 17 stores)
  │
  ▼
dicomApi.load(file, anonymize) → POST /dicom/load
  │  ─ FormData with file blob
  │
  ▼
Backend: load_dicom() route handler
  │  ─ Writes to NamedTemporaryFile(.dcm)
  │  ─ Creates LoadStudyRequest
  │
  ▼
LoadStudyUseCase.execute()
  │
  ├─▶ DicomHandler.load_study(file_path)
  │     │  ─ pydicom.dcmread(file_path, force=True)
  │     │  ─ Extracts all frames: ds.pixel_array → per-frame numpy arrays
  │     │  ─ Extracts metadata: PatientInfo, StudyInfo
  │     │  ─ Extracts pixel spacing (PixelSpacing / ImagerPixelSpacing)
  │     │  ─ Extracts DICOM metadata dict:
  │     │      positioner_primary_angle, positioner_secondary_angle,
  │     │      distance_source_to_detector (SID),
  │     │      distance_source_to_patient (SOD),
  │     │      raw_pixel_spacing, raw_imager_pixel_spacing,
  │     │      intensifier_size, frame_time, manufacturer
  │     │  ─ Constructs Study entity (aggregate root)
  │     │  ─ Study.frames = [Frame(index=i, pixel_data=arr, ...)]
  │     ▼
  │   Study entity (in-memory, held by route module state)
  │
  ├─▶ ECG Extraction (if extract_ecg=True)
  │     │  ─ DicomHandler.load_dicom() → pydicom Dataset
  │     │  ─ DicomHandler.extract_ecg_data(ds)
  │     │     └─ Reads WaveformSequence (DICOM tag 0008,104C)
  │     │  ─ ECGParser.parse_from_dicom_waveform()
  │     │     └─ SiemensECGFilter (50/60 Hz notch if vendor=Siemens)
  │     │     └─ Pan-Tompkins R-peak detection
  │     │     └─ Constructs ECGSignal entity with r_peaks, sample_rate
  │     │  ─ Fallback: generate_synthetic_ecg() from frame intensities
  │     ▼
  │   study.ecg_signal = ECGSignal(...)
  │
  └─▶ _store_current(session_id, study, file_path)
        ─ Module-level: _current_study = study
        ─ Session dict: create_session(session_id, {"study": study, ...})
        ─ Returns _build_metadata_dict(study, response)
          → JSON with all DICOM metadata for frontend

Frontend receives DicomMetadata:
  ─ dicomStore.metadata = parsed response
  ─ playerStore.setTotalFrames(metadata.numFrames)
  ─ playerStore.setFrameRate(metadata.frameRate)
  ─ playerStore.setCurrentFrame(0)
```

**Frame-by-frame loading** (after initial DICOM load):
```
Canvas needs frame N
  │
  ▼
dicomStore.getFrame(N)
  │  ─ Check frames Map cache → return if hit
  │
  ▼
dicomApi.getFrame(N) → GET /dicom/frame/{N}
  │
  ▼
Backend: get_single_frame(N)
  │  ─ _get_current_study().get_frame(N)
  │  ─ Image.fromarray(frame.pixel_data)
  │  ─ PNG encode → base64
  │
  ▼
Frontend: FrameData { data: base64, width, height }
  │  ─ Stores in dicomStore.frames Map
  │  ─ Canvas renders by creating Image from base64
```

### 1.3 Segmentation Pipeline

```
User clicks "Segment" or "Segment + Extract"
  │
  ▼
segmentationStore.segmentAndExtract(frameIndex, roi)
  │
  ▼
segmentationApi.segmentAndExtract(frameIndex, engine, roi)
  │  → POST /segmentation/segment-and-extract?frame_index=N&engine=nnunet&x=...&y=...&w=...&h=...
  │
  ▼
Backend: segment_and_extract()
  │
  ├─▶ _get_frame(frame_index)
  │     └─ _get_study() → dicom_routes._current_study
  │     └─ study.get_frame(frame_index).pixel_data → np.ndarray
  │
  ├─▶ Segmentation Engine Selection
  │     ├─ "nnunet": create_nnunet_engine() → NNUNetEngine
  │     │    └─ engine.predict(frame, roi=roi_bbox)
  │     │    └─ Dual-channel: [grayscale, gaussian_spatial_attention]
  │     │    └─ nnU-Net inference → SegmentationResult(mask=np.ndarray)
  │     │    └─ CenterComponentKeeper post-processing
  │     ├─ "nnunet-wide": create_nnunet_wide() (192x192 crop)
  │     ├─ "nnunet-fullframe": create_nnunet_fullframe() (512x512)
  │     ├─ "angiopy": AngioPy with seed points
  │     │    └─ Seed points → distance map → extra input channel
  │     │    └─ U-Net + InceptionResNetV2 backbone
  │     └─ "roi+angiopy": Hybrid pipeline
  │          └─ Step 1: nnunet-fullframe → coarse mask
  │          └─ Step 2: CenterlineExtractor → seeds from mask
  │          └─ Step 3: AngioPy refinement with extracted seeds
  │
  ├─▶ ROI Masking (if roi provided)
  │     └─ Zero out mask outside ROI bounding box
  │
  ├─▶ Centerline Extraction
  │     └─ CenterlineExtractor.extract(mask, method="mcp_auto")
  │     └─ Methods: skeleton, distance_transform, mcp, mcp_auto
  │     └─ Returns: CenterlineResult { points: Nx2 (y,x) }
  │
  └─▶ Response Construction
        └─ encode_mask(mask) → "data:image/png;base64,..." (WITH prefix)
        └─ centerline_points → [{x, y}, ...]
        └─ Return JSON with mask, centerline, seed_points
```

### 1.4 QFR Pipeline (Dual Projection → 3D)

```
                    ┌──────────────────────────────┐
                    │    QFR Mode Activation        │
                    │  qfrModeStore.setActive(true) │
                    └──────────┬───────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │                                           │
    ┌────▼─────┐                              ┌─────▼────┐
    │ Proj 1   │                              │ Proj 2   │
    │ Upload   │                              │ Upload   │
    └────┬─────┘                              └────┬─────┘
         │                                         │
    POST /qfr/projection/upload             POST /qfr/projection/upload
    (projection_id=1, file=DICOM)           (projection_id=2, file=DICOM)
         │                                         │
    ┌────▼─────────────────┐              ┌────────▼────────────┐
    │ Parse DICOM          │              │ Parse DICOM          │
    │ Extract angles       │              │ Extract angles       │
    │ (metadata dict)      │              │ (metadata dict)      │
    │ Extract SID/SOD      │              │ Extract SID/SOD      │
    │ Store in             │              │ Store in             │
    │ qfr_session.proj1    │              │ qfr_session.proj2    │
    └────┬─────────────────┘              └────────┬────────────┘
         │                                         │
    ┌────▼──────────────┐                 ┌────────▼───────────┐
    │ Calibrate (6Fr)   │                 │ Calibrate (6Fr)    │
    │ POST /qfr/        │                 │ POST /qfr/         │
    │ projection/       │                 │ projection/        │
    │ calibrate         │                 │ calibrate          │
    │ ps = 2.0mm /      │                 │ ps = 2.0mm /       │
    │   measured_px     │                 │   measured_px      │
    └────┬──────────────┘                 └────────┬───────────┘
         │                                         │
    ┌────▼──────────────┐                 ┌────────▼───────────┐
    │ Segment frame     │                 │ Segment frame      │
    │ POST /qfr/        │                 │ POST /qfr/         │
    │ projection/segment│                 │ projection/segment │
    │ → mask, centerline│                 │ → mask, centerline │
    │   diameter_profile│                 │   diameter_profile │
    └────┬──────────────┘                 └────────┬───────────┘
         │                                         │
         └────────────┬────────────────────────────┘
                      │
                ┌─────▼──────────────────────────────┐
                │ POST /qfr/reconstruct               │
                │                                     │
                │ 1. Build ProjectionParams (x2)      │
                │    ─ primary_angle, secondary_angle  │
                │    ─ SID, SOD                        │
                │    ─ pixel_spacing (calibration or   │
                │      DICOM * SOD/SID magnification)  │
                │                                     │
                │ 2. Validate angle difference ≥25°   │
                │                                     │
                │ 3. StereoReconstructor               │
                │    .reconstruct_centerline()         │
                │    ─ Build camera matrices (K,R,t,P) │
                │    ─ fx = SOD / isocenter_ps         │
                │    ─ Epipolar matching               │
                │    ─ DLT triangulation               │
                │    → ReconstructionResult:            │
                │      centerline_3d (Nx3)             │
                │      matched_pts1, matched_pts2      │
                │                                     │
                │ 4. Diameter measurement at matched   │
                │    2D points (epipolar-aligned)      │
                │    ─ QCAEngine.measure_diameters_    │
                │      at_points() for each view       │
                │    ─ Geometric mean: √(d1 × d2)     │
                │                                     │
                │ 5. TIMI Frame Count                  │
                │    ─ From request T-start/T-end      │
                │    ─ Or from session projections      │
                │    ─ Average across available values  │
                │                                     │
                │ 6. Frame rate priority:               │
                │    request > DICOM P1 > DICOM P2     │
                │    > default 15.0 fps                │
                │                                     │
                │ 7. QFR3DCalculator.calculate()       │
                │    ─ Gould/Young-Tsai pressure model │
                │    ─ Viscous: ΔP = 128μLQ / πd⁴     │
                │    ─ Turbulent: ΔP = Kt×ρ/2×(A₀/A-1)²│
                │    ─ QFR = (Pa - ΣΔP) / Pa           │
                │                                     │
                │ 8. VesselMesher.generate_mesh()      │
                │    ─ 12-sided tube at each point     │
                │    ─ QFR heatmap coloring             │
                │                                     │
                │ Response: mesh, qfr_value,           │
                │   pressure_drop, velocity, etc.      │
                └─────────────────────────────────────┘
```

### 1.5 State Synchronization: Backend ↔ Frontend

```
Backend State (module-level)                Frontend State (Zustand stores)
─────────────────────────                  ──────────────────────────────
dicom_routes._current_study  ◄─────────►  dicomStore.metadata + frames Map
dicom_routes._ecg_r_peaks   ◄─────────►  ecgStore.rPeaks
calibration_routes           ◄─────────►  calibrationStore (localStorage)
  ._calibrated_pixel_spacing
motion_routes._engine        ◄─────────►  motionStore.signal + peaks
  .motion_signal/.motion_peaks
rws_routes._rws_results      ◄─────────►  rwsStore.results
qfr_session._qfr_sessions   ◄─────────►  qfrModeStore.projection1/2

Synchronization: Request/Response only (no WebSocket, no SSE)
─ Frontend is the "driver" — all updates are initiated by user actions
─ Backend never pushes state changes
─ No conflict resolution needed (single-user)
─ Session ID mismatch: frontend generates its own, backend generates its own
  (the X-Session-ID header is mostly used for QFR session routing)
```

---

## 2. Dependency Analysis

### 2.1 Backend Import Graph

```
main.py
  └─▶ presentation/routes/* (all 16 route modules)
        ├─▶ presentation/schemas/* (Pydantic request/response models)
        ├─▶ infrastructure/config/dependencies.py (DI container)
        │     ├─▶ domain/services/* (factory functions)
        │     ├─▶ infrastructure/ml_engines/* (lazy loaded)
        │     └─▶ infrastructure/file_handlers/* (lazy loaded)
        ├─▶ domain/entities/* (Study, Frame, Vessel, etc.)
        ├─▶ domain/exceptions/* (DomainException hierarchy)
        └─▶ domain/value_objects/* (PixelSpacing, BeatRange, etc.)

presentation/routes/dicom_routes.py
  └─▶ application/use_cases/dicom/load_study.py
        ├─▶ infrastructure/file_handlers/dicom_handler.py
        │     └─▶ pydicom, Pillow, numpy
        └─▶ infrastructure/file_handlers/ecg_parser.py
              └─▶ domain/services/ecg_parser.py (Pan-Tompkins)

presentation/routes/segmentation_routes.py
  ├─▶ infrastructure/ml_engines/nnunet_engine.py (lazy)
  │     └─▶ torch, nnunetv2
  ├─▶ infrastructure/ml_engines/angiopy_engine.py (lazy)
  │     └─▶ segmentation_models_pytorch, timm
  └─▶ domain/services/centerline_extractor.py
        └─▶ scipy, scikit-image

presentation/routes/qfr_routes.py
  ├─▶ domain/qfr_session.py (module-level session store)
  ├─▶ domain/services/stereo_reconstructor.py
  │     └─▶ numpy (linear algebra)
  ├─▶ domain/services/qfr_calculator_3d.py
  │     └─▶ numpy
  ├─▶ domain/services/vessel_mesher.py
  │     └─▶ numpy
  └─▶ domain/services/qca_engine.py
        └─▶ numpy, scipy

presentation/routes/motion_routes.py
  └─▶ domain/services/motion_signal_engine.py
        └─▶ opencv (cv2.calcOpticalFlowFarneback)
```

### 2.2 Frontend Store Dependencies

```
dicomStore ──▶ playerStore (setTotalFrames, setFrameRate, reset)
             ──▶ api.ts (dicomApi)
             ──▶ sessionUtils (resetAllAnalysis)

segmentationStore ──▶ settingsStore (default engine)
                   ──▶ api.ts (segmentationApi)

qcaStore ──▶ api.ts (qcaApi)

ecgStore ──▶ api.ts (ecgApi)

rwsStore ──▶ api.ts (rwsApi)

qfrModeStore ──▶ api.ts (qfrApi)

motionStore ──▶ api.ts (motionApi)

trackingStore ──▶ api.ts (trackingApi)

playerStore ──▶ settingsStore (speed, looping defaults)

calibrationStore ──▶ localStorage (persist)

settingsStore ──▶ localStorage (persist)

annotationStore ──▶ (independent)

overlayStore ──▶ (independent)

maskEditStore ──▶ api.ts (maskEditApi)

exportStore ──▶ api.ts (exportApi)

reportStore ──▶ api.ts (reportApi)
```

Key coupling points:
- `dicomStore.loadFile` calls `resetAllAnalysis()` which resets ALL other stores
- `dicomStore` directly calls `playerStore.getState().setTotalFrames()` (cross-store)
- All domain stores depend on `api.ts` (2600-line monolith)
- No circular store dependencies detected

### 2.3 Circular Dependency Risks

**Backend**: No circular imports detected. Route modules reference `dicom_routes._current_study` but via forward import (`from app.presentation.routes.dicom_routes import _get_current_study`), which works at call time.

**Frontend**: The `resetAllAnalysis()` utility in `sessionUtils.ts` imports from all stores, creating a fan-out dependency. Not circular, but fragile — adding a new store requires updating this utility.

### 2.4 External Dependency Audit

| Dependency | Critical? | Replaceable? | Notes |
|---|---|---|---|
| **pydicom** | YES | No | Only Python DICOM library with full support |
| **numpy** | YES | No | Foundation for all computation |
| **scipy** | YES | Partially | Used for signal processing, interpolation, optimization |
| **opencv** | Moderate | Yes | Used for optical flow, tracking. Could use scikit-image |
| **Pillow** | YES | No | Image encoding (numpy→PNG→base64) |
| **torch** | Moderate | No (for ML) | Required for nnU-Net and AngioPy |
| **nnunetv2** | Moderate | Yes | Could replace with custom inference loop |
| **segmentation-models-pytorch** | Low | Yes | AngioPy backbone, could use plain PyTorch |
| **FastAPI** | YES | Yes (Flask, Litestar) | Core framework, but mostly standard REST |
| **Pydantic v2** | YES | Partially | Validation, could use msgspec |
| **React** | YES | No | UI framework |
| **Zustand** | Moderate | Yes (Jotai, Redux) | State management, lightweight |
| **Cornerstone.js** | High | Partially | DICOM decode in browser. Currently minimal usage |
| **Three.js** | Low | Yes | Only used for QFR mesh viewer |
| **Recharts** | Low | Yes | Charts for QCA/RWS/ECG |

---

## 3. State Machine Definitions

### 3.1 Application Lifecycle

```
            ┌───────────┐
            │   IDLE     │ ← App just loaded, no DICOM
            └─────┬─────┘
                  │ User loads DICOM
            ┌─────▼─────┐
            │  LOADING   │ ← Parsing DICOM, extracting frames
            └─────┬─────┘
                  │ Success/Failure
         ┌────────┴────────┐
   ┌──────▼──────┐   ┌─────▼─────┐
   │   LOADED    │   │   ERROR   │
   │  (browsing) │   │  (toast)  │
   └──────┬──────┘   └───────────┘
          │ User triggers analysis
   ┌──────▼──────┐
   │  ANALYZING  │ ← Segmentation, QCA, RWS, QFR
   └──────┬──────┘
          │ Results available
   ┌──────▼──────┐
   │  RESULTS    │ ← Charts, measurements, 3D mesh
   └─────────────┘

Transitions:
  IDLE → LOADING:     dicomStore.loadFile() or loadFromPath()
  LOADING → LOADED:   Backend returns DicomMetadata successfully
  LOADING → ERROR:    Backend returns error (400/422/500)
  LOADED → ANALYZING: User triggers segmentation/QCA/RWS/QFR
  ANALYZING → RESULTS: Computation completes
  ANY → IDLE:         dicomStore.reset() (clear session)
```

### 3.2 QFR Mode States

```
  ┌─────────────┐
  │  INACTIVE   │ ← QFR tab not active
  └──────┬──────┘
         │ User activates QFR mode
  ┌──────▼──────┐
  │   EMPTY     │ ← No projections loaded
  └──────┬──────┘
         │ Upload Projection 1
  ┌──────▼──────────┐
  │ P1_LOADED       │ ← One projection loaded
  └──────┬──────────┘
         │ Upload Projection 2
  ┌──────▼──────────┐
  │ BOTH_LOADED     │ ← Both projections loaded
  └──────┬──────────┘
         │ Calibrate each (6Fr catheter)
  ┌──────▼──────────┐
  │ CALIBRATED      │ ← Both calibrated
  └──────┬──────────┘
         │ Segment each (nnU-Net)
  ┌──────▼──────────┐
  │ SEGMENTED       │ ← Both have centerlines + diameters
  └──────┬──────────┘
         │ Click "Reconstruct" → POST /qfr/reconstruct
  ┌──────▼──────────┐
  │ RECONSTRUCTED   │ ← 3D mesh + QFR value available
  └─────────────────┘

State is tracked in:
  Backend: qfr_session.py (QFRModeSession dataclass)
  Frontend: qfrModeStore (Zustand)
```

### 3.3 Tracking States

```
  ┌───────────┐
  │   IDLE    │ ← No tracker initialized
  └─────┬─────┘
        │ User places ROI, clicks "Initialize"
  ┌─────▼────────────┐
  │  INITIALIZED     │ ← CSRT tracker initialized on current frame
  └─────┬────────────┘
        │ User clicks "Propagate" (forward/backward)
  ┌─────▼────────────┐
  │  TRACKING        │ ← Propagating through frames
  │  (per frame:     │
  │   success/fail   │
  │   + confidence)  │
  └─────┬────────────┘
        │ Propagation complete or confidence < threshold
  ┌─────▼────────────┐
  │  COMPLETED       │ ← Results available: Map<frame, ROI+confidence>
  └──────────────────┘
```

### 3.4 Segmentation States (Per Frame)

```
  ┌──────────────┐
  │ NOT_SEGMENTED│ ← No mask for this frame
  └──────┬───────┘
         │ segmentAndExtract()
  ┌──────▼───────┐
  │ SEGMENTING   │ ← API call in flight
  └──────┬───────┘
         │
    ┌────┴────┐
  ┌─▼──┐   ┌─▼─────┐
  │DONE│   │FAILED │
  │    │   │       │
  └─┬──┘   └───────┘
    │
    │ Has mask + centerline
  ┌─▼────────────────┐
  │ SEGMENTED        │
  │  mask: base64    │
  │  centerline: []  │
  │  seedPoints: []  │
  └──────────────────┘

Storage: segmentationStore.frameData Map<number, FrameSegmentationData>
```

### 3.5 Playback States

```
  ┌──────────┐   Space      ┌──────────┐
  │  PAUSED  │ ────────────▶│ PLAYING  │
  │          │◀─────────────│          │
  └──────────┘   Space      └──────────┘
       │                          │
       │ Stop                     │ End of range
       ▼                          ▼
  ┌──────────┐              ┌──────────┐
  │ STOPPED  │              │ PAUSED   │ (if !looping)
  │ frame=0  │              │          │ or
  └──────────┘              │ PLAYING  │ (if looping, wraps)
                            └──────────┘

State: playerStore.playbackState: 'playing' | 'paused' | 'stopped'
Driven by: requestAnimationFrame loop checking elapsed time vs frameRate
```

---

## 4. Error Propagation

### 4.1 Backend Error Flow

```
Domain Layer
  │
  ├─ DomainException(code, message)
  │    ├─ StudyNotFoundError
  │    ├─ FrameOutOfRangeError
  │    ├─ VesselNotFoundError
  │    ├─ SegmentationFailedError
  │    ├─ InsufficientMeasurementsError
  │    ├─ NotCalibratedError
  │    ├─ RWSCalculationError
  │    ├─ QFRCalculationError
  │    └─ QCACalculationError
  │
  ├─ ValidationError (field-level)
  │    ├─ RequiredFieldError
  │    ├─ OutOfRangeError
  │    ├─ InvalidFormatError
  │    └─ InvalidDiameterError
  │
  └─ MultipleValidationErrors (batch)
```

**Global Exception Handlers** (middleware/error_handler.py):

| Exception | HTTP Status | Response Format |
|---|---|---|
| `DomainException` | 422 | `{"error": {"code": "...", "message": "..."}}` |
| `ValidationError` | 400 | `{"error": {"code": "VALIDATION_ERROR", ...}}` |
| `MultipleValidationErrors` | 400 | `{"error": {"code": "...", "errors": [...]}}` |
| `RequestValidationError` (Pydantic) | 400 | `{"error": {"code": "VALIDATION_ERROR", "errors": [...]}}` |
| `Exception` (catch-all) | 500 | `{"error": {"code": "INTERNAL_ERROR", "message": "..."}}` |

**Route-level exception handling**: Most routes also have try/except blocks that catch specific domain exceptions and convert them to HTTPException with appropriate status codes. This creates a dual error handling layer:

1. Route-level try/except → HTTPException(status_code, detail)
2. Global handlers → catch anything that escapes routes

### 4.2 Frontend Error Handling

```
api.ts function
  │
  ├─ fetch() call with timeout
  │
  ├─ Response status check
  │    ├─ 401: Attempt token refresh, retry once
  │    ├─ 4xx: Parse error JSON, throw Error(message)
  │    └─ 5xx: throw Error("Server error")
  │
  ├─ JSON parse → return data
  │
  └─ catch → re-throw with descriptive message

Zustand store action
  │
  ├─ try { await apiCall() }
  │
  ├─ catch {
  │    set({ error: message, isLoading: false })
  │    throw error  // Re-throw for component
  │ }
  │
  └─ Component: toast notification or inline error display
```

**No React Error Boundaries** detected in the current codebase. Unhandled render errors will crash the component tree.

### 4.3 Recovery Strategies

| Error Type | Recovery |
|---|---|
| DICOM load failure | User retries with different file |
| Segmentation engine not available | Show "engine unavailable" in engine list |
| QFR angle < 25° | Warning shown but reconstruction proceeds |
| RWS < 2 frames | Error toast, user must extend frame range |
| Backend unreachable | Health check polling (not implemented) |
| Token expired | Auto-refresh via 401 handler in api.ts |

---

## 5. Caching Strategy

### 5.1 Frontend Caches

| Cache | Data Structure | Key | Invalidation |
|---|---|---|---|
| Frame cache | `Map<number, FrameData>` in dicomStore | Frame index | Clear on new DICOM load (`frames: new Map()`) |
| Segmentation cache | `Map<number, FrameSegmentationData>` in segmentationStore | Frame index | Clear on new DICOM load (via `resetAllAnalysis`) |
| QCA cache | `Map<number, QCAMetrics>` in qcaStore | Frame index | Clear on new DICOM load |
| QFR projection frames | `Map<number, FrameData>` in qfrModeStore (per projection) | Frame index | Clear on projection reset |
| Settings | `localStorage` via Zustand persist middleware | Fixed keys | User manually changes settings |
| Calibration | `localStorage` via Zustand persist middleware | Fixed keys | User recalibrates |

**Memory management**: No explicit eviction. Frame cache grows unbounded as user scrolls through frames. For a 150-frame study at 512x512, each base64 PNG is ~100-300KB, so total cache is ~15-45MB. For 1000+ frame studies this could grow to 100-300MB.

**Preloading**: `dicomStore.preloadFrames(start, end)` fetches a range of frames in parallel (`Promise.all`). Used during playback to preload adjacent frames. No LRU eviction.

### 5.2 Backend Caches

| Cache | Storage | Scope | Invalidation |
|---|---|---|---|
| Current study | `_current_study` module var in `dicom_routes.py` | Global (single-user) | Overwritten on next load, cleared by `/dicom/clear` |
| QFR sessions | `_qfr_sessions` dict in `qfr_session.py` | Keyed by session_id | Explicit delete/reset, or never (memory leak) |
| R-peaks (edited) | `_ecg_r_peaks` list in `dicom_routes.py` | Global | Reset on new DICOM load |
| Motion signal | `_engine` instance in `motion_routes.py` | Global singleton | `engine.reset()` |
| RWS results | `_rws_results` list in `rws_routes.py` | Global | `/rws/clear` endpoint |
| Calibration | `_calibrated_pixel_spacing` in `calibration_routes.py` | Global | Overwritten on recalibrate |
| Domain services | `@lru_cache()` singletons in `dependencies.py` | Process lifetime | `cleanup_dependencies()` |
| ML models | Instance vars in engine classes | Process lifetime | Engine destruction |
| Session storage | `_session_storage` dict in `dependencies.py` | Process lifetime | `delete_session()` |

**No automatic cleanup**: QFR sessions and general sessions accumulate indefinitely. There is no TTL, no max-session limit enforcement, and no garbage collection. The `session_timeout_minutes` setting exists but is never enforced.

### 5.3 Cache Invalidation Triggers

```
New DICOM Load:
  Frontend: resetAllAnalysis() → clears all 17 stores
  Backend: _store_current() → overwrites _current_study, resets _ecg_r_peaks

New QFR Projection Upload:
  Frontend: qfrModeStore updates projection state
  Backend: projection.reset() → clears seg/centerline/diameter

Calibration Change:
  Frontend: calibrationStore updates localStorage
  Backend: set_calibrated_pixel_spacing() → module-level override
  Note: Does NOT invalidate existing QCA measurements (they used old spacing)
```

---

## 6. Concurrency Model

### 6.1 Backend

**Runtime**: Single-process uvicorn with async event loop. All route handlers are `async def` but most computation is CPU-bound (numpy, OpenCV, PyTorch).

**Thread safety**: NOT safe. Module-level state (`_current_study`, `_qfr_sessions`, `_engine`, etc.) has no locks. Concurrent requests could corrupt state:
- Two simultaneous DICOM uploads → last one wins, previous session lost
- Concurrent segmentation + QCA → fine (reads same study, produces independent results)
- Concurrent motion calculate → `_engine` state could be corrupted

**ML model inference**: PyTorch inference is synchronous and blocks the event loop. A long segmentation request (~1-5s) blocks all other requests. No `run_in_executor()` wrapping is used.

**Recommendation**: For the current single-user desktop app, this is acceptable. For multi-user, need either:
- Thread pool executor for CPU-bound tasks
- Per-session state isolation (not module-level globals)
- Separate worker process for ML inference

### 6.2 Frontend

**Main thread**: All React rendering, Zustand store updates, canvas drawing, and API calls happen on the main thread.

**Web Worker**: `dicomDecodeWorker.ts` — handles Cornerstone.js DICOM frame decoding. However, the current implementation primarily uses backend-decoded base64 PNG frames, so the worker is underutilized.

**Potential race conditions**:
1. **Frame preloading during playback**: Multiple concurrent `getFrame()` calls. Mitigated by checking `seriesInstanceUid` hasn't changed before storing.
2. **Concurrent segmentation requests**: User rapidly clicks "Segment" on different frames. Each request is independent, but Map updates are not atomic (Zustand immutable updates via `new Map(state.frames)` are safe).
3. **Playback speed change during animation**: `requestAnimationFrame` loop reads `playbackSpeed` from store. Zustand's `subscribeWithSelector` ensures latest value.

### 6.3 What Happens with Concurrent Segmentation Requests?

```
Request 1: POST /segmentation/segment (frame 5)
Request 2: POST /segmentation/segment (frame 10)

Backend:
  Both read _current_study (same object, immutable frames) ✓
  Both call NNUNetEngine.predict() sequentially (GIL + async serialization)
  No shared mutable state between requests ✓

Frontend:
  Both store results in different Map keys (frame 5 vs 10) ✓
  Zustand creates new Map instances (immutable) ✓

Result: Safe, but serialized (no parallelism for GPU inference)
```

---

## 7. Security Architecture

### 7.1 Authentication Flow

```
Current state: Authentication is STUBBED / NOT ENFORCED.

api.ts implements:
  ─ Token storage: localStorage (TOKEN_KEY, REFRESH_KEY, SESSION_KEY)
  ─ Token injection: Authorization header with Bearer token
  ─ Auto-refresh: On 401 response, attempts POST /auth/refresh
  ─ Session ID: X-Session-ID header on every request

Backend:
  ─ auth_routes.py exists but is NOT registered in main.py router list
  ─ No middleware checks JWT tokens
  ─ No route-level authorization
  ─ All endpoints are publicly accessible
```

### 7.2 CORS Configuration

```python
# settings.py
cors_origins = "http://localhost:1420,http://localhost:3000,http://localhost:5173,http://localhost:4173"

# main.py
CORSMiddleware(
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Allows all methods and all headers. Origins are restricted to localhost ports. In production Docker, the frontend is served from the same origin via Nginx reverse proxy, so CORS is less relevant.

### 7.3 Input Validation

**Pydantic models**: All request bodies are validated via Pydantic v2 BaseModel classes. Field constraints include:
- `ge=1, le=2` for projection_id
- `gt=0` for diameter values
- `ge=4, le=8` for catheter French sizes

**File upload**: No DICOM-specific validation beyond pydicom parsing. No file size check (500MB limit in settings but not enforced in middleware). No DICOM anonymization on upload (only on export).

**Path traversal**: `browse_directory` and `scan_directory` accept arbitrary server paths. No allowlist or chroot. The `load_dicom_path` endpoint opens any file on the server filesystem.

### 7.4 Tauri Security

```json
{
  "security": { "csp": null },           // CSP disabled entirely
  "plugins": {
    "shell": { "scope": [...] },          // Scoped to python3 command only
    "fs": { "scope": ["**"] },            // Full filesystem access
    "dialog": { "open": true, "save": true }
  }
}
```

CSP is null (relaxed) for the medical imaging app. Filesystem scope is `**` (unrestricted). Shell plugin is scoped to only launching the Python backend.

### 7.5 DICOM Anonymization

Anonymization is available but NOT automatic:
- `anonymize_dicom_metadata()` in `domain/services/anonymization.py`
- Called on export (if requested)
- HIPAA identifiers list (`HIPAA_IDENTIFIERS`)
- Not applied during DICOM loading or storage

---

## 8. API Design Analysis

### 8.1 RESTful Compliance

**Partially RESTful**. Key deviations:

| Issue | Example | Impact |
|---|---|---|
| Verb in URL | `/segmentation/segment-and-extract` | Should be `POST /segmentation` with body params |
| No resource IDs | `GET /dicom/metadata` (global) | Should be `GET /studies/{id}/metadata` |
| Mixed session models | Some routes use path `/{session_id}/...`, others use module state | Inconsistent |
| RPC-style endpoints | `POST /motion/calculate`, `POST /rws/clear` | Not resource-oriented |
| No HATEOAS | Responses don't link to related resources | Not a major issue for SPA |

### 8.2 Payload Analysis

**Oversized payloads**:
- `GET /dicom/frames?start=0&end=150`: Returns ALL frames as base64 PNGs in a single JSON response. For 150 frames at 512x512, this is ~45-100MB of JSON.
- `POST /qfr/reconstruct` response: Includes full mesh vertices/faces arrays (potentially 10K+ numbers).

**Inefficiencies**:
- Base64 encoding adds 33% overhead to every image
- Segmentation mask is full-frame PNG even when only a small ROI was segmented
- No compression (gzip not configured in FastAPI, would help significantly for base64 data)

### 8.3 Missing Standard Patterns

| Pattern | Status | Notes |
|---|---|---|
| Pagination | Missing | Not needed (single-study app) |
| Filtering | Missing | Not needed |
| Sorting | Missing | Not needed |
| Versioning | Missing | No `/v1/` prefix, no version header |
| Rate limiting | Missing | Not needed for desktop app |
| ETags/Caching headers | Missing | Would help for frame caching |
| Content negotiation | Missing | Always JSON + base64 |
| Streaming | Missing | Could use SSE for long operations |

### 8.4 Inconsistencies

1. **Base64 prefix**: Segmentation routes use `encode_mask()` which adds `data:image/png;base64,` prefix. QFR routes return raw base64 without prefix. Frontend must handle both.

2. **Error response format**: Some routes return `{"error": {"code": ..., "message": ...}}` (via global handlers), others return `{"detail": "..."}` (via HTTPException). Some return `{"success": false, "error_message": "..."}`.

3. **Session ID usage**: Three different session mechanisms coexist:
   - URL path parameter: `/{session_id}/frames`
   - Module-level state: `_current_study` (no session needed)
   - Header: `X-Session-ID` (for QFR mode)

4. **Route path conflicts**: Both `GET /dicom/frame/{frame_index}` (flat) and `GET /dicom/{session_id}/frames/{frame_index}` (session) exist.

---

## 9. Module Boundaries

### 9.1 Clean Architecture Compliance Audit

**Domain Layer** (`domain/`):
- **entities/**: Pure data classes with business methods. No infrastructure imports. COMPLIANT.
- **services/**: Pure computation (numpy/scipy). No I/O, no HTTP, no database. COMPLIANT.
- **value_objects/**: Immutable data holders with validation. COMPLIANT.
- **exceptions/**: Domain-specific error hierarchy. COMPLIANT.
- **qfr_session.py**: Contains module-level dict `_qfr_sessions` (global mutable state). VIOLATION — this is infrastructure concern (session storage) in the domain layer.

**Application Layer** (`application/use_cases/`):
- Orchestrates domain services and infrastructure. MOSTLY COMPLIANT.
- `LoadStudyUseCase` depends on `DicomHandler` and `ECGParser` (infrastructure) via constructor injection. COMPLIANT.
- Several use cases are EMPTY stubs (calibration, export `__init__.py` files). NOT USED — routes bypass use cases.

**Presentation Layer** (`presentation/routes/`):
- Routes do too much. Many contain inline business logic instead of delegating to use cases. VIOLATION.
- Example: `rws_routes.py` contains `_compute_rws_from_diameters()` which computes RWS directly instead of going through `CalculateRWSUseCase`.
- Example: `segmentation_routes.py` directly instantiates `CenterlineExtractor` and ML engines instead of going through `SegmentVesselUseCase`.
- Example: `qfr_routes.py` reconstruct endpoint (~370 lines) contains inline 3D reconstruction, diameter measurement, QFR calculation, and mesh generation.

**Infrastructure Layer** (`infrastructure/`):
- ML engines: Properly isolated behind abstract base class (`BaseSegmentationEngine`). COMPLIANT.
- File handlers: Properly isolated. COMPLIANT.
- Config: DI container uses `@lru_cache()` singletons and `Depends()`. Appropriate.

### 9.2 Where Domain Logic Leaks into Infrastructure

1. **qfr_session.py** is in `domain/` but manages session state (an infrastructure concern).
2. ML engine selection logic is in route handlers, not in a use case or service.
3. Catheter size constants (`_CATHETER_DIAMETERS`) are duplicated in both `calibration_engine.py` (domain) and `qfr_routes.py` (presentation).

### 9.3 Where Presentation Logic Leaks into Domain

1. `_build_metadata_dict()` in `dicom_routes.py` constructs frontend-specific response shapes. This is presentation concern, not leaked into domain — it is in the correct layer.
2. Domain services are properly isolated. No presentation leakage detected.

### 9.4 Tight Coupling Points

1. **`dicom_routes._current_study`**: 6+ other route modules import and depend on this module-level variable.
2. **`api.ts`**: Single 2600-line file coupling all frontend code to all backend endpoints.
3. **Base64 encoding**: Spread across multiple files (segmentation_routes, qfr_routes, dicom_routes, calibration_routes). Each re-implements PNG→base64 conversion.
4. **Pixel spacing calculation**: Duplicated in `qfr_routes.py` (3 times), `calibration_routes.py`, and `qca_engine.py`. The `SOD/SID` magnification correction is copy-pasted.

---

## 10. Scalability Analysis

### 10.1 What Breaks with Large DICOM Files (1000+ Frames)?

**Backend**:
- `load_all_frames=True` → all 1000 frames loaded into memory as numpy arrays
- Memory: 1000 frames × 512×512 × 2 bytes (uint16) = ~500MB RAM just for pixels
- `GET /dicom/frames?start=0&end=1000` → 1000 base64 PNGs in one response → ~100-300MB JSON response

**Frontend**:
- Frame cache (`Map<number, FrameData>`) grows unbounded
- Each base64 frame stored as string (~100-300KB)
- 1000 cached frames → ~100-300MB in JS heap
- Canvas re-render on frame change: decoding base64→Image→canvas on each frame tick

**Mitigation needed**:
- Lazy frame loading (already partially implemented via getFrame on-demand)
- LRU eviction for frame cache
- Binary frame transport (ArrayBuffer instead of base64 JSON)
- Server-side frame range streaming
- GPU-accelerated rendering via Cornerstone.js (Web Worker already exists but underused)

### 10.2 Memory Footprint Analysis

| Component | Typical Study (120 frames) | Large Study (1000 frames) |
|---|---|---|
| Backend: Study.frames (pixel_data) | ~30MB | ~500MB |
| Backend: Segmentation masks (if cached) | ~15MB | ~125MB |
| Frontend: Frame cache (base64) | ~20MB | ~200MB |
| Frontend: Segmentation cache | ~10MB | ~80MB |
| ML model (nnU-Net) | ~200MB GPU/CPU | Same |
| ML model (AngioPy) | ~150MB GPU/CPU | Same |
| Total (backend) | ~400MB | ~1GB+ |
| Total (frontend) | ~30MB | ~280MB |

### 10.3 Concurrent User Support

**Current**: Single-user only. Module-level `_current_study` is overwritten by each DICOM load. No per-user isolation.

**To support N users**: Need per-session state with proper session management (Redis or DB-backed), user authentication, and resource limits.

### 10.4 ML Model Loading

- **Cold start**: First segmentation request loads model into memory. nnU-Net takes 5-15 seconds to load. AngioPy takes 3-8 seconds.
- **Warm inference**: 50-500ms per frame depending on model and device.
- **Memory**: Models stay loaded for process lifetime via `@lru_cache()` singletons.
- **Multiple models**: Loading both nnU-Net and AngioPy simultaneously requires ~350MB+ GPU memory.

---

## 11. Testing Architecture

### 11.1 Current Test Structure

```
python-backend/tests/
  ├── unit/
  │   ├── test_rws.py         # RWS calculator unit tests
  │   ├── test_qca.py         # QCA calculator tests
  │   └── ...
  ├── integration/
  │   └── ...
  └── conftest.py
```

Configuration (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --cov=app --cov-report=term-missing"
```

Frontend testing:
```
# Unit tests
npm run test          # Vitest with happy-dom/jsdom
npm run test:coverage # + coverage reporters

# E2E tests
npm run test:e2e      # Playwright (Chromium)
```

### 11.2 What Is Testable vs. Not

**Easy to test** (pure functions, no I/O):
- `domain/services/rws_calculator.py` — pure numpy computation
- `domain/services/qfr_calculator_3d.py` — pure pressure model
- `domain/services/stereo_reconstructor.py` — pure linear algebra
- `domain/services/qca_engine.py` — pure image processing
- `domain/services/centerline_extractor.py` — pure morphology
- `domain/value_objects/*` — immutable data with validation
- `domain/entities/*` — business logic methods

**Hard to test** (side effects, external deps):
- Route handlers — deep dependency chains, module-level state
- ML engine inference — requires model files and GPU/CPU
- DICOM handler — requires real DICOM files
- Frontend stores — async API calls, Map state management
- Canvas rendering — DOM and pixel-level verification

### 11.3 Mock/Stub Patterns

**Backend**:
- ML engines have `ThresholdSegmentationEngine` fallback when nnU-Net not installed
- `dependencies.py` uses `Depends()` which can be overridden via `app.dependency_overrides`
- AngioPy/SAM2/SegFormer/HRNet availability checked via `try/except ImportError`

**Frontend**:
- No mocking infrastructure detected
- Could mock `api.ts` functions for store testing
- Could use MSW (Mock Service Worker) for API layer testing

### 11.4 Recommended Testing Strategy for Rewrite

1. **Domain services**: 100% unit test coverage. Test edge cases (zero-length vessels, NaN diameters, extreme angles).
2. **Use cases**: Integration tests with mocked infrastructure. Test orchestration logic.
3. **Routes**: API integration tests with `httpx.AsyncClient` against test app. Test error handling.
4. **Frontend stores**: Unit tests with mocked API layer. Test state transitions.
5. **E2E**: Playwright tests for critical paths (load DICOM → segment → calculate QCA → compute RWS).
6. **ML engines**: Snapshot tests with known input/output pairs. Separate CI job with model files.

---

## 12. Deployment Architecture

### 12.1 Tauri Desktop Packaging

```
npm run tauri:build
  │
  ├─▶ npm run build (Vite → dist/)
  │     ├─ TypeScript compilation (tsc)
  │     ├─ Vite bundling (3526 modules)
  │     ├─ WASM codec bundling (libjpeg-turbo, CharLS, OpenJPEG, OpenJPH)
  │     └─ Output: dist/ (static HTML/JS/CSS)
  │
  └─▶ cargo build (Rust → native binary)
        ├─ Links Tauri runtime
        ├─ Embeds dist/ as frontend asset
        ├─ LTO, single codegen unit, panic=abort, opt-level=z
        └─ Output: platform-specific binary (.deb, .msi, .dmg)

Runtime:
  Binary launches → creates webview window (1400x900)
  Shell plugin spawns: python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  Webview loads frontend → frontend calls http://127.0.0.1:8000
```

### 12.2 Docker Multi-Stage Build

```yaml
# docker-compose.yml (Production)
services:
  frontend:
    image: nginx:alpine
    ports: ["80:80"]
    volumes: [./dist:/usr/share/nginx/html, ./nginx.conf]
    # Reverse proxy: /api → backend:8000

  backend:
    build: ./python-backend
    ports: ["8000:8000"]
    environment: [APP_ENV=production]
    volumes: [./models:/app/models]  # ML model weights
    healthcheck: GET http://localhost:8000/health

  postgres:
    image: postgres:15
    ports: ["5432:5432"]

  redis:
    image: redis:7
    ports: ["6379:6379"]
```

**Note**: PostgreSQL and Redis are provisioned but NOT USED by the application. They exist for future database persistence and session caching.

### 12.3 Environment Configuration

```
APP_ENV=development|production|test → selects Settings subclass
.env file → loaded by pydantic-settings BaseSettings

Key environment variables:
  CORS_ORIGINS  — comma-separated allowed origins
  DEVICE        — cpu|cuda|mps (ML model device)
  ML_MODELS_PATH — directory containing model weights
  LOG_LEVEL     — DEBUG|INFO|WARNING|ERROR
  HOST/PORT     — server binding (default 0.0.0.0:8000)
```

### 12.4 Health Check Strategy

**Implemented**:
- `GET /health` — returns `{"status": "healthy"}` (basic)
- `GET /status` — service status
- `GET /version` — version info
- `GET /dicom/health` — DICOM service health
- `GET /segmentation/health` — segmentation engine availability
- `GET /rws/health` — RWS service health

**Missing**:
- No deep health checks (database, Redis, ML model loaded, GPU available)
- No frontend health monitoring
- No backend-to-frontend health push
- Docker healthcheck configured but only checks basic HTTP response

---

## 13. Recommendations for Rewrite

### 13.1 Architectural Patterns to KEEP

1. **DDD Layer Structure**: The `domain/entities/`, `domain/services/`, `domain/value_objects/`, `domain/exceptions/` organization is clean and testable. Keep it.
2. **Zustand Stores**: Per-domain stores are the right granularity. Keep the store-per-feature pattern.
3. **Factory Functions**: `create_rws_calculator()`, `create_stereo_reconstructor()` etc. provide good testability.
4. **Lazy ML Loading**: Models loaded on first use, not at startup. Essential for fast startup.
5. **Multi-Canvas Layer System**: Independent layer rendering (video/mask/annotation/overlay) is efficient.
6. **Epipolar-Matched Diameter Measurement**: The `ReconstructionResult` with matched 2D point pairs is the correct approach. Do not revert to uniform resampling.

### 13.2 What to Change Fundamentally

1. **Module-Level State → Session Service**: Replace all `_current_study`, `_qfr_sessions`, `_engine`, `_rws_results`, `_calibrated_pixel_spacing` module globals with a proper `SessionService` that holds per-session state in a typed dataclass. Use FastAPI dependency injection to provide it to routes.

2. **Use Cases Actually Used**: Route handlers should NEVER contain business logic. All computation flows through use cases. The 370-line `reconstruct_qfr()` endpoint should be `ReconstructQFRUseCase.execute()`.

3. **Binary Frame Transport**: Replace base64 PNG with binary responses (`StreamingResponse` with `application/octet-stream`). Frontend decodes via `ArrayBuffer` → `ImageBitmap`. This eliminates the 33% overhead and reduces JSON parse cost.

4. **Split api.ts**: Break the 2600-line monolith into modules matching backend route groups:
   ```
   src/lib/api/
     ├── client.ts      (base fetch, headers, auth)
     ├── dicom.ts
     ├── segmentation.ts
     ├── qca.ts
     ├── rws.ts
     ├── qfr.ts
     ├── motion.ts
     ├── calibration.ts
     └── export.ts
   ```

5. **WebSocket for Progress**: Add WebSocket channel for long-running operations (segmentation progress, tracking propagation progress, frame preloading status). Current polling approach is insufficient.

6. **Standardize Error Format**: All endpoints should return errors in a consistent format:
   ```json
   {"error": {"code": "SEGMENTATION_FAILED", "message": "...", "details": {}}}
   ```

7. **Standardize Base64 Prefix**: Always include `data:image/png;base64,` prefix in ALL image responses. Remove the frontend workaround that adds it conditionally.

### 13.3 Suggested New Patterns

1. **Event-Driven State Updates**: Instead of stores directly calling each other (`dicomStore → playerStore`), use an event bus or Zustand middleware that reacts to state changes:
   ```
   dicomStore.metadata changed → player auto-updates totalFrames
   segmentationStore.mask changed → qcaStore auto-invalidates
   calibrationStore.pixelSpacing changed → qcaStore recalculates
   ```

2. **Command/Query Separation (CQRS-lite)**: Separate read endpoints (GET frame, GET metadata) from write endpoints (POST segment, POST calculate). Reads can be cached more aggressively.

3. **Result Cache with Content Addressing**: Hash-based cache keys (e.g., `hash(frame_pixels + engine + roi)`) to avoid re-segmenting identical inputs.

4. **Frame LRU Cache**: Implement a bounded LRU cache for frames (e.g., keep last 200 frames, evict oldest on overflow). Critical for 1000+ frame studies.

5. **Structured Logging**: Replace `print()` statements and ad-hoc `logger.info()` with structured JSON logging (e.g., `structlog`). Include session_id, operation, timing in every log entry.

### 13.4 Technology Recommendations

| Current | Recommendation | Reason |
|---|---|---|
| FastAPI + uvicorn | **Keep** | Good async support, Pydantic integration |
| Zustand | **Keep** | Lightweight, simple, well-suited |
| base64 PNG transport | **Replace with binary** | 33% overhead, slow JSON parse |
| Module-level state | **Replace with session service** | Testability, multi-user readiness |
| No WebSocket | **Add WebSocket** (FastAPI WebSocket) | Progress reporting, frame streaming |
| No database | **Add SQLite** (via SQLAlchemy) | Session persistence, study history |
| No frame eviction | **Add LRU cache** | Memory management for large studies |
| matplotlib (optional) | **Replace with ReportLab** | Already a dependency, eliminates optional dep headache |
| Cornerstone.js (underused) | **Evaluate**: either use fully or remove | Currently loading heavy WASM codecs but barely used |
| React Router | **Keep** | Simple routing needs |
| Tailwind CSS | **Keep** | Productive, well-integrated |
| Three.js | **Keep** | Only used for 3D mesh, lightweight usage |

---

*End of architecture document. This describes every architectural detail of Coronary RWS Analyser v1.1 as of February 2026.*
