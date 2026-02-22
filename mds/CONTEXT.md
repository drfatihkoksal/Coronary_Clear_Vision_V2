# Coronary RWS Analyser v2 — Application Context & Specification

> This document describes every feature, algorithm, data flow, and architectural detail of the Coronary RWS Analyser application. It is intended as the single source of truth for a complete rewrite.

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Technology Stack](#2-technology-stack)
3. [Architecture](#3-architecture)
4. [Data Model & Entities](#4-data-model--entities)
5. [Feature Catalog — Frontend](#5-feature-catalog--frontend)
6. [Feature Catalog — Backend](#6-feature-catalog--backend)
7. [API Contract](#7-api-contract)
8. [Algorithms & Domain Logic](#8-algorithms--domain-logic)
9. [ML / AI Engines](#9-ml--ai-engines)
10. [UI/UX Specification](#10-uiux-specification)
11. [Configuration & Build](#11-configuration--build)
12. [Deployment](#12-deployment)
13. [Known Issues & Design Lessons](#13-known-issues--design-lessons)

---

## 1. Product Overview

**Coronary RWS Analyser** is a desktop medical imaging application for coronary artery analysis from X-ray angiography (fluoroscopy) DICOM files.

### What It Does
- Loads multi-frame DICOM angiograms (coronary cine loops)
- Segments coronary arteries using deep learning (nnU-Net, AngioPy, SAM2, etc.)
- Measures vessel diameters along the centerline (Quantitative Coronary Analysis — QCA)
- Calculates **Radial Wall Strain (RWS)** — vessel diameter variation during the cardiac cycle
- Calculates **Quantitative Flow Ratio (QFR)** — functional significance of coronary stenosis from 3D reconstruction
- Provides ECG-gated analysis with beat boundary detection
- Supports dual-projection stereo 3D reconstruction of coronary vessels
- Generates clinical reports (PDF, CSV, Excel)

### Clinical Purpose
- **RWS** detects functionally significant stenosis by measuring how much a vessel segment changes diameter during the cardiac cycle. High RWS (>12%) suggests a vulnerable plaque.
- **QFR** estimates the fractional flow reserve (FFR) non-invasively from angiographic images, replacing the need for a pressure wire. QFR < 0.80 suggests hemodynamically significant stenosis.

### Target Users
- Interventional cardiologists
- Catheterization lab technicians
- Cardiovascular research teams

---

## 2. Technology Stack

### Frontend
| Category | Technology | Version |
|----------|-----------|---------|
| Framework | React + TypeScript | 18.2+ / 5.3 |
| Build Tool | Vite | 5.4 |
| Desktop Shell | Tauri | 2.0 |
| State Management | Zustand | 5.0 |
| Data Fetching | Axios + TanStack React Query | 1.13 / 5.90 |
| Routing | React Router DOM | 7.11 |
| UI Components | Radix UI (primitives) | Latest |
| Styling | Tailwind CSS | 3.3 |
| Charts | Recharts | 2.12 |
| 3D Rendering | Three.js + @react-three/fiber + drei | 0.160 |
| DICOM Decoding | Cornerstone.js (core + dicom-image-loader + tools) | 4.15 |
| DICOM Parsing | dicom-parser | Latest |
| Icons | Lucide React | Latest |
| Testing | Vitest (unit) + Playwright (E2E) | 2.1 / 1.40 |

### Backend
| Category | Technology | Version |
|----------|-----------|---------|
| Framework | FastAPI | 0.109+ |
| Server | Uvicorn | 0.27+ |
| Validation | Pydantic v2 | 2.5+ |
| DICOM | pydicom + pylibjpeg (JPEG, JPEG2000) | 2.4+ |
| Image Processing | OpenCV, Pillow, scikit-image | 4.8+ |
| Scientific | NumPy, SciPy | 1.24+ / 1.11+ |
| ML/DL | PyTorch, nnU-Net v2, segmentation-models-pytorch, timm | 2.0+ |
| PDF Export | matplotlib (optional), ReportLab | 3.7+ / 4.0+ |
| Testing | pytest, pytest-asyncio, pytest-cov | 7.4+ |
| Linting | Black (100 cols), Ruff, MyPy (strict) | Latest |

### Infrastructure
| Category | Technology |
|----------|-----------|
| Containerization | Docker + Docker Compose |
| Production Web Server | Nginx |
| Database (future) | PostgreSQL 15 |
| Cache (future) | Redis 7 |

---

## 3. Architecture

### 3.1 High-Level

```
┌────────────────────────────────────────────────────┐
│                    Tauri Shell                      │
│  ┌─────────────────────┐  ┌──────────────────────┐ │
│  │   React Frontend    │  │  Python Backend      │ │
│  │   (Vite, port 1420) │←→│  (FastAPI, port 8000)│ │
│  │                     │  │                      │ │
│  │  Zustand Stores     │  │  Domain Services     │ │
│  │  Canvas Layers      │  │  ML Engines          │ │
│  │  Three.js 3D        │  │  DICOM Handler       │ │
│  └─────────────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────┘
```

- **Tauri** wraps both processes: launches the Python backend via shell plugin, serves the frontend.
- Communication: REST API over HTTP (JSON + base64-encoded images).
- No WebSocket — all communication is request/response.

### 3.2 Frontend Architecture

```
src/
├── main.tsx                    # Entry: React DOM render
├── App.tsx                     # Router: / → /studies, /app → AnalysisApp
├── AnalysisApp.tsx             # Main layout: toolbar + viewer + panels
├── components/
│   ├── Panels/                 # Right sidebar tabs (12 components)
│   ├── Viewer/                 # DICOM viewers (6 components)
│   ├── Controls/               # Toolbar + playback (3 components)
│   ├── Charts/                 # RWS & QCA charts (2 components)
│   ├── Common/                 # Shared UI (10 components)
│   ├── MaskEdit/               # Mask editing tools
│   └── Dialogs/                # Modal dialogs
├── stores/                     # 17 Zustand stores
├── hooks/                      # useCanvasLayers, useKeyboardShortcuts, useTheme
├── lib/
│   ├── api.ts                  # API client (~2600 lines, all backend calls)
│   ├── canvas/                 # Multi-layer canvas system
│   └── dicomDecodeWorker.ts    # Web Worker for DICOM frame decoding
├── types/                      # TypeScript interfaces
└── index.css                   # Tailwind + theme variables
```

### 3.3 Backend Architecture (DDD / Clean Architecture)

```
python-backend/app/
├── main.py                          # FastAPI app factory, middleware, route registration
├── presentation/
│   ├── routes/                      # 16+ route files (HTTP endpoints)
│   └── schemas/                     # Pydantic request/response models
├── application/
│   └── use_cases/                   # Orchestration (load study, segment, calculate QFR/RWS)
├── domain/
│   ├── entities/                    # Study, Frame, Vessel, Segmentation, ECGSignal
│   ├── services/                    # Pure computation (18 services)
│   ├── value_objects/               # PixelSpacing, Diameter, QFRResult, RWSResult, Position
│   ├── exceptions/                  # Domain-specific errors
│   └── qfr_session.py              # QFR dual-projection session state
└── infrastructure/
    ├── ml_engines/                  # nnU-Net, AngioPy, SAM2, SegFormer, HRNet
    ├── file_handlers/               # DICOM handler, ECG parser, PDF reporter
    ├── config/                      # Settings, dependency injection
    └── persistence/                 # Future DB integration
```

### 3.4 Session & State Management

**Backend state is module-level (not DB-backed):**
- `dicom_routes.py` holds `_current_study` (the loaded Study entity)
- All other routes access it via `from .dicom_routes import _get_current_study`
- QFR mode has its own session store (`qfr_session.py`) keyed by `X-Session-ID` header
- This means: **single-user, single-study at a time**

**Frontend state is Zustand (in-memory, per-session):**
- 17 stores, each managing a specific domain
- `settingsStore` and `calibrationStore` persist to localStorage
- No backend persistence layer

---

## 4. Data Model & Entities

### 4.1 Study (Aggregate Root)
```
Study
├── id: UUID
├── patient_info: PatientInfo
│   ├── patient_id: str
│   ├── patient_name: str
│   ├── birth_date: str
│   ├── sex: str
│   └── age: int
├── study_info: StudyInfo
│   ├── study_uid: str
│   ├── study_date: str
│   ├── study_time: str
│   ├── description: str
│   ├── institution: str
│   └── modality: str
├── frames: List[Frame]
│   ├── index: int
│   ├── pixel_data: np.ndarray (H×W, uint8/uint16)
│   ├── width, height: int
│   └── timestamp_ms: float
├── vessels: List[Vessel]
│   ├── id: str
│   ├── name: str (LAD, LCx, RCA, etc.)
│   ├── centerline: List[Tuple[float, float]]
│   ├── diameter_profile: List[float]
│   └── quality_score: float
├── segmentations: Dict[int, List[Segmentation]]
│   ├── mask: np.ndarray (binary 0/255)
│   ├── method: SegmentationMethod
│   ├── confidence: float
│   └── frame_index: int
├── ecg_signal: Optional[ECGSignal]
│   ├── samples: np.ndarray (voltage)
│   ├── sample_rate: float (Hz, typically 1000)
│   ├── r_peaks: List[RPeak]
│   ├── duration_seconds: float
│   └── heart_rate: float (bpm)
├── pixel_spacing: Optional[PixelSpacing]
│   ├── row_spacing: float (mm/px)
│   ├── column_spacing: float (mm/px)
│   ├── calibration_method: enum
│   └── confidence: float
├── frame_rate: float (fps)
├── image_width, image_height: int
├── qca_measurements: Dict[int, QCAMeasurement]
├── metadata: Dict[str, Any]
│   ├── positioner_primary_angle: float (RAO/LAO degrees)
│   ├── positioner_secondary_angle: float (CRA/CAU degrees)
│   ├── distance_source_to_detector: float (SID, mm)
│   ├── distance_source_to_patient: float (SOD, mm)
│   ├── raw_pixel_spacing: Tuple[float, float]
│   ├── raw_imager_pixel_spacing: Tuple[float, float]
│   ├── intensifier_size: float (mm)
│   ├── frame_time: float (ms)
│   ├── manufacturer: str
│   ├── bits_stored: int
│   └── ecg_type: str
└── created_at: datetime
```

### 4.2 QCA Measurement
```
QCAMeasurement
├── frame_index: int
├── diameter_profile: List[float] (mm, 50 points along centerline)
├── centerline: List[Tuple[float, float]]
├── mld: float (Minimum Lumen Diameter, mm)
├── proximal_rd: float (Proximal Reference Diameter, mm)
├── distal_rd: float (Distal Reference Diameter, mm)
├── diameter_stenosis: float (% DS = (1 - MLD/RD) × 100)
└── lesion_length: float (mm)
```

### 4.3 RWS Result
```
RWSResult
├── beat_number: int
├── mld_rws: float (%)
├── proximal_rws: float (%)
├── distal_rws: float (%)
├── average_rws: float (%)
├── interpretation: "normal" | "intermediate" | "vulnerable" | "high_risk"
│   ├── < 8%: normal
│   ├── 8-12%: intermediate
│   ├── 12-14%: vulnerable
│   └── > 14%: high_risk
├── start_frame: int
├── end_frame: int
├── outlier_method: str
└── vessel: str (LAD, LCX, RCA, etc.)
```

### 4.4 QFR Result
```
QFR3DResult
├── qfr_value: float (0-1, <0.80 = significant stenosis)
├── pressure_drop_mmhg: float
├── flow_velocity_m_s: float
├── vessel_length_mm: float
├── min_diameter_mm: float
├── mean_diameter_mm: float
├── stenosis_percent: float
├── mode: "fQFR" | "cQFR" | "aQFR"
├── resting_velocity: float
├── hyperemic_velocity: float
├── qfr_profile: List[float] (along-vessel QFR values)
├── pressure_profile: List[float] (along-vessel pressure)
└── timi_frame_count: float
```

### 4.5 QFR Session (Dual Projection)
```
QFRModeSession
├── session_id: str
├── projection1: ProjectionData
│   ├── study: Study
│   ├── metadata: dict
│   ├── centerline_2d: List[Tuple[float, float]]
│   ├── diameter_profile: List[float]
│   ├── segmentation_mask: np.ndarray
│   ├── primary_angle: float (degrees)
│   ├── secondary_angle: float (degrees)
│   ├── sid: float (mm)
│   ├── sod: float (mm)
│   ├── pixel_spacing: float (mm/px)
│   └── calibration_pixel_spacing: Optional[float]
├── projection2: ProjectionData (same structure)
├── centerline_3d: Optional[np.ndarray] (Nx3)
├── diameter_3d: Optional[List[float]]
├── mesh: Optional[VesselMesh]
└── qfr_result: Optional[QFR3DResult]
```

### 4.6 Vessel Mesh (3D)
```
VesselMesh
├── vertices: np.ndarray (Nx3, mm)
├── faces: np.ndarray (Mx3, triangle indices)
└── colors: Optional[np.ndarray] (Mx3, RGB, QFR heatmap)
```

### 4.7 Vessel Types
```
LAD   — Left Anterior Descending
LCX   — Left Circumflex
RCA   — Right Coronary Artery
LM    — Left Main
Diagonal
OM    — Obtuse Marginal
IM    — Intermediate (Ramus)
PLA   — Posterolateral Artery
PDA   — Posterior Descending Artery
```

---

## 5. Feature Catalog — Frontend

### 5.1 DICOM Loading & Viewing
- **File upload** via button or drag-drop
- **File path loading** for Tauri desktop (bypasses upload)
- **Privacy dialog** on load: asks whether to anonymize DICOM metadata (HIPAA)
- **Multi-series support**: if DICOM contains multiple series, show series picker with thumbnails
- **Directory scanning**: browse server-side DICOM directories (recursive)
- **Frame display**: multi-canvas layer system renders frames at up to 60fps
- **Zoom**: scroll wheel, center-aware (zoom toward cursor), range 0.1x–10x
- **Pan**: drag in pan mode [H key] or middle-mouse
- **Reset view**: fit-to-window [R key]
- **Playback**: play/pause [Space], step frame [Arrow keys], speed control (0.25x–2x), loop toggle
- **Playback range**: restrict to frame range (beat-based)
- **Frame counter**: current / total frames display
- **DICOM metadata panel**: patient info, imaging angles, frame rate, pixel spacing, manufacturer

### 5.2 Segmentation
- **8 engine types**:
  - `nnunet` — nnU-Net v2 with ROI
  - `nnunet-wide` — nnU-Net wider ROI variant
  - `nnunet-fullframe` — nnU-Net without ROI (full image)
  - `angiopy` — Seed-guided U-Net + InceptionResNetV2 (requires 2-10 seed points)
  - `roi+angiopy` — nnU-Net fullframe → auto-seed extraction → AngioPy refinement
  - `sam2` — Segment Anything Model 2 (stub)
  - `segformer` — Semantic segmentation transformer (stub)
  - `hrnet` — High-resolution network with vessel class labels (stub)
- **Seed points**: place on vessel [S key], 2–10 points (proximal→distal)
- **ROI**: fixed 160×160 bounding box [B key], draggable
- **Combined segment+extract**: one-click segmentation + centerline extraction
- **Per-frame caching**: results stored per frame, avoid recomputation
- **Mask overlay**: semi-transparent colored overlay on viewer
- **Centerline overlay**: connected point path along vessel center
- **Probability map**: confidence heatmap visualization (optional)
- **Vessel detection**: HRNet identifies vessel type (LAD, LCX, RCA)

### 5.3 Centerline Extraction
- **Methods**: skeleton (default), distance transform, MCP (minimal cost path), MCP auto
- **Seed point generation**: extract seed points from centerline for AngioPy

### 5.4 Mask Editing
- **Brush tool**: paint vessel mask (adjustable size, hardness)
- **Eraser tool**: remove mask regions (adjustable size)
- **Smart brush**: edge-aware painting (intensity tolerance, contour influence, edge sensitivity)
- **Flood fill**: fill connected region with tolerance
- **Morphological ops**: dilate, erode, fill holes, remove islands, smooth
- **Contour extraction**: mask → contour points (with simplification)
- **Contour deformation**: interactive dragging with Gaussian influence
- **Edge snapping**: snap mask to image gradient edges
- **Region growing**: seed-based flood fill
- **Magic wand**: intensity-based selection
- **Mask interpolation**: temporal smoothing between keyframes
- **Undo/redo**: edit history
- **After editing**: re-extract centerline → recalculate QCA

### 5.5 QCA (Quantitative Coronary Analysis)
- **Calculation**: Gaussian subpixel fitting perpendicular to centerline
- **Methods**: gaussian (default), parabolic, threshold
- **Output**: diameter profile (50 points), MLD, proximal/distal reference diameters, %DS, lesion length
- **Display**: diameter profile chart (Recharts), QCA markers on viewer
- **Marker editing**: drag QCA markers to adjust measurements
- **Pixel-to-mm conversion**: requires calibration

### 5.6 Calibration
- **Sources**:
  - DICOM metadata (pixel spacing from header)
  - Catheter-based: known catheter size (4–8 French, 1.33–2.67mm) ÷ measured width in pixels
  - Manual: user enters pixel spacing directly
  - From mask: automatic catheter measurement from segmentation mask
- **Persistent**: saved to localStorage, survives page reload
- **Conversion helpers**: pixels↔mm bidirectional

### 5.7 ECG
- **Extraction**: from DICOM WaveformSequence (0008,104C)
- **R-peak detection**: Pan-Tompkins algorithm
- **Siemens ECG filter**: vendor-specific 50/60 Hz notch + preprocessing
- **Beat boundaries**: computed from R-peaks (frame indices)
- **Heart rate**: from R-R intervals
- **R-peak editing**: add, remove, move peaks manually → recalculate beat boundaries
- **ECG overlay**: optional signal display on viewer
- **Beat selection**: choose which cardiac cycle to analyze for RWS

### 5.8 Motion Signal (Alternative to ECG)
- **Algorithm**: Farneback dense optical flow (frame-to-frame motion magnitude)
- **Use case**: when DICOM lacks embedded ECG
- **Peak detection**: identifies cardiac phase events (systolic peaks)
- **Beat boundaries**: derived from motion peaks
- **Peak editing**: add, remove, move peaks manually
- **Signal overlay**: optional display on viewer

### 5.9 RWS (Radial Wall Strain)
- **Formula**: RWS = (Dmax − Dmin) / Dmax × 100%
- **Positions**: MLD, proximal reference, distal reference (each independently)
- **Input**: QCA diameter measurements across a cardiac cycle (beat)
- **Outlier methods**: none, Hampel (default), Double Hampel, IQR, temporal
- **Physiological constraints**: 0.2–5.5mm diameter range, max 20% frame-to-frame change
- **Interpretation**:
  - < 8%: Normal
  - 8–12%: Intermediate
  - 12–14%: Vulnerable
  - > 14%: High-risk
- **Multiple results**: track RWS per beat, delete individual results
- **Vessel annotation**: label which vessel (LAD, LCX, RCA, etc.)
- **Summary statistics**: across all beats
- **Chart**: RWS values over time with beat boundaries (Recharts)

### 5.10 QFR (Quantitative Flow Ratio)

#### Single-View QFR (Legacy)
- 2D geometry-based estimation
- Less accurate, kept for quick assessment

#### Dual-Projection QFR (Primary Feature)
1. **Load two projections**: two DICOM cine loops from different viewing angles (≥25° separation)
2. **Calibrate each**: catheter-based pixel spacing (6F default)
3. **Segment each**: nnU-Net or AngioPy on a key frame per projection
4. **Dual viewer**: side-by-side synchronized playback
5. **TIMI frame count**: mark T-start and T-end frames for blood flow velocity
6. **3D reconstruction**: stereo triangulation from epipolar geometry
7. **Diameter measurement**: epipolar-matched 2D points → geometric mean √(d1×d2)
8. **QFR calculation**: Gould/Young-Tsai pressure model
9. **3D mesh**: vessel tube with QFR coloring (heatmap: red/yellow/green)
10. **Three modes**:
    - **fQFR** (Fixed): hyperemic velocity = 0.35 m/s (no patient-specific flow)
    - **cQFR** (Contrast-TIMI): resting velocity from TIMI → empirical hyperemic conversion
    - **aQFR** (Adenosine-TIMI): direct hyperemic velocity from TIMI

### 5.11 Vessel Tracking (CSRT)
- **Algorithm**: CSRT (Correlation Strength Response Tracking) + optical flow
- **ROI modes**: fixed 160×160 or adaptive
- **Initialize**: place ROI on frame, start tracker
- **Propagate**: forward and/or backward through frame range
- **Confidence threshold**: 0.0–1.0 (default 0.6)
- **Auto Seg+QCA**: optionally run segmentation+QCA on each tracked frame
- **Per-frame results**: Map of frame → (ROI, confidence, success)

### 5.12 3D Visualization
- **Engine**: Three.js via @react-three/fiber + drei
- **QFR Mesh Viewer**: triangle mesh with Phong shading
- **Interaction**: orbit (drag), zoom (scroll), pan (right-click)
- **Coloring**: optional QFR heatmap (red=severe stenosis, green=normal)
- **Auto-fit**: camera automatically frames the vessel
- **Lighting**: ambient + directional

### 5.13 Export & Reports
- **Formats**: CSV, XLSX, JSON, PDF, NPY (NumPy arrays)
- **QCA export**: diameter profiles, MLD, %DS
- **RWS export**: per-beat results with interpretation
- **Combined export**: all data (QCA + RWS + metadata)
- **PDF report**: clinical summary with graphs, ECG timeline, frame images
- **Enhanced PDF**: rich format with Recharts-rendered graphs embedded
- **Training dataset export**: frames + masks in ML training format
- **Dataset eligibility check**: validate data completeness before export
- **Anonymization**: HIPAA-compliant de-identification on export

### 5.14 Authentication (Stub)
- Token-based JWT (access + refresh tokens)
- Session ID tracking (X-Session-ID header)
- Auto token refresh on 401
- Register, login, logout, verify
- Currently optional / not enforced

### 5.15 Study Browser
- Route `/studies` with CompactLayout
- List/browse previously loaded studies
- Directory scanning for DICOM files

### 5.16 Keyboard Shortcuts
| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Arrow Left/Right | Previous/Next frame |
| Arrow Up/Down | Playback speed |
| B | Toggle Fixed ROI (160×160) |
| S | Toggle Seed point mode |
| H | Toggle Pan mode |
| C | Cycle annotation modes |
| R | Reset view (fit to window) |
| +/− | Zoom in/out |
| ? | Show shortcuts help |
| Ctrl+S | Save/Export |
| Ctrl+O | Open file |

### 5.17 Theme System
- **Modes**: light, dark, system (auto-detect OS preference)
- **Implementation**: Tailwind `dark` class on document root
- **Colors**: HSL-based CSS variables for semantic tokens
- **Vessel colors**: LAD=#ef4444, LCX=#22c55e, RCA=#3b82f6, LM=#f59e0b
- **Persisted**: to settingsStore (localStorage)

### 5.18 Settings
- Theme preference
- Default segmentation engine
- Tracking confidence threshold
- Default catheter size
- Mask edit tool preferences (brush size, hardness, etc.)
- Player defaults (speed, looping)
- Overlay visibility defaults
- Path history (last opened directory)

---

## 6. Feature Catalog — Backend

### 6.1 DICOM Loading
- Parse DICOM via pydicom with JPEG/JPEG2000 decompression
- Extract: patient info, study info, all frames as pixel arrays, ECG signal, metadata
- Multi-frame support (NumberOfFrames tag)
- Metadata extraction: angles (RAO/LAO, CRA/CAU), SID, SOD, pixel spacing, frame rate, manufacturer
- Frame encoding: numpy → PNG → base64 for transport
- Anonymization: HIPAA-compliant de-identification (patient name, ID, DOB, etc.)

### 6.2 Vessel Segmentation
- **nnU-Net**: ROI-based (crop + segment + uncrop), Dataset600 trained weights
- **nnU-Net Wide**: larger ROI variant
- **nnU-Net Fullframe**: no ROI required, full image input
- **AngioPy**: seed-guided U-Net + InceptionResNetV2 backbone, requires 2+ seed points
  - Model: `modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth`
- **Hybrid (roi+angiopy)**: nnU-Net fullframe → auto seed extraction → AngioPy refinement
- **SAM2, SegFormer, HRNet**: stubs/future engines
- All engines: lazy model loading, CPU/CUDA/MPS device selection
- Output: binary mask (0/255 np.ndarray), optional probability map, inference time

### 6.3 Centerline Extraction
- **Skeleton**: morphological thinning → longest path extraction
- **Distance transform**: ridge detection from distance map
- **MCP (Minimal Cost Path)**: shortest path between seed points through cost map
- **MCP Auto**: automatic seed point detection + MCP
- Output: Nx2 array of (y, x) coordinates, ordered proximal→distal

### 6.4 QCA Engine
- **Gaussian subpixel fitting**: 1D Gaussian profiles perpendicular to centerline
- **Bilinear interpolation**: sub-pixel accuracy in intensity sampling
- **50-point resampling**: standard N-point sampling along centerline
- **Reference diameter**: proximal and distal segments (typically 5mm from lesion)
- **%DS**: (1 − MLD/ReferenceD) × 100
- **Lesion length**: continuous stenosis extent in mm
- **measure_diameters_at_points()**: class method for measuring at specific 2D locations (used by QFR)

### 6.5 RWS Calculator
- **Input**: diameter arrays (MLD, proximal, distal) across frame range
- **Hampel filter**: MAD-based outlier detection with stenosis-aware windowing
  - Window size adapts to cardiac frequency
  - Threshold: 3.5 × MAD (standard Hampel)
- **Double Hampel**: 2-pass for extra robustness
- **IQR filter**: interquartile range outlier detection
- **Temporal filter**: consistency-based (max 20% frame-to-frame change)
- **Physiological bounds**: 0.2–5.5mm diameter (reject out-of-range)
- **Output**: RWSMultiPositionResult with mld_rws, proximal_rws, distal_rws, average_rws

### 6.6 QFR Calculator (3D, State-of-the-Art)
- **Gould/Young-Tsai pressure model**:
  - Viscous loss: ΔP_visc = (8μL) / (π r⁴) × Q
  - Turbulent loss: ΔP_turb = Kt × (ρ/2) × (A₀/As − 1)² × Q|Q| / A₀²
  - Kt = 1.52 (empirical turbulent coefficient)
- **Physical constants**:
  - Blood density: 1050 kg/m³
  - Blood viscosity: 0.0035 Pa·s (3.5 cP)
  - Aortic pressure: 100 mmHg
- **Flow estimation**:
  - Resting velocity from TIMI frame count: v = vessel_length / (tfc / frame_rate)
  - Hyperemic conversion: empirical CFR model per vessel type
  - LAD CTFC ÷1.7 correction (for corrected TIMI frame count)
  - Max resting velocity: 0.30 m/s, max hyperemic: 0.60 m/s
- **Three modes**: fQFR (fixed 0.35 m/s), cQFR (contrast-TIMI), aQFR (adenosine-TIMI)
- **Output**: QFR value (Pd/Pa), pressure drop, flow velocity, QFR and pressure profiles

### 6.7 Stereo Reconstructor
- **Epipolar geometry**: fundamental matrix from projection angles + SID/SOD
- **Camera matrix**: built from RAO/LAO, CRA/CAU angles + focal length (SOD/isocenter_ps)
- **Focal length formula**: fx = SOD / isocenter_pixel_spacing (NOT SID)
- **Triangulation**: linear least squares from matched 2D point pairs
- **Epipolar matching**: for each point on centerline1, find best match on centerline2 via epipolar constraint
- **ReconstructionResult**: centerline_3d (Nx3), matched 2D point pairs, confidence score
- **Angle validation**: ≥25° separation between projections required

### 6.8 Vessel Mesher
- **Input**: 3D centerline + diameter profile + optional QFR values
- **Output**: triangle mesh (vertices, faces, optional colors)
- **Method**: circular cross-section at each centerline point, 12 radial subdivisions
- **QFR coloring**: heatmap red→yellow→green for low→high QFR
- **Format**: compatible with Three.js BufferGeometry

### 6.9 ECG Processing
- **Parser**: reads DICOM WaveformSequence (0008,104C)
- **R-peak detection**: Pan-Tompkins algorithm (bandpass filter → derivative → squaring → moving average → threshold)
- **Siemens filter**: 50/60 Hz notch filter for vendor-specific noise
- **ECG Analyzer**: HRV, arrhythmia detection, QT interval (advanced features)
- **Manual editing**: set/add/remove/move R-peaks → recalculate beat boundaries

### 6.10 Motion Signal Engine
- **Algorithm**: Farneback dense optical flow
- **Process**: compute frame-to-frame optical flow magnitude → signal
- **Peak detection**: find cardiac phase events (systolic peaks) in motion signal
- **Beat boundaries**: midpoints between peaks
- **Manual editing**: set/add/remove/move peaks

### 6.11 Tracking Engine
- **CSRT**: OpenCV's discriminative correlation filter tracker
- **Template matching**: refinement step
- **Optical flow**: Lucas-Kanade for motion estimation
- **Propagation**: forward/backward across frame range with confidence tracking
- **ROI modes**: fixed 160×160 or adaptive

### 6.12 Mask Editor
- **Brush/eraser**: soft falloff painting
- **Smart brush**: gradient-aware edge following
- **Flood fill**: connected component filling
- **Morphological**: dilate, erode, fill_holes, remove_islands, smooth
- **Contour ops**: extract, deform (Gaussian influence), convert back to mask
- **Edge snap**: gradient-based attraction
- **Region grow**: seed-based expansion
- **Magic wand**: intensity selection
- **Interpolation**: between keyframes

### 6.13 Calibration
- **Catheter**: known_diameter_mm / measured_diameter_pixels
- **Catheter sizes**: 4F=1.33mm, 5F=1.67mm, 6F=2.00mm, 7F=2.33mm, 8F=2.67mm
- **From DICOM**: use pixel spacing from header
- **From mask**: automatic catheter width measurement from segmentation
- **Manual**: user-specified pixel spacing
- **Module-level state**: `_calibrated_pixel_spacing` overrides DICOM when set

### 6.14 Export
- CSV, XLSX, JSON, PDF, NPY formats
- QCA data, RWS results, combined exports
- PDF with clinical summary, graphs, ECG timeline
- Training dataset export (frames + masks for ML)
- Anonymization on export

### 6.15 Report Generation
- Coronary anatomy SVG report
- Clinical PDF with summary, anatomy, dominance, lesion descriptions
- Lazy-loaded (matplotlib optional — wrapped in try/except)

---

## 7. API Contract

### 7.1 Base Configuration
- **No `/api/v1` prefix** — routes registered directly (e.g., `/dicom/upload`)
- **Dev base**: `http://127.0.0.1:8000`
- **Prod base**: `/api` (reverse proxy)
- **Image transport**: base64-encoded PNG strings
- **Segmentation masks**: include `data:image/png;base64,` prefix
- **QFR routes**: return raw base64 WITHOUT prefix (frontend must add it)
- **Session**: `X-Session-ID` header

### 7.2 Endpoint Summary

#### DICOM (`/dicom/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/dicom/load` | Upload DICOM file |
| POST | `/dicom/load-path` | Load from server path |
| GET | `/dicom/metadata` | Study metadata |
| GET | `/dicom/frame/{frame_index}` | Single frame (base64 PNG) |
| GET | `/dicom/frames?start=&end=` | Frame range |
| GET | `/dicom/ecg` | ECG signal + R-peaks |
| GET | `/dicom/num_frames` | Frame count |
| POST | `/dicom/clear` | Clear session |
| POST | `/dicom/ecg/r-peaks` | Set all R-peaks |
| POST | `/dicom/ecg/r-peaks/add` | Add R-peak |
| POST | `/dicom/ecg/r-peaks/remove` | Remove R-peak |
| POST | `/dicom/ecg/r-peaks/move` | Move R-peak |
| POST | `/dicom/scan-directory` | Scan for DICOM files |

#### Segmentation (`/segmentation/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/segmentation/segment` | Segment frame |
| POST | `/segmentation/centerline` | Extract centerline |
| POST | `/segmentation/segment-and-extract` | Combined segment + centerline |
| GET | `/segmentation/engines` | Available engines |
| GET | `/segmentation/model-info` | Model metadata |

#### QCA (`/qca/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/qca/calculate` | Calculate QCA metrics |
| GET | `/qca/measurements/{session_id}/{frame_index}` | Get QCA |
| GET | `/qca/diameters/{session_id}/{frame_index}` | Get diameter profile |

#### RWS (`/rws/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/rws/calculate` | Calculate RWS for frame range |

#### QFR (`/qfr/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/qfr/calculate` | Single-view QFR |
| POST | `/qfr/projection/upload` | Upload projection DICOM |
| POST | `/qfr/projection/segment` | Segment projection |
| POST | `/qfr/projection/{id}/clear-segmentation` | Clear projection seg |
| GET | `/qfr/projection/frame/{frame_index}` | Get projection frame |
| GET | `/qfr/projection/ecg/{id}` | Projection ECG |
| POST | `/qfr/projection/calibrate` | Calibrate projection |
| POST | `/qfr/reconstruct` | 3D reconstruction + QFR |
| POST | `/qfr/thumbnail` | Generate preview |
| GET | `/qfr/thresholds` | Clinical thresholds |

#### Motion (`/motion/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/motion/calculate` | Calculate motion signal |
| POST | `/motion/detect-peaks` | Re-detect peaks |
| GET | `/motion/peaks` | Get peaks |
| POST | `/motion/peaks` | Set peaks |
| POST | `/motion/peaks/add` | Add peak |
| POST | `/motion/peaks/remove` | Remove peak |
| POST | `/motion/peaks/move` | Move peak |
| GET | `/motion/beat-boundaries` | Beat boundaries |
| GET | `/motion/signal` | Signal + peaks |

#### Calibration (`/calibration/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/calibration/catheter` | Catheter calibration |
| POST | `/calibration/manual` | Manual pixel spacing |
| POST | `/calibration/from-mask` | Auto from mask |

#### Tracking (`/tracking/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/tracking/initialize` | Initialize tracker |
| POST | `/tracking/track/{frame_index}` | Track frame |
| POST | `/tracking/propagate` | Propagate range |
| GET | `/tracking/state` | Tracker state |
| POST | `/tracking/reset` | Reset tracker |

#### Mask Edit (`/mask-edit/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/mask-edit/brush` | Draw/erase brush |
| POST | `/mask-edit/smart-brush` | Edge-aware brush |
| POST | `/mask-edit/flood-fill` | Fill region |
| POST | `/mask-edit/morphological` | Morph operations |
| POST | `/mask-edit/extract-contour` | Mask → contour |
| POST | `/mask-edit/deform-contour` | Drag contour |
| POST | `/mask-edit/contour-to-mask` | Contour → mask |
| POST | `/mask-edit/edge-snap` | Snap to edges |
| POST | `/mask-edit/region-grow` | Region growing |
| POST | `/mask-edit/magic-wand` | Intensity selection |
| POST | `/mask-edit/interpolate` | Temporal interpolation |

#### Export (`/export/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/export/to-csv` | Export CSV |
| POST | `/export/to-json` | Export JSON |
| POST | `/export/to-dicom` | Export DICOM |
| POST | `/export/anonymized` | Anonymized export |

#### Report (`/report/`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/report/generate` | Generate PDF report |

#### Health (`/health`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Service status |
| GET | `/version` | Version info |

---

## 8. Algorithms & Domain Logic

### 8.1 QCA — Gaussian Subpixel Diameter Fitting
1. At each centerline point, compute the normal direction (perpendicular to centerline tangent)
2. Sample intensity values along this normal line (both sides, bilinear interpolation)
3. Fit a 1D Gaussian to the intensity profile
4. Vessel boundary = where Gaussian drops below threshold (e.g., half-max)
5. Diameter = distance between boundaries (in pixels, convert to mm via calibration)
6. Repeat for all 50 resampled points along centerline

### 8.2 RWS — Radial Wall Strain
1. For each frame in a cardiac beat, compute QCA (diameter profile)
2. Extract MLD, proximal RD, distal RD per frame → time series
3. Apply outlier filtering (Hampel: median ± 3.5×MAD)
4. Find Dmax and Dmin in filtered series
5. RWS = (Dmax − Dmin) / Dmax × 100%
6. Compute for MLD, proximal, distal positions independently

### 8.3 QFR — Gould/Young-Tsai Model
1. **Input**: 3D centerline + diameter profile (mm) + flow velocity
2. **Discretize**: vessel into N segments, each with length ds and diameter d
3. **For each segment**:
   - Viscous: ΔP_v = (128 μ L Q) / (π d⁴)
   - Turbulent: ΔP_t = Kt × (ρ/2) × [(A₀/A_s) − 1]² × Q² / A₀²
   - Where Q = v × A_ref (flow rate), A = π(d/2)²
4. **Total pressure drop**: ΣΔP = Σ(ΔP_v + ΔP_t)
5. **QFR** = Pd / Pa = (Pa − ΣΔP) / Pa
6. **Flow velocity**: from TIMI frame count or fixed value

### 8.4 Stereo 3D Reconstruction
1. **Camera matrices**: from projection parameters
   - R = rotation from primary/secondary angles (RAO/LAO, CRA/CAU)
   - K = intrinsic matrix with fx = SOD / isocenter_pixel_spacing
   - P = K × [R | t]
2. **Fundamental matrix**: F = K⁻ᵀ × [t]× × R × K⁻¹
3. **Epipolar matching**: for each point on centerline1, compute epipolar line in view2, find nearest centerline2 point
4. **Triangulation**: for each matched pair, solve linear system (DLT) for 3D point
5. **Quality**: confidence = 1 − normalized_reprojection_error
6. **Requirement**: ≥25° angle separation between views

### 8.5 TIMI Frame Count → Flow Velocity
1. T-start = frame where contrast first enters vessel
2. T-end = frame where contrast reaches distal landmark
3. TFC = T-end − T-start (frames)
4. Transit time = TFC / frame_rate (seconds)
5. Resting velocity = vessel_length_mm / transit_time_ms × 1000 (m/s)
6. Hyperemic velocity = f(resting_velocity, vessel_type) — empirical model

### 8.6 Pan-Tompkins R-Peak Detection
1. Bandpass filter (5–15 Hz)
2. Derivative filter
3. Squaring (emphasize QRS)
4. Moving window integration
5. Adaptive thresholding (signal + noise level tracking)
6. Search-back for missed beats

### 8.7 Farneback Optical Flow Motion Signal
1. For consecutive frame pairs (f[i], f[i+1]):
   - Compute dense optical flow (Farneback method)
   - Calculate magnitude at each pixel
   - Sum or average magnitude → single motion value
2. Result: time series of motion values
3. Peaks correspond to maximum cardiac motion (systolic)

---

## 9. ML / AI Engines

### 9.1 nnU-Net v2
- **Architecture**: 2D/3D U-Net with automated hyperparameter tuning
- **Training**: Dataset600 (coronary angiography)
- **Input**: grayscale frame (with or without ROI crop)
- **Output**: binary segmentation mask
- **Variants**: standard ROI, wide ROI, fullframe
- **Device**: auto-select CUDA > MPS > CPU

### 9.2 AngioPy
- **Architecture**: U-Net encoder-decoder with InceptionResNetV2 backbone
- **Library**: segmentation-models-pytorch + timm
- **Input**: grayscale frame + 2–10 seed points (defining vessel path)
- **Output**: binary mask + probability map
- **Model**: `modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth`
- **Note**: seed points are converted to a distance map as additional input channel

### 9.3 SAM2 (Stub)
- Segment Anything Model 2 for interactive segmentation
- Point/box prompt-based
- Not yet integrated

### 9.4 SegFormer (Stub)
- Transformer-based semantic segmentation
- Not yet integrated

### 9.5 HRNet (Stub)
- High-Resolution Network for vessel classification
- Classifies vessel type: LAD, LCx, RCA, catheter, guide wire, etc.
- Not yet integrated

---

## 10. UI/UX Specification

### 10.1 Layout Structure

```
┌──────────────────────────────────────────────────────────┐
│ Header (12px): Title │ Version │ Status │ Fullscreen     │
├───┬──────────────────────────────────────────┬───────────┤
│ T │                                          │ Right     │
│ o │           Main Viewer                    │ Panel     │
│ o │     (DICOM canvas + overlay layers)      │ (280px)   │
│ l │                                          │           │
│ b │  ┌──────────────────────────────────┐    │ Tabs:     │
│ a │  │  Tabbed Charts (below viewer):   │    │ - Seg+QCA │
│ r │  │  ECG │ QCA │ RWS │ Motion        │    │ - RWS     │
│   │  └──────────────────────────────────┘    │ - QFR     │
│   │                                          │ - Report  │
│   ├──────────────────────────────────────────┤ - Export  │
│   │ Playback Controls (24px):                │ - Info    │
│   │ ⏮ ⏪ ▶ ⏩ ⏭ │ Frame slider │ Speed │   │           │
├───┴──────────────────────────────────────────┴───────────┤
│ Status Bar (6px): Connection │ Shortcuts hint            │
└──────────────────────────────────────────────────────────┘
```

### 10.2 QFR Mode Layout (replaces main viewer)

```
┌──────────────────────┬──────────────────────┐
│  Projection 1        │  Projection 2        │
│  (DICOM + overlay)   │  (DICOM + overlay)   │
│                      │                      │
│  Angle: RAO 30°     │  Angle: LAO 45°     │
│  Frame: 15/120      │  Frame: 22/150      │
├──────────────────────┴──────────────────────┤
│  Synchronized Playback Controls             │
└─────────────────────────────────────────────┘
```

Or 3D view mode:
```
┌─────────────────────────────────────────────┐
│          3D Vessel Mesh (Three.js)          │
│                                             │
│     QFR: 0.76  │  Length: 52.3mm           │
│     ΔP: 12.4 mmHg  │  v: 0.28 m/s        │
└─────────────────────────────────────────────┘
```

### 10.3 Canvas Layer System
Four stacked canvases (z-ordered):
1. **Video Layer** (bottom): DICOM frame rendering, windowing/leveling
2. **Segmentation Layer**: semi-transparent mask overlay + centerline
3. **Annotation Layer**: seed points, ROI boxes, measurements, labels
4. **Overlay Layer** (top): diameter markers, QCA heatmap, ECG/motion signals

Each layer updates independently (only dirty layers re-render).

### 10.4 Color Scheme

**Light Theme**:
- Background: #FFFFFF
- Text: #1F2937
- Surface: #F3F4F6
- Primary: #3B82F6 (blue)
- Success: #10B981 (green)
- Danger: #EF4444 (red)

**Dark Theme**:
- Background: #0F172A
- Text: #E5E7EB
- Surface: #1E293B
- Same accent colors

**Vessel Colors**: LAD=#ef4444, LCX=#22c55e, RCA=#3b82f6, LM=#f59e0b

### 10.5 Responsive Design
- Mobile (<640px): stacked, overlay panel, hamburger menu
- Tablet (640–1024px): flexible grid
- Desktop (>1024px): full three-column layout

### 10.6 Accessibility
- ARIA labels on buttons
- Keyboard navigation (all controls accessible)
- Tab order preserved
- High-contrast theme option
- Icon + text labels
- Semantic HTML

---

## 11. Configuration & Build

### 11.1 Frontend Build (Vite)
- Custom plugin: `cross-origin-isolation` injects COOP/COEP/CORP headers (required for SharedArrayBuffer / Cornerstone.js)
- `__API_BASE__` define: dev → `http://127.0.0.1:8000`, prod → `/api`
- Path alias: `@/*` → `src/*`
- Worker format: ES modules with WASM
- DICOM codec optimization: pre-bundled libjpeg-turbo, CharLS, OpenJPEG, OpenJPH
- Tauri detection: `TAURI_ENV_PLATFORM` environment variable
- Dev port: 1420 (strict)
- Watch ignores: `python-backend/**`, `src-tauri/**`

### 11.2 TypeScript
- Target: ES2020
- Module: ESNext
- Strict mode: enabled
- Module resolution: bundler

### 11.3 Tailwind CSS
- Dark mode: class-based
- HSL CSS variable color system
- Container: centered, max-width 1400px
- Custom animations: accordion open/close

### 11.4 Backend Build
- setuptools (pyproject.toml)
- Python 3.10–3.12
- Extras: `[dev]` (testing + linting), `[ml]` (PyTorch + nnU-Net)
- Black: line-length 100
- Ruff: E, F, W, I, N, UP, B, C4
- MyPy: strict, ignore missing imports

### 11.5 Tauri
- Window: 1400×900 (min 1200×800)
- CSP: null (relaxed for medical app)
- Shell plugin: can launch Python backend
- FS plugin: full scope
- Dialog plugin: file open/save
- Release: LTO, single codegen unit, panic=abort, opt-level=z

### 11.6 Docker
**Production** (docker-compose.yml):
- Frontend: Nginx container
- Backend: FastAPI container
- PostgreSQL 15 + Redis 7
- Health checks: 30s interval

**Development** (docker-compose.dev.yml):
- Vite dev server (port 5174)
- Backend with hot-reload
- DB/Redis on alternate ports (5433/6380)

### 11.7 Testing
- **Frontend unit**: Vitest, happy-dom/jsdom, coverage reporters (text, JSON, HTML)
- **Frontend E2E**: Playwright, Chromium, parallel execution
- **Backend**: pytest with asyncio mode=auto, coverage term-missing

---

## 12. Deployment

### 12.1 Desktop (Tauri)
1. `npm run tauri:build` → native binary (Linux/macOS/Windows)
2. Tauri shell plugin auto-launches `python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
3. Frontend served from bundled `dist/`

### 12.2 Docker
1. `docker-compose up` → 4 containers (nginx, backend, postgres, redis)
2. Nginx reverse-proxies `/api` → backend:8000
3. Frontend static files served by Nginx

### 12.3 Development
1. Terminal 1: `cd python-backend && source .venv/bin/activate && uvicorn app.main:app --reload`
2. Terminal 2: `npm run dev` (Vite on :1420)
3. Frontend calls backend directly at `http://127.0.0.1:8000`

---

## 13. Known Issues & Design Lessons

### 13.1 Architectural Issues to Address in Rewrite

1. **Module-level state**: Backend uses `_current_study` as module global. This is single-user, non-thread-safe, and makes testing difficult. **Recommendation**: use proper session management with a session store (Redis or in-memory with cleanup).

2. **Base64 image transport**: Every frame and mask is base64 PNG — massive overhead (~33% size increase). **Recommendation**: consider binary transport (ArrayBuffer) or streaming for frames.

3. **No WebSocket**: All communication is request/response. Frame streaming and progress updates require polling. **Recommendation**: add WebSocket for real-time events (segmentation progress, tracking progress, frame streaming).

4. **Massive api.ts file**: 2600 lines in a single file. **Recommendation**: split into modules matching backend route groups.

5. **QFR route inconsistency**: QFR routes return raw base64 without prefix while segmentation routes include prefix. **Recommendation**: standardize on one format.

6. **Session ID mismatch**: Frontend and backend generate session IDs independently. **Recommendation**: backend generates, frontend stores.

7. **No persistent storage**: Everything is in-memory. Closing the app loses all data. **Recommendation**: add SQLite or similar for session persistence.

8. **17 Zustand stores**: Some are tightly coupled. **Recommendation**: consider consolidating related stores.

### 13.2 Bugs That Were Fixed (Avoid in Rewrite)

1. **URL spaces in template literals**: Migration introduced spaces in API URLs. Use constant path strings.
2. **DICOM metadata access**: Study metadata is in `study.metadata['key']`, NOT `study.key`. Document clearly.
3. **Focal length uses SOD, not SID**: `fx = SOD / pixel_spacing`. This is a physics requirement.
4. **Pydantic defaults override DICOM values**: Use `Optional[T] = None` with priority chain.
5. **Lazy imports for optional dependencies**: Always wrap optional deps in try/except at import.
6. **Epipolar diameter alignment**: Never interpolate diameter profiles independently — always measure at matched 2D points.

### 13.3 Performance Considerations

1. **Frame caching**: Use sparse Map (only loaded frames), not dense array
2. **Canvas layers**: Only re-render dirty layers
3. **ML model loading**: Lazy on first use, keep loaded for session
4. **Frame preloading**: Preload adjacent frames during playback
5. **Retina display**: Handle pixel ratio for crisp rendering

---

*End of context document. This describes every feature of Coronary RWS Analyser v1.1 as of February 2026.*
