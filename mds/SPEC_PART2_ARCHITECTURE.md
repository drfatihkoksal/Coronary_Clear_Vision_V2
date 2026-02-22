# SPEC Part 2: Architecture & API Specification

> Definitive technical specification for Coronary RWS Analyser v2. This document is the authoritative reference for building the complete backend, frontend architecture, API layer, and algorithm implementations from scratch. Every interface, schema, and contract is specified to the level where a developer can implement without ambiguity.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Backend Architecture](#2-backend-architecture)
3. [Frontend Architecture](#3-frontend-architecture)
4. [Data Model](#4-data-model)
5. [API Specification](#5-api-specification)
6. [Algorithm Specifications](#6-algorithm-specifications)
7. [ML Engine Interface](#7-ml-engine-interface)
8. [Session & State Management](#8-session--state-management)
9. [Communication Protocol](#9-communication-protocol)
10. [Caching Strategy](#10-caching-strategy)
11. [Performance Budget](#11-performance-budget)
12. [Integration Points](#12-integration-points)

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
                          ┌─────────────────────────────────┐
                          │         Tauri Shell (v2)         │
                          │    ┌────────────────────────┐    │
                          │    │    Native Window       │    │
                          │    │    (WebView2/WKWebView) │    │
                          │    └─────────┬──────────────┘    │
                          │              │                    │
                          │    ┌─────────▼──────────────┐    │
                          │    │   Frontend (Vite SPA)   │    │
                          │    │   React + TypeScript    │    │
                          │    │   Port 1420 (dev)       │    │
                          │    └─────────┬──────────────┘    │
                          │              │                    │
                          │         HTTP REST + WebSocket     │
                          │              │                    │
                          │    ┌─────────▼──────────────┐    │
                          │    │   Backend (FastAPI)     │    │
                          │    │   Python 3.11+          │    │
                          │    │   Port 8000             │    │
                          │    │                         │    │
                          │    │   ┌──────────────────┐  │    │
                          │    │   │  ML Worker Pool   │  │    │
                          │    │   │  (ProcessPool)    │  │    │
                          │    │   └──────────────────┘  │    │
                          │    │                         │    │
                          │    │   ┌──────────────────┐  │    │
                          │    │   │  SQLite (local)   │  │    │
                          │    │   └──────────────────┘  │    │
                          │    └─────────────────────────┘    │
                          └─────────────────────────────────┘

Docker deployment (alternative):
  ┌──────────────┐   ┌─────────────────┐
  │ Nginx        │──▶│ Backend (8000)   │
  │ (80, static) │   │ + SQLite         │
  └──────────────┘   └─────────────────┘
```

### 1.2 Communication Patterns

| Channel | Transport | Direction | Purpose |
|---------|-----------|-----------|---------|
| REST API | HTTP/1.1 JSON | Request/Response | All CRUD, computation triggers, data retrieval |
| WebSocket | WS `/ws/{session_id}` | Bidirectional | Progress events, frame streaming, real-time status |
| Binary frames | HTTP `GET` with `Accept: application/octet-stream` | Response stream | Frame pixel data (raw uint8/uint16 PNG) |
| File upload | HTTP `POST` multipart/form-data | Request | DICOM file upload |

**v1 problem**: All image transport was base64 JSON, adding 33% overhead. All communication was request/response with no progress reporting.

**v2 solution**: Binary frame responses via `StreamingResponse`, WebSocket for long-running operation progress, standardized JSON for structured data.

### 1.3 Process Model

**Desktop (Tauri)**:
1. Tauri binary launches, creates native window
2. Shell plugin spawns: `python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
3. Frontend checks `GET /health` in a polling loop until backend responds
4. Backend initializes SQLite database at `~/.coronary-rws/sessions.db`
5. On window close: Tauri sends SIGTERM to Python process, waits 5s, then SIGKILL

**Web (Docker)**:
1. Nginx serves static frontend at `/`
2. Nginx reverse-proxies `/api/*` to backend container at port 8000
3. Backend uses same SQLite for persistence (mounted volume)

### 1.4 Key Architectural Decisions (Fixing v1 Problems)

**Problem 1: Module-level `_current_study` global variable**
- v1 used module-level globals in `dicom_routes.py` shared by all route modules
- No concurrency safety, no crash recovery, untestable

**v2 Solution: SessionService with typed session store**
- Backend generates session ID on DICOM upload, returns it to frontend
- All state lives in a `SessionStore` keyed by session ID
- FastAPI dependency injection provides session to route handlers
- Optional SQLite persistence for crash recovery
- See [Section 8](#8-session--state-management) for full design

**Problem 2: Base64 image transport (33% overhead)**
- v1 encoded every frame and mask as `data:image/png;base64,...` in JSON

**v2 Solution: Binary frame transport**
- Frame endpoints return `Content-Type: image/png` with raw PNG bytes
- Masks returned as PNG binary when fetched individually
- Batch operations return JSON with binary references (frame URLs), not inline data
- Segmentation results embed mask as data URI only when the mask is small (<50KB)

**Problem 3: No WebSocket for long-running operations**
- v1 had no progress reporting; UI appeared frozen during 2-30s operations

**v2 Solution: WebSocket event channel**
- Single WebSocket per session at `/ws/{session_id}`
- Server pushes progress events for segmentation, tracking, reconstruction, motion
- Client can send cancellation requests over the same socket
- See [Section 9](#9-communication-protocol) for event definitions

**Problem 4: Inconsistent base64 prefix handling**
- v1 segmentation routes included `data:image/png;base64,` prefix; QFR routes did not

**v2 Solution: Never include data URI prefix in API responses**
- All binary image data is returned as raw bytes with proper `Content-Type` headers
- When JSON must contain image data (e.g., batch responses), use raw base64 without prefix
- Frontend is responsible for constructing data URIs when needed

---

## 2. Backend Architecture

### 2.1 Layer Diagram

```
python-backend/
├── app/
│   ├── main.py                    # FastAPI app factory
│   ├── api/                       # HTTP layer (thin)
│   │   ├── __init__.py
│   │   ├── dependencies.py        # FastAPI Depends() providers
│   │   ├── middleware/
│   │   │   ├── error_handler.py   # Global exception → HTTP response
│   │   │   ├── logging.py         # Request/response logging
│   │   │   └── session.py         # Session ID extraction middleware
│   │   └── routes/
│   │       ├── dicom.py           # DICOM upload, frames, metadata
│   │       ├── segmentation.py    # Segment, centerline, engines
│   │       ├── qca.py             # QCA calculation
│   │       ├── rws.py             # RWS calculation
│   │       ├── qfr.py             # QFR dual-projection + 3D
│   │       ├── ecg.py             # ECG signal, R-peak editing
│   │       ├── motion.py          # Optical flow motion signal
│   │       ├── calibration.py     # Pixel spacing calibration
│   │       ├── tracking.py        # CSRT vessel tracking
│   │       ├── mask_edit.py       # Mask editing operations
│   │       ├── export.py          # Data export (CSV, JSON, PDF)
│   │       └── health.py          # Health checks
│   │
│   ├── core/                      # Domain logic (pure, no I/O)
│   │   ├── __init__.py
│   │   ├── qca_engine.py          # QCA diameter measurement
│   │   ├── rws_calculator.py      # RWS computation + outlier filters
│   │   ├── qfr_calculator.py      # QFR Gould/Young-Tsai model
│   │   ├── stereo_reconstructor.py # Epipolar 3D reconstruction
│   │   ├── centerline_extractor.py # Morphological centerline
│   │   ├── vessel_mesher.py       # 3D mesh from centerline
│   │   ├── ecg_analyzer.py        # Pan-Tompkins R-peak detection
│   │   ├── motion_analyzer.py     # Farneback optical flow
│   │   ├── calibration.py         # Calibration math
│   │   ├── mask_ops.py            # Morphological mask operations
│   │   ├── outlier_filter.py      # Hampel, IQR, temporal filters
│   │   └── anonymization.py       # HIPAA de-identification
│   │
│   ├── models/                    # Pydantic models
│   │   ├── __init__.py
│   │   ├── domain.py              # Internal domain types
│   │   ├── requests.py            # API request bodies
│   │   ├── responses.py           # API response bodies
│   │   └── enums.py               # Shared enumerations
│   │
│   ├── services/                  # Orchestration (combines core + infra)
│   │   ├── __init__.py
│   │   ├── study_service.py       # Load DICOM, manage study lifecycle
│   │   ├── segmentation_service.py # Coordinate engine + post-processing
│   │   ├── analysis_service.py    # QCA + RWS pipeline
│   │   ├── qfr_service.py         # QFR dual-projection pipeline
│   │   ├── tracking_service.py    # CSRT tracking orchestration
│   │   └── export_service.py      # Export generation
│   │
│   ├── infra/                     # External I/O
│   │   ├── __init__.py
│   │   ├── dicom_handler.py       # pydicom parsing
│   │   ├── ecg_parser.py          # DICOM ECG waveform extraction
│   │   ├── ml_engines/
│   │   │   ├── base.py            # Abstract engine interface
│   │   │   ├── registry.py        # Engine discovery + factory
│   │   │   ├── nnunet.py          # nnU-Net v2
│   │   │   └── angiopy.py         # AngioPy (U-Net + InceptionResNetV2)
│   │   ├── persistence/
│   │   │   ├── session_store.py   # In-memory + SQLite session store
│   │   │   └── db.py              # SQLite connection management
│   │   └── pdf_reporter.py        # PDF report generation (optional)
│   │
│   └── config/
│       ├── __init__.py
│       └── settings.py            # Pydantic BaseSettings
│
├── tests/
│   ├── unit/
│   │   ├── core/                  # Pure function tests (100% coverage target)
│   │   └── models/                # Pydantic validation tests
│   ├── integration/
│   │   ├── api/                   # HTTP endpoint tests
│   │   └── services/              # Service orchestration tests
│   └── conftest.py
│
├── pyproject.toml
└── requirements.txt
```

### 2.2 Layer Responsibilities and Contracts

#### API Layer (`app/api/`)
- **Responsibility**: HTTP concerns only. Request parsing, response formatting, status codes.
- **Rules**:
  - Route handlers are `async def`, max 30 lines
  - All business logic delegated to services
  - No direct imports from `core/` or `infra/` (only `services/` and `models/`)
  - Exception handling via global middleware, not per-route try/except
  - Dependency injection via `Depends()` for session, services

```python
# Example: api/routes/segmentation.py
from fastapi import APIRouter, Depends
from app.models.requests import SegmentRequest
from app.models.responses import SegmentResponse
from app.services.segmentation_service import SegmentationService
from app.api.dependencies import get_session, get_segmentation_service

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

@router.post("/segment", response_model=SegmentResponse)
async def segment_frame(
    request: SegmentRequest,
    session = Depends(get_session),
    service: SegmentationService = Depends(get_segmentation_service),
):
    result = await service.segment_frame(session, request)
    return SegmentResponse.from_domain(result)
```

#### Service Layer (`app/services/`)
- **Responsibility**: Orchestration. Coordinates core logic with infrastructure.
- **Rules**:
  - Receives domain models, returns domain models
  - May call multiple core functions and infra adapters
  - Manages transactions (session state updates)
  - Emits WebSocket events for progress

```python
# Example: services/segmentation_service.py
class SegmentationService:
    def __init__(self, engine_registry, session_store):
        self.engines = engine_registry
        self.sessions = session_store

    async def segment_frame(self, session, request):
        study = session.study
        frame = study.get_frame(request.frame_index)
        engine = self.engines.get(request.engine)
        result = await run_in_executor(engine.predict, frame.pixel_data, ...)
        if request.extract_centerline:
            centerline = centerline_extractor.extract(result.mask)
            ...
        session.store_segmentation(request.frame_index, result)
        return result
```

#### Core Layer (`app/core/`)
- **Responsibility**: Pure computation. No I/O, no HTTP, no database, no file system.
- **Rules**:
  - Functions take numpy arrays, floats, lists; return same
  - Only imports: `numpy`, `scipy`, `cv2` (for image processing), `dataclasses`
  - No logging side effects (return structured results instead)
  - Fully unit-testable with synthetic data

#### Infrastructure Layer (`app/infra/`)
- **Responsibility**: External world interaction. DICOM files, ML models, database, PDF.
- **Rules**:
  - Implements interfaces defined by services
  - Handles I/O errors with domain-specific exceptions
  - ML engines implement `BaseSegmentationEngine` abstract class
  - Lazy loading for heavy resources (ML models, optional deps)

### 2.3 Dependency Injection Strategy

```python
# app/api/dependencies.py
from functools import lru_cache
from fastapi import Depends, Header
from app.infra.persistence.session_store import SessionStore
from app.infra.ml_engines.registry import EngineRegistry

@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(db_path="~/.coronary-rws/sessions.db")

@lru_cache
def get_engine_registry() -> EngineRegistry:
    return EngineRegistry(models_dir=settings.ml_models_path)

async def get_session(
    x_session_id: str = Header(..., alias="X-Session-ID"),
    store: SessionStore = Depends(get_session_store),
):
    session = store.get(x_session_id)
    if session is None:
        raise HTTPException(404, detail={"code": "SESSION_NOT_FOUND"})
    return session

def get_segmentation_service(
    registry: EngineRegistry = Depends(get_engine_registry),
    store: SessionStore = Depends(get_session_store),
) -> SegmentationService:
    return SegmentationService(registry, store)
```

### 2.4 Module Boundary Rules

| From \ To | `api/` | `services/` | `core/` | `models/` | `infra/` | `config/` |
|-----------|--------|-------------|---------|-----------|----------|-----------|
| `api/` | -- | YES | NO | YES | NO | YES |
| `services/` | NO | -- | YES | YES | YES | YES |
| `core/` | NO | NO | -- | YES (domain only) | NO | NO |
| `models/` | NO | NO | NO | -- | NO | NO |
| `infra/` | NO | NO | YES (implements) | YES | -- | YES |
| `config/` | NO | NO | NO | NO | NO | -- |

### 2.5 Addressing "Is DDD Earning Its Complexity?"

**Answer: No. Drop the DDD label. Keep the good parts.**

v1 had DDD folder names (`domain/entities/`, `domain/services/`, `application/use_cases/`) but none of the actual DDD patterns (no bounded contexts, no domain events, no repositories, empty persistence layer). The "use cases" were one-liner wrappers around domain services. Routes bypassed use cases and contained 370-line inline business logic.

**v2 approach**: Pragmatic layered architecture.
- `core/` replaces `domain/services/`. Pure functions, no ceremony.
- `services/` replaces `application/use_cases/`. Real orchestration logic.
- `models/` replaces `domain/entities/` + `domain/value_objects/` + `presentation/schemas/`. All Pydantic models in one place.
- `infra/` stays. External I/O is properly isolated.
- No empty `events/`, `repositories/`, `persistence/models/` directories.
- No aggregate roots, no Unit of Work, no specification pattern.

The architecture earns its complexity only where it provides testability (core/ has no I/O) and replaceability (infra/ engines can be swapped). Everything else is flat and direct.

---

## 3. Frontend Architecture

### 3.1 Component Tree

```
App.tsx (Router)
├── / → StudyBrowser
│   └── CompactLayout
│       ├── StudyList
│       └── DirectoryScanner
│
└── /app → AnalysisApp
    ├── Header
    │   ├── Logo + Version
    │   ├── ConnectionStatus
    │   └── ThemeToggle + Fullscreen
    │
    ├── Toolbar (left, vertical)
    │   ├── ToolButton (Seed, ROI, Pan, Zoom, MaskEdit)
    │   └── EngineSelector
    │
    ├── MainArea (center)
    │   ├── ViewerContainer
    │   │   ├── StandardViewer (single DICOM view)
    │   │   │   └── CanvasLayers [4 stacked canvases]
    │   │   │       ├── VideoLayer (z:0) - DICOM frame rendering
    │   │   │       ├── SegmentationLayer (z:1) - Mask overlay
    │   │   │       ├── AnnotationLayer (z:2) - Seeds, ROI, labels
    │   │   │       └── OverlayLayer (z:3) - QCA markers, ECG
    │   │   │
    │   │   ├── QFRDualViewer (side-by-side projections)
    │   │   │   ├── ProjectionViewer[1] → CanvasLayers
    │   │   │   └── ProjectionViewer[2] → CanvasLayers
    │   │   │
    │   │   └── QFRMesh3DViewer (Three.js)
    │   │       ├── VesselMesh (BufferGeometry)
    │   │       ├── OrbitControls
    │   │       └── QFRLegend
    │   │
    │   ├── ChartTabs (below viewer)
    │   │   ├── ECGChart (Recharts)
    │   │   ├── QCAChart (Recharts)
    │   │   ├── RWSChart (Recharts)
    │   │   └── MotionChart (Recharts)
    │   │
    │   └── PlaybackControls
    │       ├── FrameSlider
    │       ├── PlayPauseButton
    │       ├── SpeedControl
    │       └── FrameCounter
    │
    └── RightPanel (280px, tabbed)
        ├── SegmentationPanel
        │   ├── EngineConfig
        │   ├── SeedPointList
        │   ├── ROIControls
        │   └── ActionButtons (Segment, Extract, Combined)
        ├── QCAPanel
        ├── RWSPanel
        │   ├── BeatSelector
        │   ├── OutlierMethodSelector
        │   ├── ResultsList
        │   └── VesselAnnotation
        ├── QFRPanel
        │   ├── ProjectionUpload[1,2]
        │   ├── CalibrationControl[1,2]
        │   ├── TIMIFrameCount
        │   ├── QFRModeSelector (fQFR/cQFR/aQFR)
        │   └── ReconstructButton
        ├── CalibrationPanel
        ├── TrackingPanel
        ├── ExportPanel
        ├── ReportPanel
        └── MetadataPanel
```

### 3.2 State Management Redesign

**v1 problem**: 17 independent Zustand stores with hidden `getState()` cross-references. `dicomStore` directly called `playerStore.getState().setTotalFrames()`. No dependency graph documentation.

**v2 approach**: Consolidate into 7 stores organized by lifecycle, not feature. Use an event bus for cross-store communication.

```
src/stores/
├── sessionStore.ts       # Session ID, connection status, backend health
├── studyStore.ts         # DICOM metadata, frames cache, frame loading
├── playerStore.ts        # Playback state, current frame, speed, loop
├── analysisStore.ts      # Segmentation, QCA, RWS results (per-frame Maps)
├── qfrStore.ts           # QFR mode: projections, calibration, 3D results
├── toolStore.ts          # Active tool, brush settings, ROI, seeds, overlays
└── settingsStore.ts      # Persisted: theme, defaults, preferences (localStorage)
```

**Cross-store communication via event bus:**

```typescript
// src/lib/eventBus.ts
type Events = {
  'study:loaded': { sessionId: string; metadata: StudyMetadata };
  'study:cleared': void;
  'frame:changed': { frameIndex: number };
  'segmentation:completed': { frameIndex: number };
  'calibration:changed': { pixelSpacing: number };
  'beat:selected': { startFrame: number; endFrame: number };
};

// In studyStore:
eventBus.emit('study:loaded', { sessionId, metadata });

// In playerStore (subscriber):
eventBus.on('study:loaded', ({ metadata }) => {
  set({ totalFrames: metadata.numFrames, frameRate: metadata.frameRate });
});
```

This makes cross-store dependencies explicit, traceable, and testable. No more `getState()` calls across store boundaries.

### 3.3 Data Fetching Strategy

Use TanStack React Query for all API calls. Remove the 2600-line `api.ts` monolith.

```
src/lib/api/
├── client.ts             # Axios instance, interceptors, headers
├── queryKeys.ts          # Centralized query key factory
├── dicom.ts              # DICOM queries and mutations
├── segmentation.ts       # Segmentation queries and mutations
├── qca.ts                # QCA queries and mutations
├── rws.ts                # RWS queries and mutations
├── qfr.ts                # QFR queries and mutations
├── ecg.ts                # ECG queries and mutations
├── motion.ts             # Motion queries and mutations
├── calibration.ts        # Calibration mutations
├── tracking.ts           # Tracking queries and mutations
├── mask-edit.ts          # Mask editing mutations
└── export.ts             # Export mutations
```

```typescript
// src/lib/api/queryKeys.ts
export const queryKeys = {
  frame: (sessionId: string, index: number) => ['frame', sessionId, index],
  metadata: (sessionId: string) => ['metadata', sessionId],
  segmentation: (sessionId: string, frame: number) => ['segmentation', sessionId, frame],
  qca: (sessionId: string, frame: number) => ['qca', sessionId, frame],
  engines: () => ['engines'],
};
```

Frame fetching uses React Query with binary response handling:

```typescript
// src/lib/api/dicom.ts
export function useFrame(sessionId: string, frameIndex: number) {
  return useQuery({
    queryKey: queryKeys.frame(sessionId, frameIndex),
    queryFn: () => client.get(`/dicom/frame/${frameIndex}`, {
      responseType: 'arraybuffer',
      headers: { 'X-Session-ID': sessionId },
    }).then(res => createImageBitmap(new Blob([res.data], { type: 'image/png' }))),
    staleTime: Infinity,  // Frames don't change
    gcTime: 5 * 60 * 1000,  // Keep for 5 minutes after unmount
  });
}
```

### 3.4 Canvas Layer Architecture

Four stacked HTML5 Canvas elements, each updating independently:

| Layer | Z-Index | Content | Update Trigger |
|-------|---------|---------|---------------|
| Video | 0 | DICOM frame pixels | Frame change |
| Segmentation | 1 | Mask overlay (semi-transparent) | Segmentation result |
| Annotation | 2 | Seeds, ROI box, measurements | User interaction |
| Overlay | 3 | QCA markers, ECG trace, motion signal | Analysis results |

```typescript
// src/hooks/useCanvasLayers.ts
interface CanvasLayerConfig {
  id: string;
  zIndex: number;
  render: (ctx: CanvasRenderingContext2D, state: LayerState) => void;
  shouldUpdate: (prev: LayerState, next: LayerState) => boolean;
}
```

Each layer has its own `requestAnimationFrame` loop. The `shouldUpdate` function prevents unnecessary repaints. The video layer uses `ImageBitmap` (from binary frame response) for zero-copy rendering via `drawImage()`.

### 3.5 Worker Strategy

```
src/workers/
├── dicomDecoder.worker.ts    # DICOM frame decoding (Cornerstone.js)
└── framePreloader.worker.ts  # Background frame fetching + caching
```

**DICOM Decoder Worker**: Used only for local Tauri mode where DICOM files are loaded directly from disk via Cornerstone.js. In web mode, the backend handles all DICOM decoding.

**Frame Preloader Worker**: During playback, prefetches N frames ahead. Uses `MessageChannel` to communicate with the main thread's frame cache without blocking the UI.

---

## 4. Data Model

### 4.1 Core Entities

#### Study (Backend: Python dataclass / Pydantic, Frontend: TypeScript interface)

```python
# Python (app/models/domain.py)
class Study(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    patient: PatientInfo
    study_info: StudyInfo
    frames: list[Frame] = []
    frame_rate: float = Field(gt=0, default=15.0)
    image_width: int = Field(gt=0, default=512)
    image_height: int = Field(gt=0, default=512)
    pixel_spacing: PixelSpacing | None = None
    metadata: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PatientInfo(BaseModel):
    patient_id: str | None = None
    name: str | None = None
    birth_date: str | None = None
    sex: Literal["M", "F", "O"] | None = None
    age: int | None = Field(None, ge=0, le=150)

class StudyInfo(BaseModel):
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    study_date: str | None = None
    study_time: str | None = None
    description: str | None = None
    institution: str | None = None
    modality: str = "XA"

class PixelSpacing(BaseModel):
    row_spacing: float = Field(gt=0, description="mm per pixel, row direction")
    col_spacing: float = Field(gt=0, description="mm per pixel, column direction")
    source: Literal["dicom", "catheter", "manual", "from_mask"] = "dicom"
    confidence: float = Field(ge=0, le=1, default=1.0)

class Frame(BaseModel):
    index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    timestamp_ms: float = 0.0
    # pixel_data: np.ndarray is NOT serialized; kept in memory only
```

```typescript
// TypeScript (src/types/study.ts)
interface Study {
  id: string;
  patient: PatientInfo;
  studyInfo: StudyInfo;
  frameRate: number;
  numFrames: number;
  imageWidth: number;
  imageHeight: number;
  pixelSpacing: PixelSpacing | null;
  metadata: StudyMetadata;
}

interface PatientInfo {
  patientId: string | null;
  name: string | null;
  birthDate: string | null;
  sex: 'M' | 'F' | 'O' | null;
  age: number | null;
}

interface StudyInfo {
  studyInstanceUid: string | null;
  seriesInstanceUid: string | null;
  studyDate: string | null;
  studyTime: string | null;
  description: string | null;
  institution: string | null;
  modality: string;
}

interface PixelSpacing {
  rowSpacing: number;
  colSpacing: number;
  source: 'dicom' | 'catheter' | 'manual' | 'from_mask';
  confidence: number;
}

interface StudyMetadata {
  positionerPrimaryAngle: number | null;
  positionerSecondaryAngle: number | null;
  distanceSourceToDetector: number | null;  // SID, mm
  distanceSourceToPatient: number | null;   // SOD, mm
  rawPixelSpacing: [number, number] | null;
  rawImagerPixelSpacing: [number, number] | null;
  intensifierSize: number | null;           // mm
  frameTime: number | null;                 // ms
  manufacturer: string | null;
  bitsStored: number | null;
}
```

#### QCA Measurement

```python
class QCAMeasurement(BaseModel):
    frame_index: int = Field(ge=0)
    diameter_profile_mm: list[float]     # N-point (default 50)
    diameter_profile_px: list[float]     # N-point in pixels
    centerline_points: list[tuple[float, float]]  # (y, x) coords
    distances_mm: list[float]            # Arc-length positions
    mld_mm: float = Field(ge=0)
    mld_index: int = Field(ge=0)
    mld_position: tuple[float, float]
    proximal_rd_mm: float = Field(ge=0)
    proximal_rd_index: int = Field(ge=0)
    distal_rd_mm: float = Field(ge=0)
    distal_rd_index: int = Field(ge=0)
    interpolated_rd_mm: float = Field(ge=0)
    diameter_stenosis_pct: float = Field(ge=0, le=100)
    lesion_length_mm: float | None = None
    pixel_spacing_mm: float = Field(gt=0)
    num_points: int = Field(gt=0, default=50)
    method: Literal["gaussian", "parabolic", "threshold"] = "gaussian"
```

#### RWS Result

```python
class RWSResult(BaseModel):
    beat_number: int = Field(ge=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    mld_rws_pct: float = Field(ge=0)
    proximal_rws_pct: float = Field(ge=0)
    distal_rws_pct: float = Field(ge=0)
    average_rws_pct: float = Field(ge=0)
    interpretation: Literal["normal", "intermediate", "vulnerable", "high_risk"]
    outlier_method: Literal["none", "hampel", "double_hampel"] = "hampel"
    vessel: str | None = None
    quality_score: float = Field(ge=0, le=1, default=1.0)
    num_frames_used: int = Field(ge=2)

    @computed_field
    @property
    def interpretation(self) -> str:
        avg = self.average_rws_pct
        if avg < 8.0: return "normal"
        if avg < 12.0: return "intermediate"
        if avg < 14.0: return "vulnerable"
        return "high_risk"
```

#### QFR Result

```python
class QFR3DResult(BaseModel):
    qfr_value: float = Field(ge=0, le=1)
    pressure_drop_mmhg: float = Field(ge=0)
    flow_velocity_m_s: float = Field(gt=0)
    timi_frame_count: float = Field(gt=0)
    vessel_length_mm: float = Field(gt=0)
    min_diameter_mm: float = Field(gt=0)
    mean_diameter_mm: float = Field(gt=0)
    stenosis_pct: float = Field(ge=0, le=100)
    mode: Literal["fqfr", "cqfr", "aqfr"]
    vessel_type: Literal["LAD", "LCx", "RCA", "other"]
    resting_velocity_m_s: float = Field(ge=0)
    hyperemic_velocity_m_s: float = Field(ge=0)
    qfr_profile: list[float]
    pressure_profile_mmhg: list[float]
    interpretation: Literal["negative", "grey_zone", "positive"]

    @computed_field
    @property
    def interpretation(self) -> str:
        if self.qfr_value > 0.80: return "negative"
        if self.qfr_value >= 0.75: return "grey_zone"
        return "positive"
```

#### ECG Signal

```python
class ECGSignal(BaseModel):
    samples: list[float]       # Voltage values
    sample_rate: float = Field(gt=0, description="Hz")
    frame_rate: float = Field(gt=0, description="fps")
    r_peaks: list[RPeak] = []
    units: str = "mV"
    duration_seconds: float = Field(gt=0)
    heart_rate_bpm: float | None = None

class RPeak(BaseModel):
    sample_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    amplitude: float
    confidence: float = Field(ge=0, le=1, default=1.0)
```

#### QFR Session (Dual Projection)

```python
class QFRSession(BaseModel):
    session_id: str
    projection1: ProjectionData = Field(default_factory=ProjectionData)
    projection2: ProjectionData = Field(default_factory=ProjectionData)
    centerline_3d: list[list[float]] | None = None  # Nx3
    diameter_3d: list[float] | None = None
    mesh: VesselMesh | None = None
    qfr_result: QFR3DResult | None = None

class ProjectionData(BaseModel):
    loaded: bool = False
    series_uid: str | None = None
    num_frames: int = 0
    frame_rate: float = 15.0
    primary_angle: float | None = None
    secondary_angle: float | None = None
    pixel_spacing: tuple[float, float] | None = None
    sid: float | None = None
    sod: float | None = None
    calibration_pixel_spacing: float | None = None
    catheter_size_fr: int | None = None
    t_start: int | None = None
    t_end: int | None = None
    has_segmentation: bool = False
    has_centerline: bool = False

class VesselMesh(BaseModel):
    vertices: list[list[float]]  # Nx3
    faces: list[list[int]]       # Mx3
    colors: list[list[float]] | None = None  # Nx3 RGB
```

### 4.2 Relationships

```
Study 1──N Frame
Study 1──N Segmentation (per frame, per engine)
Study 1──N QCAMeasurement (per frame)
Study 0..1──1 ECGSignal
Study 0..1──1 PixelSpacing

QFRSession 1──2 ProjectionData
QFRSession 0..1──1 QFR3DResult
QFRSession 0..1──1 VesselMesh

RWSResult N──1 BeatRange
BeatRange derives from ECGSignal.r_peaks
```

### 4.3 SQLite Schema (Session Persistence)

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    study_metadata JSON,        -- StudyInfo + PatientInfo as JSON
    frame_count INTEGER,
    frame_rate REAL,
    image_width INTEGER,
    image_height INTEGER,
    pixel_spacing JSON,
    dicom_metadata JSON,
    file_path TEXT               -- Original DICOM path for reload
);

CREATE TABLE segmentations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    engine TEXT NOT NULL,
    mask_blob BLOB,              -- PNG-compressed binary mask
    centerline JSON,             -- [[y,x], ...] points
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, frame_index, engine)
);

CREATE TABLE qca_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    diameter_profile JSON,       -- [float, ...]
    mld_mm REAL,
    proximal_rd_mm REAL,
    distal_rd_mm REAL,
    diameter_stenosis_pct REAL,
    pixel_spacing_mm REAL,
    UNIQUE(session_id, frame_index)
);

CREATE TABLE rws_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    beat_number INTEGER,
    start_frame INTEGER,
    end_frame INTEGER,
    mld_rws_pct REAL,
    proximal_rws_pct REAL,
    distal_rws_pct REAL,
    average_rws_pct REAL,
    outlier_method TEXT,
    vessel TEXT
);

CREATE TABLE ecg_data (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    sample_rate REAL,
    r_peak_frames JSON,          -- [int, ...]
    source TEXT                   -- 'dicom', 'motion', 'manual'
);

CREATE TABLE calibrations (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    pixel_spacing_mm REAL,
    source TEXT,
    catheter_size_fr INTEGER
);
```

---

## 5. API Specification

### 5.1 Base Configuration

| Property | Value |
|----------|-------|
| Base URL (dev) | `http://127.0.0.1:8000` |
| Base URL (prod/Docker) | `/api` (Nginx proxy) |
| API prefix | None (routes define their own, e.g. `/dicom`, `/segmentation`) |
| Content-Type (requests) | `application/json` (or `multipart/form-data` for uploads) |
| Content-Type (responses) | `application/json` (structured), `image/png` (binary frames/masks) |
| Session header | `X-Session-ID: <uuid>` (required on all endpoints except health and DICOM upload) |
| Error format | `{"error": {"code": "ERROR_CODE", "message": "Human-readable", "details": {}}}` |

### 5.2 Standard Error Response

```json
{
  "error": {
    "code": "STUDY_NOT_LOADED",
    "message": "No DICOM study is loaded for this session",
    "details": {
      "session_id": "abc-123"
    }
  }
}
```

| HTTP Status | Error Code Pattern | When |
|-------------|-------------------|------|
| 400 | `VALIDATION_*` | Malformed request body |
| 404 | `*_NOT_FOUND` | Session, frame, or resource not found |
| 409 | `CONFLICT` | Concurrent operation in progress |
| 422 | Domain error codes | Business logic failure |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `ENGINE_UNAVAILABLE` | ML model not loaded/available |

### 5.3 DICOM Management

#### POST /dicom/upload
Upload and parse a DICOM angiography file. Creates a new session.

**Request**: `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | DICOM file (.dcm) |
| `anonymize` | bool | No (default: false) | Strip HIPAA identifiers |
| `extract_ecg` | bool | No (default: true) | Extract embedded ECG |

**Response** (200): `application/json`
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "patient": { "patientId": "...", "name": "...", "birthDate": "...", "sex": "M", "age": 65 },
  "studyInfo": { "studyInstanceUid": "1.2.3...", "studyDate": "20240115", "modality": "XA", ... },
  "numFrames": 120,
  "frameRate": 15.0,
  "imageWidth": 512,
  "imageHeight": 512,
  "pixelSpacing": { "rowSpacing": 0.308, "colSpacing": 0.308, "source": "dicom" },
  "metadata": {
    "positionerPrimaryAngle": 30.0,
    "positionerSecondaryAngle": -15.0,
    "distanceSourceToDetector": 1050.0,
    "distanceSourceToPatient": 750.0,
    "manufacturer": "SIEMENS"
  },
  "ecg": {
    "available": true,
    "sampleRate": 1000.0,
    "durationSeconds": 8.0,
    "rPeaks": [{"sampleIndex": 500, "frameIndex": 7, "amplitude": 1.2}, ...],
    "heartRateBpm": 72.0
  }
}
```

**Errors**: 400 (not a valid DICOM), 422 (not angiography / no frames)

#### POST /dicom/load-path
Load DICOM from a server-side file path (Tauri desktop mode).

**Request**: `application/json`
```json
{ "file_path": "/path/to/study.dcm", "anonymize": false, "extract_ecg": true }
```

**Response**: Same as POST /dicom/upload

#### GET /dicom/frame/{frame_index}
Get a single frame as binary PNG.

**Headers**: `X-Session-ID` required

**Response** (200): `Content-Type: image/png`, raw PNG bytes

**Response** (200, if `Accept: application/json`): JSON with base64
```json
{ "data": "<base64-png>", "width": 512, "height": 512, "index": 42 }
```

**Errors**: 404 (session or frame not found)

#### GET /dicom/metadata
Get study metadata for the current session.

**Response** (200): Same structure as the upload response (minus `session_id`)

#### GET /dicom/ecg
Get ECG signal data.

**Response** (200):
```json
{
  "available": true,
  "samples": [0.1, 0.15, ...],
  "sampleRate": 1000.0,
  "frameRate": 15.0,
  "rPeaks": [{"sampleIndex": 500, "frameIndex": 7, "amplitude": 1.2}, ...],
  "heartRateBpm": 72.0,
  "beatBoundaries": [{"startFrame": 0, "endFrame": 14, "beatNumber": 1}, ...]
}
```

#### POST /dicom/ecg/r-peaks
Set all R-peaks (manual override).

**Request**:
```json
{ "frame_indices": [7, 22, 37, 52, 67] }
```

**Response** (200):
```json
{ "rPeaks": [...], "heartRateBpm": 75.0, "beatBoundaries": [...] }
```

#### POST /dicom/ecg/r-peaks/add
Add a single R-peak.

**Request**: `{ "frame_index": 45 }`

#### POST /dicom/ecg/r-peaks/remove
Remove a single R-peak.

**Request**: `{ "frame_index": 45 }`

#### POST /dicom/ecg/r-peaks/move
Move an R-peak from one frame to another.

**Request**: `{ "from_frame": 45, "to_frame": 47 }`

#### GET /dicom/num_frames
**Response**: `{ "num_frames": 120 }`

#### POST /dicom/clear
Clear the current session (free memory).

**Response**: `{ "success": true }`

#### POST /dicom/scan-directory
Scan a directory for DICOM files.

**Request**: `{ "directory": "/path/to/dicoms", "recursive": true }`

**Response**:
```json
{
  "files": [
    { "path": "/path/to/file.dcm", "patient": "John Doe", "modality": "XA", "numFrames": 120 }
  ]
}
```

### 5.4 Segmentation

#### POST /segmentation/segment
Segment a single frame.

**Request**:
```json
{
  "frame_index": 42,
  "engine": "nnunet",
  "roi": { "x": 150, "y": 120, "width": 160, "height": 160 },
  "seed_points": [{"x": 200, "y": 180}, {"x": 250, "y": 220}]
}
```

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `frame_index` | int | Yes | >= 0 |
| `engine` | string | Yes | One of: `nnunet`, `nnunet-wide`, `nnunet-fullframe`, `angiopy`, `roi+angiopy` |
| `roi` | object | No | Required for `nnunet`, `nnunet-wide` |
| `seed_points` | array | No | Required for `angiopy` (2-10 points) |

**Response** (200): `Content-Type: image/png` (binary mask)

**Response** (200, if `Accept: application/json`):
```json
{
  "mask": "<base64-png-no-prefix>",
  "confidence": 0.95,
  "width": 512,
  "height": 512,
  "engine": "nnunet",
  "inference_time_ms": 450
}
```

#### POST /segmentation/centerline
Extract centerline from an existing segmentation mask.

**Request**:
```json
{
  "frame_index": 42,
  "method": "mcp_auto"
}
```

**Response** (200):
```json
{
  "centerline": [{"x": 200.5, "y": 180.3}, {"x": 201.2, "y": 181.1}, ...],
  "num_points": 85,
  "method": "mcp_auto"
}
```

#### POST /segmentation/segment-and-extract
Combined: segment + extract centerline + (optional) QCA.

**Request**: Same as `/segment` plus:
```json
{
  "extract_centerline": true,
  "centerline_method": "mcp_auto",
  "calculate_qca": false
}
```

**Response** (200):
```json
{
  "mask": "<base64-png>",
  "centerline": [{"x": 200.5, "y": 180.3}, ...],
  "seed_points": [{"x": 200, "y": 180}, ...],
  "confidence": 0.95,
  "engine": "nnunet"
}
```

#### GET /segmentation/engines
List available segmentation engines.

**Response** (200):
```json
{
  "engines": [
    { "id": "nnunet", "name": "nnU-Net", "available": true, "requires_roi": true, "requires_seeds": false },
    { "id": "angiopy", "name": "AngioPy", "available": true, "requires_roi": false, "requires_seeds": true },
    { "id": "nnunet-fullframe", "name": "nnU-Net (Full Frame)", "available": true, "requires_roi": false, "requires_seeds": false }
  ]
}
```

### 5.5 QCA

#### POST /qca/calculate

**Request**:
```json
{
  "frame_index": 42,
  "pixel_spacing_mm": 0.308,
  "num_points": 50,
  "method": "gaussian"
}
```

**Response** (200):
```json
{
  "frame_index": 42,
  "diameter_profile_mm": [2.1, 2.0, 1.8, ...],
  "diameter_profile_px": [6.8, 6.5, 5.8, ...],
  "centerline_points": [{"x": 200.5, "y": 180.3}, ...],
  "distances_mm": [0.0, 0.31, 0.62, ...],
  "mld_mm": 1.2,
  "mld_index": 28,
  "proximal_rd_mm": 2.8,
  "distal_rd_mm": 2.5,
  "interpolated_rd_mm": 2.65,
  "diameter_stenosis_pct": 54.7,
  "lesion_length_mm": 8.2,
  "num_points": 50,
  "method": "gaussian"
}
```

### 5.6 RWS

#### POST /rws/calculate

**Request**:
```json
{
  "start_frame": 7,
  "end_frame": 22,
  "beat_number": 1,
  "outlier_method": "hampel",
  "pixel_spacing_mm": 0.308,
  "vessel": "LAD"
}
```

**Response** (200):
```json
{
  "beat_number": 1,
  "start_frame": 7,
  "end_frame": 22,
  "mld_rws_pct": 11.3,
  "proximal_rws_pct": 5.2,
  "distal_rws_pct": 7.8,
  "average_rws_pct": 8.1,
  "interpretation": "intermediate",
  "outlier_method": "hampel",
  "vessel": "LAD",
  "quality_score": 0.92,
  "num_frames_used": 14,
  "per_frame_diameters": {
    "mld": [1.2, 1.15, 1.1, ...],
    "proximal": [2.8, 2.75, 2.7, ...],
    "distal": [2.5, 2.45, 2.4, ...],
    "frame_indices": [7, 8, 9, ...]
  }
}
```

### 5.7 QFR (Dual-Projection)

#### POST /qfr/projection/upload
Upload a DICOM file as a QFR projection.

**Request**: `multipart/form-data`
| Field | Type | Required |
|-------|------|----------|
| `file` | File | Yes |
| `projection_id` | int (1 or 2) | Yes |

**Response** (200):
```json
{
  "projection_id": 1,
  "num_frames": 120,
  "frame_rate": 15.0,
  "primary_angle": 30.0,
  "secondary_angle": -15.0,
  "sid": 1050.0,
  "sod": 750.0,
  "pixel_spacing": [0.308, 0.308]
}
```

#### POST /qfr/projection/calibrate
Calibrate a projection with catheter diameter.

**Request**:
```json
{
  "projection_id": 1,
  "catheter_size_fr": 6,
  "measured_diameter_px": 6.5
}
```

**Response** (200):
```json
{ "pixel_spacing_mm": 0.308, "catheter_size_fr": 6 }
```

#### POST /qfr/projection/segment
Segment a frame in a QFR projection.

**Request**:
```json
{
  "projection_id": 1,
  "frame_index": 15,
  "engine": "nnunet-fullframe",
  "seed_points": [{"x": 200, "y": 180}, {"x": 250, "y": 220}]
}
```

**Response** (200):
```json
{
  "mask": "<base64-png>",
  "centerline": [{"x": 200.5, "y": 180.3}, ...],
  "diameter_profile_mm": [2.1, 2.0, 1.8, ...]
}
```

#### GET /qfr/projection/frame/{frame_index}?projection_id=1
Get a QFR projection frame as binary PNG.

**Response** (200): `Content-Type: image/png`

#### POST /qfr/reconstruct
Perform 3D reconstruction and QFR calculation.

**Request**:
```json
{
  "mode": "cqfr",
  "vessel_type": "LAD",
  "frame_rate": 15.0,
  "timi_frame_count": null
}
```

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `mode` | string | No | `"cqfr"` |
| `vessel_type` | string | No | `"other"` |
| `frame_rate` | float | No | Use DICOM fps or 15.0 |
| `timi_frame_count` | float \| null | No | Estimate from vessel length |

**Response** (200):
```json
{
  "qfr_result": {
    "qfr_value": 0.76,
    "pressure_drop_mmhg": 12.4,
    "flow_velocity_m_s": 0.28,
    "timi_frame_count": 18.0,
    "vessel_length_mm": 52.3,
    "min_diameter_mm": 1.1,
    "mean_diameter_mm": 2.4,
    "stenosis_pct": 58.3,
    "mode": "cqfr",
    "vessel_type": "LAD",
    "resting_velocity_m_s": 0.17,
    "hyperemic_velocity_m_s": 0.28,
    "qfr_profile": [1.0, 0.99, 0.98, ...],
    "pressure_profile_mmhg": [100.0, 99.5, 99.0, ...],
    "interpretation": "positive"
  },
  "mesh": {
    "vertices": [[0.1, 0.2, 0.3], ...],
    "faces": [[0, 1, 2], ...],
    "colors": [[0.2, 0.8, 0.2], ...]
  },
  "reconstruction": {
    "num_matched_points": 150,
    "angle_separation_deg": 45.0,
    "reprojection_error_px": 2.3
  }
}
```

#### GET /qfr/thresholds
Clinical QFR interpretation thresholds.

**Response** (200):
```json
{
  "positive": { "below": 0.75, "interpretation": "Hemodynamically significant" },
  "grey_zone": { "range": [0.75, 0.80], "interpretation": "Borderline" },
  "negative": { "above": 0.80, "interpretation": "Not significant" }
}
```

### 5.8 ECG & Motion

#### POST /motion/calculate
Calculate optical flow motion signal from frames.

**Request**:
```json
{ "start_frame": 0, "end_frame": 119 }
```

**Response** (200):
```json
{
  "signal": [0.1, 0.15, 0.22, ...],
  "frame_indices": [0, 1, 2, ...],
  "peaks": [12, 27, 42, 57],
  "beat_boundaries": [{"startFrame": 0, "endFrame": 19, "beatNumber": 1}, ...],
  "num_frames": 120,
  "calculation_time_ms": 3500
}
```

#### POST /motion/detect-peaks
Re-detect peaks with different parameters.

**Request**:
```json
{ "min_distance": 8, "prominence": 0.3 }
```

#### GET /motion/signal
Get current motion signal + peaks.

#### POST /motion/peaks
Set all peaks manually.

**Request**: `{ "peaks": [12, 27, 42, 57] }`

#### POST /motion/peaks/add
Add peak. **Request**: `{ "frame_index": 35 }`

#### POST /motion/peaks/remove
Remove peak. **Request**: `{ "frame_index": 27 }`

#### POST /motion/peaks/move
Move peak. **Request**: `{ "from_frame": 27, "to_frame": 29 }`

#### GET /motion/beat-boundaries
Get beat boundaries from current peaks.

### 5.9 Calibration

#### POST /calibration/catheter
Calibrate from catheter measurement.

**Request**:
```json
{
  "catheter_size_fr": 6,
  "measured_diameter_px": 6.5
}
```

Catheter sizes: 4F=1.33mm, 5F=1.67mm, 6F=2.00mm, 7F=2.33mm, 8F=2.67mm

**Response** (200):
```json
{ "pixel_spacing_mm": 0.308, "source": "catheter" }
```

#### POST /calibration/manual
Set pixel spacing directly.

**Request**: `{ "pixel_spacing_mm": 0.308 }`

#### POST /calibration/from-mask
Automatically measure catheter from segmentation mask.

**Request**: `{ "frame_index": 42, "catheter_size_fr": 6 }`

**Response** (200):
```json
{ "pixel_spacing_mm": 0.305, "measured_diameter_px": 6.56, "source": "from_mask" }
```

### 5.10 Tracking

#### POST /tracking/initialize
Initialize CSRT tracker on a frame.

**Request**:
```json
{
  "frame_index": 42,
  "roi": { "x": 150, "y": 120, "width": 160, "height": 160 },
  "confidence_threshold": 0.6
}
```

#### POST /tracking/propagate
Propagate tracker through a frame range.

**Request**:
```json
{
  "start_frame": 42,
  "end_frame": 80,
  "direction": "forward",
  "auto_segment": false
}
```

**Response** (200):
```json
{
  "results": {
    "42": { "roi": { "x": 150, "y": 120, "width": 160, "height": 160 }, "confidence": 0.95, "success": true },
    "43": { "roi": { "x": 151, "y": 121, "width": 160, "height": 160 }, "confidence": 0.92, "success": true }
  },
  "frames_tracked": 38,
  "frames_failed": 0
}
```

Progress events sent via WebSocket during propagation.

#### GET /tracking/state
Get current tracker state.

#### POST /tracking/reset
Reset tracker.

### 5.11 Mask Editing

All mask edit endpoints follow the same pattern:

**Request**: JSON with operation parameters + `frame_index`
**Response**: `Content-Type: image/png` (updated mask as binary PNG)

#### POST /mask-edit/brush
```json
{ "frame_index": 42, "points": [{"x": 200, "y": 180}], "radius": 5, "hardness": 0.8, "erase": false }
```

#### POST /mask-edit/smart-brush
```json
{ "frame_index": 42, "points": [...], "radius": 5, "tolerance": 30, "edge_sensitivity": 0.5 }
```

#### POST /mask-edit/flood-fill
```json
{ "frame_index": 42, "seed_point": {"x": 200, "y": 180}, "tolerance": 25 }
```

#### POST /mask-edit/morphological
```json
{ "frame_index": 42, "operation": "dilate", "kernel_size": 3, "iterations": 1 }
```

Operations: `dilate`, `erode`, `fill_holes`, `remove_islands`, `smooth`

#### POST /mask-edit/extract-contour
```json
{ "frame_index": 42, "simplify_epsilon": 1.0 }
```

Response: `{ "contour": [{"x": 200, "y": 180}, ...], "num_points": 120 }`

#### POST /mask-edit/deform-contour
```json
{ "frame_index": 42, "drag_point": {"x": 200, "y": 180}, "target_point": {"x": 205, "y": 182}, "influence_radius": 20 }
```

#### POST /mask-edit/edge-snap
```json
{ "frame_index": 42, "attraction_radius": 10, "strength": 0.8 }
```

### 5.12 Export & Reports

#### POST /export/csv
```json
{ "data_type": "qca", "frame_indices": [7, 8, 9, ...] }
```
**Response**: `Content-Type: text/csv`, file download

#### POST /export/json
```json
{ "data_type": "combined", "include_metadata": true }
```
**Response**: `Content-Type: application/json`

#### POST /export/pdf
```json
{ "include_charts": true, "include_ecg": true, "anonymize": true }
```
**Response**: `Content-Type: application/pdf`

#### POST /report/generate
```json
{ "report_type": "clinical_summary", "vessel": "LAD" }
```

### 5.13 Health

#### GET /health
```json
{ "status": "healthy", "version": "2.0.0", "uptime_seconds": 3600 }
```

#### GET /health/detailed
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "engines": {
    "nnunet": { "available": true, "loaded": false, "device": "cuda" },
    "angiopy": { "available": true, "loaded": true, "device": "cuda" }
  },
  "sessions": { "active": 1, "total_memory_mb": 450 },
  "gpu": { "available": true, "name": "NVIDIA RTX 3080", "memory_used_mb": 2048, "memory_total_mb": 10240 }
}
```

---

## 6. Algorithm Specifications

### 6.1 QCA Gaussian Subpixel Fitting

**Mathematical formulation**:

At each of N resampled centerline points p_i with unit normal n_i:

1. Sample intensity profile along normal: `I(t) = bilinear_interp(image, p_i + t * n_i)` for `t in [-W, W]` where W is the search width (default: 30 pixels).

2. Fit 1D Gaussian: `G(t) = A * exp(-(t - mu)^2 / (2 * sigma^2)) + offset`

3. Vessel diameter (pixels): `d_i = 2 * sigma * sqrt(2 * ln(2))` (FWHM of fitted Gaussian)

4. Convert to mm: `d_mm = d_px * pixel_spacing`

**Input contract**:
- `mask`: binary numpy array (H x W), dtype uint8, values {0, 255}
- `centerline`: Nx2 array of (y, x) coordinates, ordered proximal to distal
- `pixel_spacing`: float, mm/pixel, must be > 0
- `num_points`: int, default 50, range [10, 200]
- `method`: enum {"gaussian", "parabolic", "threshold"}

**Output contract**:
- `QCAMetrics` with all fields populated
- `diameter_profile_mm`: list of exactly `num_points` floats
- `mld_mm` >= 0, `diameter_stenosis_pct` in [0, 100]

**Parameters**:
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_points` | 50 | [10, 200] | Resampling resolution |
| `search_width` | 30 | [10, 60] | Normal search distance (px) |
| `min_diameter_px` | 1.0 | [0.5, 5.0] | Minimum valid diameter |

**Known limitations**:
- Fails on calcified lesions (sharp bright edges, not Gaussian)
- Bimodal profiles from overlapping vessels produce unreliable FWHM
- Requires valid centerline; poor centerline produces poor diameter measurements

**Validation criteria**: For a synthetic uniform-diameter tube (d=3mm), QCA should measure 3.0 +/- 0.2mm at all points.

**Reference**: SCAI Guidelines for Quantitative Coronary Analysis

### 6.2 RWS Calculation (with Hampel Filter)

**Mathematical formulation**:

```
RWS_position = (D_max - D_min) / D_max * 100%
```

where D_max and D_min are the 95th and 5th percentile values of the diameter time series for a given position (MLD, proximal RD, or distal RD) after outlier filtering.

**Hampel filter**:
1. For window of size k around each point:
   - m = median(window)
   - MAD = median(|x_i - m|) for x_i in window
   - sigma_est = 1.4826 * MAD (assuming Gaussian)
   - If |x - m| > 3.5 * sigma_est: replace x with m

**Stenosis-aware thresholding**: Adaptive threshold based on median diameter:
- Critical stenosis (<0.8mm): threshold multiplier 5.0 (very permissive)
- Severe (0.8-1.5mm): multiplier 4.5
- Moderate (1.5-2.0mm): multiplier 4.0
- Mild (2.0-2.5mm): multiplier 3.5
- Normal (>2.5mm): multiplier 3.5 (standard)

**Input contract**:
- `diameters`: dict mapping position name to list[float] (mm), one per frame
- `frame_indices`: list[int], same length as diameter lists
- `outlier_method`: enum {"none", "hampel", "double_hampel"}

**Output contract**:
- `RWSResult` with `mld_rws_pct`, `proximal_rws_pct`, `distal_rws_pct`, `average_rws_pct`
- All RWS values in [0, 100]
- `interpretation`: based on average_rws_pct thresholds

**Physiological constraints**:
- Valid diameter range: 0.2 - 5.5 mm
- Max frame-to-frame change: 20%

**Interpretation thresholds** (Hong et al., EuroIntervention 2023):
| RWS (%) | Interpretation |
|---------|---------------|
| < 8 | Normal |
| 8 - 12 | Intermediate |
| 12 - 14 | Vulnerable plaque |
| > 14 | High-risk |

**Validation criteria**: For synthetic sinusoidal diameter (d(t) = 2.5 + 0.25*sin(2*pi*t)) with frame_rate=15, RWS should be ~18.2% (theoretical: (2.75-2.25)/2.75 * 100).

### 6.3 QFR Gould/Young-Tsai Model

**Mathematical formulation**:

Total pressure drop across vessel:
```
ΔP_total = ΔP_viscous + ΔP_turbulent
```

**Viscous loss** (Poiseuille, computed per segment):
```
ΔP_viscous_i = (8 * mu * L_i * Q) / (pi * r_i^4)
```

where:
- mu = blood viscosity = 0.0035 Pa*s
- L_i = segment length (m)
- Q = volumetric flow rate = V_ref * A_ref (m^3/s)
- r_i = average radius of segment i (m)

**Turbulent loss** (Borda-Carnot expansion at stenosis exit):
```
ΔP_turbulent = Kt * (rho/2) * (1 - A_mld/A_downstream)^2 * V_mld^2
```

where:
- Kt = 1.52 (Young-Tsai empirical coefficient)
- rho = blood density = 1050 kg/m^3
- A_mld = cross-sectional area at minimum lumen diameter
- A_downstream = area of recovery zone distal to stenosis
- V_mld = Q / A_mld (velocity at stenosis by continuity)

**QFR calculation**:
```
QFR = Pd / Pa = (Pa - ΔP_total) / Pa
```
where Pa = aortic pressure = 100 mmHg (assumed).

**Kt = 1.52 justification and limitations**:

Young & Tsai (J. Biomech., 1973) measured pressure drops across symmetric stenosis models in rigid tubes under steady flow. Kt = 1.52 is the combined entrance/exit loss coefficient. This value:
- Was validated for stenosis models in rigid tubes, not compliant coronary arteries
- Does not account for pulsatile flow effects
- Does not account for asymmetric stenosis geometry
- Commercial QFR systems (Medis QAngio, Pulse Medical) use proprietary coefficients calibrated against invasive FFR in large clinical trials (FAVOR II, FAVOR III)

**This implementation is for RESEARCH USE ONLY and has NOT been validated against invasive FFR.**

**Flow velocity modes**:
| Mode | Hyperemic velocity source |
|------|--------------------------|
| fQFR | Fixed: 0.35 m/s |
| cQFR | V_rest from TIMI, V_hyp = K * V_rest^0.5 (K = 0.35 / 0.15^0.5) |
| aQFR | Directly from TIMI under adenosine |

**Resting-to-hyperemic conversion** (cQFR):
```
V_hyp = K * V_rest^alpha
alpha = 0.5
K = V_HYP_REF / V_REST_REF^alpha = 0.35 / 0.15^0.5 ≈ 0.904
```

This is a single-parameter empirical model, NOT from a published clinical study. The Tu et al. (JACC CI, 2016) population-derived model uses different covariates. The power-law exponent alpha=0.5 provides moderate damping of TFC measurement noise.

**Velocity clamping**:
- Resting: [0.0, 0.30] m/s
- Hyperemic: [0.10, 0.60] m/s

**Physical constants**:
| Constant | Value | Source |
|----------|-------|--------|
| Blood density | 1050 kg/m^3 | Standard |
| Blood viscosity | 0.0035 Pa*s (3.5 cP) | Standard |
| Aortic pressure | 100 mmHg | Assumed |
| Kt (turbulent coefficient) | 1.52 | Young & Tsai, 1973 |

**References**:
- Gould KL, "Pressure-flow characteristics of coronary stenoses", Am J Cardiol, 1978
- Young DF, Tsai FY, "Flow characteristics in models of arterial stenoses", J Biomech, 1973
- Tu S et al., "Fractional Flow Reserve Calculation from 3-D QCA and TIMI Frame Count", JACC CI, 2014
- Gibson CM et al., "TIMI frame count: a quantitative method of assessing coronary artery flow", Circulation, 1996

### 6.4 Stereo 3D Reconstruction

**Camera model**:

Intrinsic matrix:
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

where `fx = SOD / pixel_spacing_col`, `fy = SOD / pixel_spacing_row`.

IMPORTANT: Use SOD (source-to-object distance), NOT SID (source-to-image distance). When pixel_spacing is magnification-corrected to isocenter coordinates (isocenter_ps = detector_ps * SOD/SID), the projection equation becomes u = X / isocenter_ps, giving fx = SOD / isocenter_ps.

**Rotation from gantry angles**:
```
Ry(alpha) = rotation around Y-axis by primary_angle (RAO/LAO)
Rx(beta)  = rotation around X-axis by secondary_angle (CRA/CAU)
R_gantry  = Rx @ Ry
R_camera  = R_gantry^T   (camera rotates WITH the gantry)
```

**Camera position**:
```
initial_pos = [0, 0, -SOD]^T   (AP position, source behind patient)
camera_pos_world = R_gantry @ initial_pos
t = -R_camera @ camera_pos_world
```

**Epipolar matching**:
1. Compute fundamental matrix F from camera matrices
2. For each point x1 on centerline 1, compute epipolar line l2 = F @ x1 in view 2
3. Find closest centerline 2 point to epipolar line (minimum perpendicular distance)
4. Extract Longest Increasing Subsequence (LIS) of match indices for monotonicity
5. Coupled resampling using shared arc-length parameterization

**Triangulation**: Direct Linear Transform (DLT) via SVD of the system:
```
A = [x1*P1[2,:] - P1[0,:]]
    [y1*P1[2,:] - P1[1,:]]
    [x2*P2[2,:] - P2[0,:]]
    [y2*P2[2,:] - P2[1,:]]
```
X = last column of V^T from SVD(A).

**Angle validation**: Require >= 25 degrees angular separation between projections.

**Known limitations**:
- Sensitive to calibration errors (1 degree angle error at 750mm SOD = ~13mm position error)
- Foreshortened segments produce degenerate triangulation
- Only 2 views means no redundancy for error detection

### 6.5 Pan-Tompkins R-Peak Detection

**Algorithm steps**:
1. **Bandpass filter**: 5-15 Hz (emphasize QRS complex)
2. **Derivative filter**: 5-point: `y[n] = (1/8)(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])`
3. **Squaring**: `y[n] = x[n]^2` (emphasize large peaks)
4. **Moving window integration**: Window width = 150ms (0.150 * sample_rate)
5. **Adaptive thresholding**:
   - Signal level: SPKI = 0.125 * peak + 0.875 * SPKI
   - Noise level: NPKI = 0.125 * peak + 0.875 * NPKI
   - Threshold = NPKI + 0.25 * (SPKI - NPKI)
6. **Search-back**: If no peak found within 1.66 * average RR interval, search backwards with lowered threshold

**Known limitations for cath lab ECG**:
- Single-lead, often noisy
- Pacing spikes detected as R-peaks
- Baseline wander from respiration
- High-frequency catheter manipulation noise

**Reference**: Pan J, Tompkins WJ, "A Real-Time QRS Detection Algorithm", IEEE Trans. Biomed. Eng., 1985

### 6.6 Farneback Optical Flow Motion Signal

**Algorithm**:
1. For consecutive frame pairs (f[i], f[i+1]):
   - Compute dense optical flow using Farneback method
   - Parameters: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
2. Calculate magnitude: `mag = sqrt(flow_x^2 + flow_y^2)`
3. Global motion metric: `signal[i] = mean(mag)` over entire frame
4. Normalize to reference frame rate (15 fps): `signal *= frame_rate / 15.0`
5. Peak detection: `scipy.signal.find_peaks(signal, distance=min_distance, prominence=prominence)`

**Known limitations**:
- Captures ALL motion (table panning, breathing, contrast injection), not just cardiac
- No ROI restriction to cardiac region
- No frequency-based filtering to isolate cardiac frequency band

### 6.7 CSRT Tracking

**Algorithm**: OpenCV's CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) combined with optical flow refinement.

1. Initialize tracker on frame with ROI bounding box
2. For each subsequent frame:
   - CSRT update: `success, bbox = tracker.update(frame)`
   - If confidence < threshold: mark frame as failed, stop propagation
3. Optional: run segmentation + QCA on each successfully tracked frame

**Parameters**:
| Parameter | Default | Range |
|-----------|---------|-------|
| Confidence threshold | 0.6 | [0.0, 1.0] |
| ROI size | 160x160 | Fixed or adaptive |
| Direction | forward | forward, backward, both |

### 6.8 Vessel Meshing (Cross-Section Sweep)

**Algorithm**: Bishop frame (parallel transport) tube generation.

1. For each centerline point p_i with diameter d_i:
   - Compute tangent: `T = normalize(p_{i+1} - p_{i-1})`
   - Transport normal N and binormal B along centerline (no twist)
   - Generate `radial_resolution` (default 12) vertices on circle of radius d_i/2
2. Connect consecutive rings with triangle faces
3. Apply QFR coloring if provided: map QFR value to RGB heatmap (red < 0.75, yellow 0.75-0.85, green > 0.85)

**Output**: Compatible with Three.js `BufferGeometry` (flat vertex/face/color arrays).

---

## 7. ML Engine Interface

### 7.1 Abstract Interface

```python
# app/infra/ml_engines/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class EngineConfig:
    model_path: str | None = None
    device: str = "cpu"          # "cpu", "cuda", "cuda:0", "mps"
    threshold: float = 0.5
    min_area: int = 100          # Minimum connected component area
    batch_size: int = 1

@dataclass
class SegmentationResult:
    mask: np.ndarray             # HxW uint8 {0, 255}
    confidence: float            # [0, 1]
    probability_map: np.ndarray | None = None  # HxW float32 [0, 1]
    inference_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

class BaseSegmentationEngine(ABC):
    """All segmentation engines must implement this interface."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory. Must be idempotent."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model from memory. Free GPU resources."""
        ...

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,  # (x, y, w, h)
        seed_points: list[tuple[float, float]] | None = None,
    ) -> SegmentationResult:
        """
        Run segmentation inference on a single image.

        Args:
            image: HxW uint8 grayscale image
            roi: Optional ROI bounding box
            seed_points: Optional seed points for guided segmentation

        Returns:
            SegmentationResult with binary mask
        """
        ...

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique engine identifier (e.g., 'nnunet', 'angiopy')."""
        ...

    @property
    @abstractmethod
    def requires_roi(self) -> bool:
        """Whether this engine requires an ROI bounding box."""
        ...

    @property
    @abstractmethod
    def requires_seeds(self) -> bool:
        """Whether this engine requires seed points."""
        ...

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self._loaded:
            self.load_model()
```

### 7.2 Engine Registry

```python
# app/infra/ml_engines/registry.py
class EngineRegistry:
    """Registry for discovering and instantiating segmentation engines."""

    def __init__(self, models_dir: str):
        self._engines: dict[str, BaseSegmentationEngine] = {}
        self._factories: dict[str, type[BaseSegmentationEngine]] = {}
        self._models_dir = models_dir
        self._discover_engines()

    def _discover_engines(self) -> None:
        """Auto-discover available engines based on installed packages and model files."""
        try:
            from .nnunet import NNUNetEngine
            self._factories["nnunet"] = NNUNetEngine
            self._factories["nnunet-wide"] = NNUNetEngine
            self._factories["nnunet-fullframe"] = NNUNetEngine
        except ImportError:
            pass

        try:
            from .angiopy import AngioPyEngine
            self._factories["angiopy"] = AngioPyEngine
        except ImportError:
            pass

    def get(self, engine_id: str) -> BaseSegmentationEngine:
        """Get or create engine instance. Lazy loading."""
        if engine_id not in self._engines:
            if engine_id not in self._factories:
                raise EngineNotFoundError(engine_id)
            config = EngineConfig(
                model_path=f"{self._models_dir}/{engine_id}",
                device=self._detect_device(),
            )
            self._engines[engine_id] = self._factories[engine_id](config)
        return self._engines[engine_id]

    def list_available(self) -> list[dict]:
        """List all discovered engines with availability info."""
        ...

    def _detect_device(self) -> str:
        """Auto-detect best available device: cuda > mps > cpu."""
        ...
```

### 7.3 Device Management

- Auto-detect: CUDA > MPS > CPU (via `torch.cuda.is_available()`, `torch.backends.mps.is_available()`)
- Max GPU memory: If GPU memory < 2GB free, fall back to CPU
- OOM recovery: Wrap inference in `try/except RuntimeError`, call `torch.cuda.empty_cache()` on failure, return error "Insufficient GPU memory"
- Only one model loaded per engine type at a time (singleton pattern)

### 7.4 Implemented Engines Only

Do NOT include stub engines (SAM2, SegFormer, HRNet) in v2. Only spec what works:

| Engine | Architecture | Input | ROI? | Seeds? | Model File |
|--------|-------------|-------|------|--------|-----------|
| `nnunet` | nnU-Net v2, Dataset600 | Grayscale + spatial attention | Yes (160x160) | No | `nnunet/fold_0/checkpoint_best.pth` |
| `nnunet-wide` | nnU-Net v2, wider crop | Grayscale + spatial attention | Yes (192x192) | No | Same |
| `nnunet-fullframe` | nnU-Net v2, no crop | Grayscale (512x512) | No | No | Same |
| `angiopy` | U-Net + InceptionResNetV2 | Grayscale + distance map | No | Yes (2-10) | `angiopy/modelWeights-*.pth` |
| `roi+angiopy` | Pipeline: nnunet-fullframe -> seed extraction -> angiopy | Grayscale | No | Auto | Both |

---

## 8. Session & State Management

### 8.1 Session Lifecycle

```
Client                                Backend
  │                                     │
  │  POST /dicom/upload                 │
  │ ─────────────────────────────────▶  │
  │                                     │  Create session (uuid4)
  │                                     │  Parse DICOM, store Study
  │                                     │  Write to SQLite
  │  ◀───────────────────────────────── │
  │  { session_id: "abc-123", ... }     │
  │                                     │
  │  All subsequent requests include:   │
  │  X-Session-ID: abc-123              │
  │ ─────────────────────────────────▶  │
  │                                     │  Lookup session from store
  │                                     │  Return data
  │  ◀───────────────────────────────── │
  │                                     │
  │  POST /dicom/clear                  │
  │ ─────────────────────────────────▶  │
  │                                     │  Free memory
  │                                     │  Mark session inactive
  │  ◀───────────────────────────────── │
```

### 8.2 Session Store Design

```python
# app/infra/persistence/session_store.py
@dataclass
class Session:
    id: str
    study: Study | None = None
    segmentations: dict[int, SegmentationResult] = field(default_factory=dict)
    qca_measurements: dict[int, QCAMetrics] = field(default_factory=dict)
    rws_results: list[RWSResult] = field(default_factory=list)
    ecg_r_peaks: list[int] | None = None
    calibration: PixelSpacing | None = None
    motion_signal: MotionSignalResult | None = None
    qfr_session: QFRSession | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)

class SessionStore:
    """In-memory session store with optional SQLite persistence."""

    def __init__(self, db_path: str | None = None, max_sessions: int = 5):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._db = SQLiteDB(db_path) if db_path else None

    def create(self, study: Study) -> Session:
        """Create new session. Evict LRU if at capacity."""
        if len(self._sessions) >= self._max_sessions:
            self._evict_lru()
        session = Session(id=str(uuid4()), study=study)
        self._sessions[session.id] = session
        if self._db:
            self._db.save_session_metadata(session)
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session:
            session.last_accessed = datetime.utcnow()
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        if self._db:
            self._db.delete_session(session_id)

    def _evict_lru(self) -> None:
        """Remove least recently accessed session."""
        if not self._sessions:
            return
        lru_id = min(self._sessions, key=lambda k: self._sessions[k].last_accessed)
        self.delete(lru_id)
```

### 8.3 What's Stored Per Session

| Data | Storage | Persistence |
|------|---------|-------------|
| Study (frames, metadata) | Memory | DICOM file path saved to SQLite for reload |
| Segmentation masks | Memory (numpy) | PNG blobs saved to SQLite |
| QCA measurements | Memory | JSON saved to SQLite |
| RWS results | Memory | JSON saved to SQLite |
| ECG R-peaks | Memory | Frame indices saved to SQLite |
| Calibration | Memory | Saved to SQLite |
| Motion signal | Memory | NOT persisted (fast to recompute) |
| QFR session | Memory | NOT persisted (complex state, reload from projections) |

### 8.4 Memory Management

- Maximum 5 concurrent sessions (configurable)
- LRU eviction when limit reached
- Per-session memory estimate: ~50-500MB depending on frame count
- Total backend memory budget: ~2GB for sessions + ~500MB per loaded ML model
- Frame data stored as numpy arrays (NOT base64 strings)

### 8.5 Crash Recovery

On backend restart:
1. Load session metadata from SQLite
2. Frontend receives 404 on first API call (session not in memory)
3. Frontend offers "Restore previous session?" dialog
4. If accepted: POST /dicom/load-path with saved file_path
5. Backend reloads DICOM, restores segmentation masks and QCA from SQLite
6. ECG R-peaks and calibration restored from SQLite

---

## 9. Communication Protocol

### 9.1 REST API Conventions

- **Naming**: Lowercase, hyphen-separated paths (`/mask-edit/flood-fill`)
- **Methods**: GET for reads, POST for creates/actions, PUT for full updates, DELETE for removal
- **Pagination**: Not needed (single-study, bounded data)
- **Versioning**: No version prefix in v2 (single active version). If needed later, use `Accept` header versioning.
- **Request IDs**: Optional `X-Request-ID` header for tracing. Backend echoes in response.

### 9.2 WebSocket Events

Connection: `ws://127.0.0.1:8000/ws/{session_id}`

**Server -> Client events**:

```typescript
// Progress event (for long-running operations)
{
  "type": "progress",
  "operation": "segmentation" | "tracking" | "motion" | "reconstruction",
  "progress": 0.45,           // 0.0 to 1.0
  "message": "Segmenting frame 54/120",
  "frame_index": 54,          // optional
  "eta_seconds": 12.5         // optional
}

// Operation complete
{
  "type": "completed",
  "operation": "tracking",
  "result_summary": { "frames_tracked": 38, "frames_failed": 2 }
}

// Operation error
{
  "type": "error",
  "operation": "segmentation",
  "error": { "code": "ENGINE_OOM", "message": "GPU out of memory" }
}

// Session event
{
  "type": "session",
  "event": "expiring",        // Session about to expire (5 min warning)
  "session_id": "abc-123"
}
```

**Client -> Server events**:

```typescript
// Cancel operation
{
  "type": "cancel",
  "operation": "tracking"
}
```

### 9.3 Binary Frame Transport

Frame endpoints (`GET /dicom/frame/{index}`) return raw PNG bytes:
- `Content-Type: image/png`
- `Content-Length: <size>`
- `X-Frame-Index: 42`
- `X-Frame-Width: 512`
- `X-Frame-Height: 512`

Frontend decodes via `createImageBitmap(new Blob([arrayBuffer], {type: 'image/png'}))` for zero-copy canvas rendering.

### 9.4 Error Response Format

All errors follow this envelope:

```json
{
  "error": {
    "code": "FRAME_NOT_FOUND",
    "message": "Frame index 150 is out of range [0, 120)",
    "details": {
      "frame_index": 150,
      "max_frame": 119
    }
  }
}
```

Error codes are SCREAMING_SNAKE_CASE strings. The `details` object is optional and contains context-specific data. The `message` is human-readable and safe to display.

---

## 10. Caching Strategy

### 10.1 Frontend Caching Layers

| Cache | Implementation | Key | Max Size | Eviction | Invalidation |
|-------|---------------|-----|----------|----------|-------------|
| Frame cache | React Query | `['frame', sessionId, index]` | 200 entries | LRU (gcTime: 5min) | New DICOM load |
| Segmentation | React Query | `['seg', sessionId, frame, engine]` | 50 entries | LRU | Re-segmentation |
| QCA | React Query | `['qca', sessionId, frame]` | 50 entries | LRU | Calibration change |
| ImageBitmap pool | Custom LRU Map | `frame_index` | 100 entries | LRU | New DICOM load |
| Settings | localStorage | Fixed keys | Unbounded | Manual | User changes |

### 10.2 Cache Key Strategy

```typescript
// Content-addressed where possible
const segCacheKey = `${sessionId}:${frameIndex}:${engine}:${roiHash}`;
const qcaCacheKey = `${sessionId}:${frameIndex}:${pixelSpacing}`;
```

### 10.3 Backend Caching

| Cache | Storage | Lifetime | Eviction |
|-------|---------|----------|----------|
| Session data | In-memory dict | Session lifetime | LRU (max 5 sessions) |
| ML model instances | Singleton per engine | Process lifetime | Manual unload |
| Segmentation results | Session.segmentations dict | Session lifetime | Per-session eviction |

### 10.4 Cache Invalidation Triggers

| Event | Frontend invalidation | Backend invalidation |
|-------|----------------------|---------------------|
| New DICOM load | Clear all query caches | Evict LRU session if needed |
| Re-segmentation | Invalidate `['seg', sessionId, frame, *]` | Overwrite in session |
| Calibration change | Invalidate all QCA queries | QCA measurements in session become stale |
| New R-peaks | Invalidate beat boundaries | Recompute beat ranges |

---

## 11. Performance Budget

| Metric | Target | Measurement |
|--------|--------|-------------|
| Frame rendering | < 16ms (60fps) | Canvas `drawImage(ImageBitmap)` |
| Frame fetch (cached) | < 5ms | React Query cache hit |
| Frame fetch (network) | < 50ms | Binary PNG, localhost |
| Segmentation (GPU) | < 3s per frame | ML inference + post-processing |
| Segmentation (CPU) | < 15s per frame | Same, no GPU acceleration |
| QCA calculation | < 200ms | Per-frame Gaussian fitting |
| RWS calculation | < 500ms | Per-beat, includes all filtering |
| QFR reconstruction | < 5s | Stereo + QFR + mesh generation |
| Motion calculation | < 20s for 120 frames | Dense optical flow |
| DICOM upload + parse | < 10s for 500 frames | Parse + ECG extraction |
| Startup to first frame | < 5s | Backend health check + DICOM load |
| Frontend memory | < 2GB | 200 cached frames + UI |
| Backend memory (no ML) | < 1GB | Study + sessions |
| Backend memory (with ML) | < 4GB | + loaded model(s) |

---

## 12. Integration Points

### 12.1 Tauri IPC

| API | Purpose | Tauri Plugin |
|-----|---------|-------------|
| File dialog | Open/save DICOM files | `@tauri-apps/plugin-dialog` |
| File system | Read DICOM from disk (bypass upload) | `@tauri-apps/plugin-fs` |
| Shell | Launch Python backend process | `@tauri-apps/plugin-shell` |
| Window | Minimize, maximize, fullscreen | Built-in |

Backend process lifecycle:
```rust
// src-tauri/src/main.rs
let child = Command::new("python3")
    .args(["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"])
    .current_dir(python_backend_dir)
    .spawn()
    .expect("failed to start backend");

// On app close:
child.kill().expect("failed to stop backend");
```

### 12.2 Future Integrations

| Integration | Protocol | Priority |
|-------------|----------|----------|
| PACS server | DICOM C-FIND / C-MOVE (via pynetdicom) | Medium |
| DICOM Web (DICOMweb) | WADO-RS / STOW-RS over HTTPS | Medium |
| Cloud storage | S3-compatible API for study archive | Low |
| FHIR | ImagingStudy resource for EHR integration | Low |
| HL7 | ADT messages for patient context | Low |

These integrations are NOT included in v2 scope. The architecture supports them through the service layer (e.g., `study_service.py` could accept DICOM from PACS instead of file upload) without modifying the API or core layers.

---

*End of Architecture & API Specification. This document, combined with Part 1 (Product & UX) and Part 3 (Testing & Security), provides the complete blueprint for Coronary RWS Analyser v2.*
