# SPEC Part 3: Quality, Testing, Security & Decisions

> Definitive quality, testing, security, and architectural decision documentation for the Coronary RWS Analyser v2 rewrite. Every devil's advocate concern is addressed with an explicit decision, rationale, and action plan.

---

## Table of Contents

1. [Decision Log (ADR Format)](#1-decision-log-adr-format)
2. [Testing Strategy](#2-testing-strategy)
3. [Security & Compliance](#3-security--compliance)
4. [Performance & Reliability](#4-performance--reliability)
5. [Deployment & Operations](#5-deployment--operations)
6. [Observability](#6-observability)
7. [Migration Plan (v1 to v2)](#7-migration-plan-v1-to-v2)

---

## 1. Decision Log (ADR Format)

### ADR-001: Desktop App (Tauri) vs Web-Only

**Status**: Accepted

**Context**: The devil's advocate review (Issue #1) argues that Tauri adds Rust build complexity (~200MB binary) while contributing nothing beyond launching a Python subprocess and opening a webview. Hospital IT departments prefer web-based PACS-integrated solutions. The current Tauri integration uses no IPC, no Rust-side computation, and no native APIs beyond the shell plugin for launching the backend.

**Decision**: Ship as a **web application with an optional Tauri desktop wrapper**. The web deployment (Docker or simple `uvicorn` + `npm run preview`) is the primary target. Tauri is maintained for users who need a single-click installer with bundled Python backend, but the frontend must work identically in both modes.

**Consequences**:
- (+) Reduces mandatory build complexity; contributors need only Node.js + Python
- (+) Enables deployment in browser-based clinical environments (PACS integration)
- (+) Tauri desktop wrapper still available for offline/research use
- (-) Must test both deployment modes (web and Tauri)
- (-) Tauri-specific features (native file dialog, auto-update) remain secondary

**Alternatives considered**:
- **Tauri-first**: Would require moving compute to Rust (DICOM decoding, image processing). Not justified given the Python ML dependency.
- **Electron**: Larger binaries, higher memory usage, no advantage over Tauri.
- **Web-only (drop Tauri)**: Would lose the installer-based deployment option valued by research teams.

---

### ADR-002: Python Backend vs Alternative

**Status**: Accepted

**Context**: Issue #6 notes that Python's GIL limits concurrent request handling and that only ML inference (PyTorch) truly requires Python. Domain services (QCA, RWS, QFR, stereo reconstruction) are pure NumPy/SciPy. FastAPI async does not help when calling synchronous NumPy/OpenCV code, so a 2-5s segmentation request blocks all other requests.

**Decision**: **Keep Python/FastAPI** as the backend. Mitigate GIL limitations with targeted architectural changes:
1. Run ML inference in a thread pool executor (`asyncio.to_thread()` or `run_in_executor`) so it does not block the event loop.
2. Use `ProcessPoolExecutor` for truly CPU-bound batch operations (tracking propagation across 100+ frames).
3. Evaluate ONNX Runtime for inference in v2.1+ if model portability becomes needed.

**Consequences**:
- (+) Preserves the existing algorithm implementations (12,000+ lines of tested domain logic)
- (+) PyTorch, nnU-Net, and AngioPy remain directly accessible
- (+) FastAPI's async ecosystem, Pydantic v2 validation, and OpenAPI docs remain available
- (-) ML inference still requires Python runtime
- (-) Thread pool introduces minor complexity for CPU-bound tasks

**Alternatives considered**:
- **ONNX Runtime from Rust/Node**: Would enable non-Python backends but requires model conversion and loses PyTorch flexibility for new models. Premature for v2.
- **Celery/RQ worker process**: Over-engineered for a single-user desktop app. Consider only if multi-user web deployment is required.
- **Litestar**: Newer Python framework with better performance in benchmarks. Switching frameworks provides marginal benefit versus the rewrite cost.

---

### ADR-003: State Management (Zustand Stores)

**Status**: Accepted with modifications

**Context**: Issue #2 identifies 17 Zustand stores with hidden cross-store `getState()` coupling. Stores like `dicomStore` directly call `playerStore.getState().setTotalFrames()`, creating implicit bidirectional dependencies.

**Decision**: **Keep Zustand** but reduce to **10-12 stores** by consolidating related domains and replacing `getState()` bridges with an event-based coordination pattern:

| v1.1 Stores (17) | v2 Stores (11) |
|---|---|
| dicomStore + playerStore | `studyStore` (DICOM data + playback state) |
| segmentationStore + annotationStore | `segmentationStore` (masks + annotations + seeds + ROI) |
| qcaStore + calibrationStore | `analysisStore` (QCA + calibration, since calibration drives QCA) |
| rwsStore | `rwsStore` (unchanged) |
| qfrModeStore | `qfrStore` (unchanged) |
| ecgStore + motionStore | `timingStore` (ECG + motion = both provide beat boundaries) |
| trackingStore | `trackingStore` (unchanged) |
| maskEditStore | `maskEditStore` (unchanged) |
| overlayStore | `viewerStore` (overlay visibility + viewer state) |
| settingsStore | `settingsStore` (unchanged, localStorage persisted) |
| exportStore + reportStore | `exportStore` (combined) |

Cross-store communication via **Zustand subscriptions** (not direct `getState()` calls):
```typescript
// studyStore subscribes to its own metadata changes, then calls other stores
useStudyStore.subscribe(
  (state) => state.metadata,
  (metadata) => {
    useTimingStore.getState().onStudyLoaded(metadata);
    useSegmentationStore.getState().reset();
  }
);
```

**Consequences**:
- (+) Fewer stores to maintain (11 vs 17)
- (+) Subscriptions make cross-store dependencies explicit and auditable
- (+) Each store is self-contained; cross-store effects are declared at the boundary
- (-) Subscriptions add indirection; debugging requires tracing subscription chains

**Alternatives considered**:
- **Redux Toolkit**: Single store with slices. More traceable but heavier API surface. Not justified for a desktop app.
- **Jotai**: Atom-based, finer granularity. Good for large apps with many independent pieces. Our domains are inherently chunked, so store-per-domain is appropriate.
- **Single Zustand store with namespaces**: Loses the benefit of isolated resets and selective subscriptions.

---

### ADR-004: DDD Architecture Depth

**Status**: Revised (downgrade to Layered Architecture)

**Context**: Issue #8 and #30 argue that the codebase has DDD folder structure without DDD substance: no bounded contexts, no aggregates enforcing invariants, no domain events, no repositories. The `application/use_cases/` layer is mostly bypassed. Empty `events/`, `persistence/`, and `repositories/` directories confirm this is layered CRUD+compute, not DDD.

**Decision**: **Drop the DDD label.** Adopt a pragmatic **3-layer architecture**:

```
presentation/   (routes: HTTP handlers, request/response mapping)
services/       (domain logic: QCA, RWS, QFR, stereo, ECG, motion)
infrastructure/ (ML engines, DICOM I/O, file handlers)
```

Eliminate the `application/use_cases/` layer. Routes call services directly. Keep the `domain/entities/` and `domain/value_objects/` as data structures (they are well-designed). Remove empty directories (`events/`, `persistence/`, `repositories/`).

**Consequences**:
- (+) Eliminates 6+ empty files and 6+ thin wrapper use cases
- (+) Reduces import path depth by one level
- (+) Accurately represents the architecture to new contributors
- (-) If multi-step orchestration is later needed (e.g., "segment + track + calculate RWS" as one operation), a coordinator pattern must be introduced explicitly

**Alternatives considered**:
- **Full DDD with event sourcing**: Massive over-engineering for a single-user desktop tool with no database.
- **Keep 4-layer but use the use cases**: The use cases would need real orchestration logic. Currently they are one-liners. Not worth the indirection.

---

### ADR-005: Session Management

**Status**: Proposed (critical fix for v2)

**Context**: Issue #7 identifies module-level globals (`_current_study`, `_ecg_r_peaks`, `_qfr_sessions`, `_engine`, `_calibrated_pixel_spacing`) as the fundamental design flaw. Type-erased (`Any`), untestable, not concurrent-safe, not crash-recoverable.

**Decision**: Replace all module-level state with a **SessionStore** class:

```python
@dataclass
class SessionState:
    """All mutable state for one analysis session."""
    study: Optional[Study] = None
    ecg_r_peaks: Optional[List[int]] = None
    calibrated_pixel_spacing: Optional[float] = None
    motion_engine: Optional[MotionSignalEngine] = None
    rws_results: List[Any] = field(default_factory=list)
    qfr_session: Optional[QFRModeSession] = None
    tracking_engine: Optional[TrackingEngine] = None

class SessionStore:
    """In-memory session store with cleanup."""
    _sessions: Dict[str, SessionState] = {}
    _max_sessions: int = 5
    _timeout_minutes: int = 120

    def get(self, session_id: str) -> SessionState: ...
    def create(self, session_id: str) -> SessionState: ...
    def cleanup_expired(self) -> None: ...
```

The `SessionStore` is instantiated once in the app factory and injected via FastAPI `Depends()`:

```python
def get_session(session_id: str = Header(..., alias="X-Session-ID")) -> SessionState:
    return session_store.get(session_id)

@router.post("/segmentation/segment")
async def segment(session: SessionState = Depends(get_session)):
    study = session.study  # Typed as Optional[Study], not Any
```

**Session ID ownership**: Backend generates the session ID on DICOM upload, returns it in the response. Frontend stores it and sends it as `X-Session-ID` on every request. No independent ID generation on the frontend.

**Consequences**:
- (+) Typed state (no more `Any`)
- (+) Testable (inject mock SessionState)
- (+) Concurrent-safe (each tab gets its own session)
- (+) Prepares for persistence (serialize SessionState to SQLite)
- (-) Every route handler gains a `session` parameter
- (-) Slightly more complex than global access

**Alternatives considered**:
- **Redis-backed sessions**: Over-engineered for desktop. Consider only for multi-user web deployment.
- **Keep globals but add locks**: Fixes concurrency but not testability or type safety.

---

### ADR-006: Image Transport

**Status**: Proposed

**Context**: Issue #9 calculates that base64 PNG encoding adds 33% overhead. A 512x512 grayscale frame is ~100-200KB as PNG, ~130-260KB as base64. At 15fps, that is 2-4MB/s of JSON text. Initial load of 100 frames transfers 15-25MB of base64 strings.

**Decision**: **Hybrid approach** for v2:

| Data Type | Transport Method | Rationale |
|---|---|---|
| DICOM frames | Binary (ArrayBuffer via `StreamingResponse`) | High volume, latency-sensitive |
| Segmentation masks | Base64 PNG with standard prefix | Small, infrequent, needs interop |
| QFR/mesh data | JSON (numbers) | Already efficient |
| Thumbnails | Base64 JPEG | Small, lossy OK |

For frame delivery:
```python
@router.get("/dicom/frame/{frame_index}")
async def get_frame(frame_index: int, session: SessionState = Depends(get_session)):
    frame = session.study.get_frame(frame_index)
    png_bytes = encode_frame_png(frame.pixel_data)
    return Response(content=png_bytes, media_type="image/png")
```

Frontend decodes:
```typescript
const response = await fetch(`${API_BASE}/dicom/frame/${index}`);
const blob = await response.blob();
const bitmap = await createImageBitmap(blob);
// Render bitmap directly to canvas
```

This eliminates JSON wrapping, base64 encoding/decoding, and the 33% size overhead.

**Consequences**:
- (+) ~33% reduction in frame transfer size
- (+) Eliminates base64 encode/decode CPU cost on both sides
- (+) `createImageBitmap()` is off-main-thread in modern browsers
- (-) Cannot batch multiple frames in a single JSON response (use individual requests or range streaming)
- (-) Frontend must handle binary responses alongside JSON

**Alternatives considered**:
- **WebSocket frame streaming**: Better for real-time playback but adds complexity. Consider for v2.1.
- **MessagePack/Protobuf**: Requires schema management. Over-engineered for image transport.
- **Keep base64**: Simple but increasingly painful as frame counts grow.

---

### ADR-007: Real-Time Communication

**Status**: Proposed

**Context**: Issue #10 notes that long operations (segmentation 2-10s, tracking propagation 10-30s, 3D reconstruction 5-15s) have no progress reporting. Users think the app has frozen.

**Decision**: **Add Server-Sent Events (SSE)** for progress reporting. SSE is simpler than WebSocket, unidirectional (server to client), and supported natively by FastAPI.

```python
@router.post("/tracking/propagate")
async def propagate(request: PropagateRequest, session: SessionState = Depends(get_session)):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(_propagate_worker, task_id, request, session)
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
async def progress(task_id: str):
    async def event_stream():
        while not is_complete(task_id):
            yield f"data: {json.dumps(get_progress(task_id))}\n\n"
            await asyncio.sleep(0.2)
        yield f"data: {json.dumps({'complete': True, 'result': get_result(task_id)})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Apply SSE to these operations:
1. Tracking propagation (frame-by-frame progress)
2. Motion signal calculation (dense optical flow is slow)
3. Batch segmentation (multi-frame)
4. QFR 3D reconstruction (multi-step pipeline)

Single-frame segmentation (<5s) retains synchronous request/response with a frontend timeout indicator.

**Consequences**:
- (+) Users see progress (current frame, percentage, ETA) instead of a spinner
- (+) SSE is simpler than WebSocket (no connection management, auto-reconnect)
- (+) Enables cancellation (frontend closes SSE connection, backend detects and stops)
- (-) Adds task management infrastructure (task IDs, progress store, cleanup)
- (-) Frontend needs `EventSource` handling

**Alternatives considered**:
- **WebSocket**: Bidirectional but overkill for progress reporting. Consider for v2.1 if frame streaming is needed.
- **Polling**: Simple but wastes bandwidth and adds latency. Not acceptable for 2026.
- **Long polling**: Fragile and complex. No advantage over SSE.

---

### ADR-008: ML Engine Strategy

**Status**: Accepted

**Context**: Issue #27 identifies SAM2 as "dead code that looks functional" (329 lines of private API calls to model methods that will break on version updates). SegFormer and HRNet are stubs. Issue #24 notes no OOM recovery for ML inference.

**Decision**: For v2, ship only **verified, tested engines**:

| Engine | Status | Action |
|---|---|---|
| **nnU-Net (ROI)** | Production | Keep, add OOM recovery |
| **nnU-Net (wide ROI)** | Production | Keep |
| **nnU-Net (fullframe)** | Production | Keep |
| **AngioPy** | Production | Keep, add OOM recovery |
| **Hybrid (roi+angiopy)** | Production | Keep |
| SAM2 | Dead code | **Remove** from v2 codebase |
| SegFormer | Stub | **Remove** from v2 codebase |
| HRNet | Stub | **Remove** from v2 codebase |

Add OOM recovery wrapper:
```python
def safe_predict(engine, frame, **kwargs):
    try:
        return engine.predict(frame, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise SegmentationError(
                "GPU memory insufficient. Reduce ROI size or switch to CPU mode."
            ) from e
        raise
```

Re-add SAM2/SegFormer/HRNet only when:
1. Model weights are available and tested
2. Public API is used (no private `_method` calls)
3. Integration tests pass with known input/output pairs

**Consequences**:
- (+) No dead code in v2
- (+) Every listed engine actually works
- (+) OOM recovery prevents backend crash loops
- (-) Fewer engine options in the UI (5 vs 8)
- (-) Must re-implement SAM2 if later needed

**Alternatives considered**:
- **Keep stubs with "unavailable" badge**: Confuses users with non-functional options.
- **Move to plugin architecture**: Good idea for v2.1, but premature for initial release.

---

### ADR-009: Authentication

**Status**: Accepted (remove for v2.0)

**Context**: Issue #29 notes that `auth_routes.py` returns "Authentication not implemented" for all endpoints while the frontend has full JWT token management code. This is dead code adding complexity to every API call.

**Decision**: **Remove authentication entirely from v2.0.** The application is a single-user desktop tool. Authentication adds no security value when the backend runs on `127.0.0.1`.

Remove:
- `auth_routes.py`
- `AuthContext.tsx`
- JWT token storage/refresh logic in `api.ts`
- `X-Session-ID` is retained (for session routing, not authentication)

If multi-user web deployment is needed in the future, add authentication as a dedicated feature with:
- OAuth2/OIDC integration (institutional SSO)
- Role-based access control (viewer/editor/admin)
- Audit logging (see ADR for audit trail)

**Consequences**:
- (+) Removes ~400 lines of dead code from frontend API client
- (+) Simplifies every API call (no Authorization header, no token refresh)
- (+) No false sense of security
- (-) Multi-user deployment requires adding auth from scratch

**Alternatives considered**:
- **Keep stubs for future**: Stubs become debt. Better to add real auth when needed.
- **HTTP Basic Auth**: Trivially bypassable. Not worth the complexity.

---

### ADR-010: Persistence

**Status**: Proposed (high priority for v2)

**Context**: Issue #26 identifies the complete lack of persistence as the most impactful usability gap. Closing the browser tab or crashing the backend loses all analysis results. A 30-minute segmentation and RWS analysis session is irrecoverable.

**Decision**: Add **SQLite persistence** via the Python `sqlite3` standard library (no ORM needed for the simple schema):

**What to persist** (on every significant action):
- Study metadata (DICOM metadata, not pixel data)
- Segmentation results per frame (mask as compressed blob, centerline, engine used)
- Calibration settings
- QCA measurements per frame
- RWS results per beat
- QFR session state and results
- ECG R-peak edits
- Action log (see ADR for audit trail)

**What NOT to persist**:
- Raw DICOM pixel data (re-loaded from file)
- ML model state
- Transient UI state (zoom level, pan offset)

**Schema sketch**:
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    dicom_path TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata_json TEXT
);

CREATE TABLE segmentations (
    session_id TEXT REFERENCES sessions(id),
    frame_index INTEGER,
    engine TEXT,
    mask_blob BLOB,  -- compressed PNG
    centerline_json TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY (session_id, frame_index)
);

CREATE TABLE qca_results (
    session_id TEXT REFERENCES sessions(id),
    frame_index INTEGER,
    metrics_json TEXT,
    pixel_spacing REAL,
    PRIMARY KEY (session_id, frame_index)
);

CREATE TABLE rws_results (
    session_id TEXT REFERENCES sessions(id),
    beat_number INTEGER,
    result_json TEXT,
    outlier_method TEXT,
    vessel TEXT,
    PRIMARY KEY (session_id, beat_number)
);

CREATE TABLE action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    timestamp TIMESTAMP,
    action TEXT,
    details_json TEXT
);
```

**Frontend persistence**: `IndexedDB` for frame cache (binary blobs, LRU eviction). Settings remain in `localStorage`.

**Session restore**: On startup, offer to restore the most recent session if the DICOM file still exists at its original path.

**Consequences**:
- (+) Work survives browser tab close, backend crash, system reboot
- (+) Session restore eliminates re-upload and re-segmentation
- (+) Action log provides audit trail (see Security section)
- (+) SQLite requires no external service (no PostgreSQL, no Redis)
- (-) Adds ~500 lines of persistence code
- (-) Must handle schema migrations as features evolve

**Alternatives considered**:
- **PostgreSQL**: Over-engineered for single-user desktop. SQLite is the right tool.
- **IndexedDB only (frontend)**: Cannot persist backend state (segmentation masks, QCA measurements computed server-side).
- **JSON files**: Fragile, no concurrent access safety, no query capability.
- **Redis**: Ephemeral (in-memory), requires separate process. Not suitable for persistence.

---

### ADR-011: QFR Turbulent Coefficient (Kt = 1.52)

**Status**: Accepted with mandatory validation plan

**Context**: Issue #12 questions the use of Kt=1.52 from Young & Tsai (1973), which studied rigid tube stenosis models, not pulsatile coronary flow. Commercial QFR systems use proprietary coefficients calibrated against invasive FFR in large clinical trials (FAVOR II, FAVOR III).

**Decision**: **Keep Kt=1.52** as the default with the following mandatory actions:

1. **Prominent disclaimer**: Every QFR result in the UI and in exports must display: *"Research implementation. Not validated for clinical decision-making. QFR values may differ from commercially validated systems."*

2. **Configurable coefficient**: Make Kt a parameter (default 1.52) that researchers can adjust:
   ```python
   class QFR3DCalculator:
       def __init__(self, kt: float = 1.52, ...):
   ```

3. **Validation test suite**: Create golden tests against:
   - **Straight tube (no stenosis)**: QFR must be > 0.98
   - **50% diameter stenosis, 10mm length**: QFR must be 0.85-0.92 (published range)
   - **70% diameter stenosis, 20mm length**: QFR must be 0.60-0.75
   - **FAVOR II phantom data** (if publicly available): compare against published QFR values

4. **Sensitivity analysis**: Document how QFR varies with Kt in [1.0, 2.0] range for typical stenosis geometries.

**Consequences**:
- (+) Published coefficient with documented source (Young & Tsai 1973)
- (+) Configurable for researchers who want to experiment
- (+) Disclaimer prevents misuse for clinical decisions
- (-) May differ from commercial QFR systems by 0.02-0.05 units
- (-) No clinical validation dataset available for ground truth comparison

**Alternatives considered**:
- **Kt from FAVOR III**: Proprietary, not published. Cannot implement.
- **Kt=0 (ignore turbulent losses)**: Would overestimate QFR (miss significant stenoses).
- **CFD-derived Kt**: Would require per-patient CFD simulation. Not feasible for real-time analysis.

---

### ADR-012: Rewrite vs Incremental Improvement

**Status**: Accepted with constraints

**Context**: Issue #34 notes the v1.0 to v1.1 migration introduced 22+ bugs. Issue #36 argues a single-developer rewrite of 25,000+ lines is a 6-12 month project with high risk. The devil's advocate recommends incremental improvement instead.

**Decision**: **Incremental rewrite**, not big-bang. The v2 spec documents the target architecture, but implementation proceeds in phases where each phase produces a working application:

1. Each phase is independently deployable and testable
2. Old and new code coexist during transition (strangler fig pattern)
3. Regression tests are written BEFORE rewriting each component
4. No phase takes longer than 2-3 weeks

See Section 7 (Migration Plan) for the detailed phased approach.

**Consequences**:
- (+) Risk is bounded per phase
- (+) Working application at every stage
- (+) Bugs are isolated to the current phase
- (-) Longer total timeline than a clean rewrite
- (-) Temporary complexity during transition (old + new code)

**Alternatives considered**:
- **Big-bang rewrite**: High risk, as proven by v1.1 migration bugs. Rejected.
- **No rewrite (incremental fixes only)**: The module-level state, 2600-line api.ts, and lack of persistence are systemic issues that benefit from coordinated restructuring.

---

### ADR-013: Three.js for 3D Visualization

**Status**: Accepted (keep)

**Context**: Issue #4 notes that Three.js adds ~500KB gzipped for a single tube mesh. Could be replaced with raw WebGL or Canvas 2D projection.

**Decision**: **Keep Three.js**. The cost-benefit is acceptable:
- The 3D QFR mesh viewer requires orbit controls, lighting, and color-mapped geometry. Implementing this from scratch in raw WebGL would take 2-3 weeks and produce less maintainable code.
- Bundle impact is mitigated by dynamic import (code-split the 3D viewer, loaded only when QFR mode is activated).
- Future features (multi-vessel 3D, lesion markers, measurement tools in 3D) benefit from Three.js ecosystem.

```typescript
const QFRMesh3DViewer = React.lazy(() => import('./QFRMesh3DViewer'));
```

**Consequences**:
- (+) Orbit controls, lighting, and mesh rendering work out of the box
- (+) Future-proof for more complex 3D features
- (-) ~500KB bundle (mitigated by code splitting)
- (-) Dependency on Three.js ecosystem (three, @react-three/fiber, drei)

**Alternatives considered**:
- **Raw WebGL**: Lower-level, no abstraction for camera/controls/lighting. Maintenance burden.
- **regl**: Functional WebGL wrapper. Good for custom shaders but no camera/controls built in.
- **Canvas 2D projection**: Loses 3D interaction (orbit, zoom). Not acceptable for vessel inspection.

---

### ADR-014: Cornerstone.js vs Alternatives

**Status**: Proposed (evaluate removal)

**Context**: Issue #5 flags the `@ts-nocheck` on `dicomDecodeWorker.ts` as a red flag. Cornerstone.js requires SharedArrayBuffer headers (COOP/COEP) and its internal APIs break TypeScript strict mode. Meanwhile, the application primarily uses backend-decoded base64 PNG frames, making the Web Worker underutilized.

**Decision**: **Remove Cornerstone.js from v2.** Replace with:
1. **Server-side DICOM decoding** (already implemented via pydicom): Backend decodes frames and sends PNG binary.
2. **Browser-side DICOM parsing** for metadata only (using `dicom-parser`, which is lightweight and TypeScript-friendly).
3. **Canvas rendering** via `createImageBitmap()` from binary PNG responses.

This eliminates:
- The `@ts-nocheck` worker
- COOP/COEP header requirements
- SharedArrayBuffer dependency
- ~2MB of WASM codec bundles (libjpeg-turbo, CharLS, OpenJPEG, OpenJPH)

If client-side DICOM decoding is needed later (e.g., for offline Tauri mode without backend), add it back as an optional module with version-pinned Cornerstone.

**Consequences**:
- (+) Removes ~2MB from bundle
- (+) Eliminates `@ts-nocheck` and WASM codec complexity
- (+) Simpler deployment (no COOP/COEP headers needed)
- (-) All frame decoding goes through the backend (adds network latency)
- (-) Offline Tauri mode requires backend running

**Alternatives considered**:
- **Keep Cornerstone, fix types**: Cornerstone's internal APIs are not typed. Maintaining type stubs is fragile.
- **dwv (DICOM Web Viewer)**: Alternative DICOM viewer. Same WASM codec complexity.
- **OHIF Viewer integration**: Full DICOM viewer platform. Over-engineered for our use case.

---

### ADR-015: Docker / PostgreSQL / Redis

**Status**: Accepted (remove unused services)

**Context**: Issue #28 identifies PostgreSQL and Redis in `docker-compose.yml` as unused infrastructure. No database models, no Redis connections, commented-out dependencies in `requirements.txt`.

**Decision**: **Remove PostgreSQL and Redis from Docker Compose.** Simplify to a 2-container setup:

```yaml
services:
  frontend:
    image: nginx:alpine
    ports: ["80:80"]
    volumes:
      - ./dist:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      backend:
        condition: service_healthy

  backend:
    build: ./python-backend
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models
      - ./data:/app/data  # SQLite storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

If multi-user web deployment is needed later, add PostgreSQL and Redis with actual code that uses them.

**Consequences**:
- (+) Docker Compose starts faster (2 containers vs 4)
- (+) No misleading infrastructure in the repository
- (+) Developers do not waste time configuring unused services
- (-) Must add PostgreSQL/Redis from scratch if multi-user is needed

---

## 2. Testing Strategy

### 2.1 Testing Pyramid

```
           ╱  E2E Tests  ╲             5-10 critical user flows
          ╱ Integration    ╲            API + service combinations
         ╱   Unit Tests     ╲           Every algorithm, store, component
        ╱ Contract Tests      ╲         API schemas, request/response
       ╱ Snapshot Tests          ╲      UI regression
      ╱ Benchmark Tests            ╲    Performance regression
```

**Target coverage**:
- Domain services: 95%+ line coverage
- Route handlers: 80%+ line coverage
- Frontend stores: 90%+ branch coverage
- UI components: 70%+ (critical interactions, not every CSS class)
- Overall backend: 85%+
- Overall frontend: 75%+

### 2.2 Algorithm Test Plan

#### 2.2.1 QFR Calculator (`qfr_calculator_3d.py`)

**Golden test cases**:

| Test | Input | Expected QFR | Source |
|---|---|---|---|
| Uniform 3mm vessel, 50mm length | d=[3.0]*50, v=0.35 | > 0.98 | Physics: no stenosis = no pressure drop |
| 50% stenosis (1.5mm), 10mm length | d varies, min=1.5 | 0.85-0.92 | Tu et al. JACC CI 2016, Table 2 |
| 70% stenosis (0.9mm), 20mm length | d varies, min=0.9 | 0.55-0.75 | FAVOR II range |
| Serial 50% + 50% stenoses | two 1.5mm minima | < single 50% | Physics: serial lesions compound |
| Zero-length vessel | d=[3.0], length=0 | ~1.0 | Edge case |

**Property-based tests** (invariants):
- QFR is always in range [0.0, 1.0]
- QFR monotonically decreases along the vessel (pressure only drops)
- Larger stenosis always produces lower QFR (all else equal)
- fQFR, cQFR, aQFR with identical hyperemic velocity produce identical QFR
- Doubling vessel length approximately doubles viscous pressure loss

**Edge cases**:
- Single-point vessel (length = 0)
- All diameters identical (no stenosis)
- Diameter = 0 at one point (complete occlusion)
- Negative TFC (invalid input)
- TFC = 0 (division by zero)
- Extremely high velocity (> 1 m/s)
- Extremely low velocity (< 0.01 m/s)

**Regression tests** (bugs from v1.1):
- Verify `pressure_drop_mmhg` key (not `pressure_drop_mm_hg`)
- Verify LAD CTFC correction is NOT applied for segment-level TFC
- Verify velocity clamping at physiological bounds

#### 2.2.2 RWS Calculator (`rws_calculator.py`)

**Golden test cases**:

| Test | Input | Expected RWS | Interpretation |
|---|---|---|---|
| Constant diameter (3.0mm) | [3.0, 3.0, 3.0, 3.0] | 0.0% | Normal |
| Normal variation | Dmax=3.0, Dmin=2.85 | 5.0% | Normal |
| Intermediate variation | Dmax=3.0, Dmin=2.7 | 10.0% | Intermediate |
| Vulnerable plaque | Dmax=3.0, Dmin=2.6 | 13.3% | Vulnerable |
| High risk | Dmax=3.0, Dmin=2.5 | 16.7% | High risk |

**Property-based tests**:
- RWS is always >= 0%
- RWS is always <= 100%
- RWS = 0 when all diameters are equal
- Hampel-filtered RWS <= unfiltered RWS (filtering can only reduce extremes)
- RWS interpretation boundaries match documented thresholds exactly

**Edge cases**:
- Two frames only (minimum for calculation)
- 500 frames (maximum typical)
- All diameters = 0 (degenerate input)
- Single outlier 10x larger than others
- Diameter series with NaN values

**Regression tests (Issue #42 - silent Hampel fallback)**:
- Verify IQR method actually uses IQR algorithm (NOT Hampel fallback)
- Verify temporal method actually uses temporal algorithm (NOT Hampel fallback)
- Each outlier method must produce a different result on the same input with known outlier patterns

#### 2.2.3 QCA Engine (`qca_engine.py`)

**Golden test cases** (synthetic phantoms):

| Test | Input Mask | Expected MLD | Expected DS% |
|---|---|---|---|
| Uniform 20px vessel | Rectangle mask | ~20px * ps | ~0% |
| 50% stenosis at center | Tapered mask | ~10px * ps | ~50% |
| Gradual taper (normal) | Linearly narrowing | proximal/2 | based on taper |
| Circular vessel cross-section | Round mask | diameter | ~0% |

**Property-based tests**:
- All diameter values are non-negative
- MLD <= all other diameters in the profile
- DS% is in range [0%, 100%]
- Proximal RD >= MLD (by definition)
- Distal RD >= MLD (by definition)
- Profile has exactly N points (30, 50, or 70 as requested)

**Edge cases**:
- Empty mask (all zeros)
- Single-pixel-wide vessel
- Mask with disconnected components
- Centerline outside the mask (error case)
- Very large mask (2048x2048)

#### 2.2.4 Stereo Reconstructor (`stereo_reconstructor.py`)

**Golden test cases**:

| Test | Input | Expected | Source |
|---|---|---|---|
| Parallel views (0 degrees) | Same angle for both | Error: insufficient angle | Validation |
| Orthogonal views (90 degrees) | RAO 45 / LAO 45 | Well-conditioned 3D | Geometry |
| Known 3D curve, project to 2 views | Synthetic helix | Reconstruct matches input within 1mm | Round-trip test |
| Identical centerlines | Same points | Degenerate Z (all points at same depth) | Edge case |

**Property-based tests**:
- Reconstructed 3D points reproject to within 2 pixels of observed 2D points
- Vessel length in 3D >= vessel length in either 2D projection
- Confidence score is in range [0.0, 1.0]
- Angle validation rejects < 25 degrees separation

**Regression tests**:
- Verify focal length uses SOD, not SID (`fx = SOD / isocenter_ps`)
- Verify angles are read from `study.metadata['positioner_primary_angle']`
- Verify 0-degree angles produce error, not ball-shaped mesh

#### 2.2.5 ECG R-Peak Detection (`ecg_analyzer.py`)

**Test approach**: Compare against MIT-BIH Arrhythmia Database (publicly available) for a subset of records. Since cath lab ECG differs from standard ECG, also create synthetic test signals.

**Synthetic test cases**:

| Test | Input | Expected |
|---|---|---|
| Clean sinus rhythm (60 BPM) | Synthetic QRS at 1Hz, 1000Hz sample rate | 60 peaks, 1000-sample intervals |
| Tachycardia (120 BPM) | Synthetic QRS at 2Hz | 120 peaks |
| Bradycardia (40 BPM) | Synthetic QRS at 0.67Hz | 40 peaks |
| Noisy signal (SNR 10dB) | Clean + Gaussian noise | Peaks detected within 50ms |
| Baseline wander | Clean + 0.2Hz sine | Peaks detected despite drift |
| 50 Hz mains interference | Clean + 50Hz sine | Peaks detected after filtering |

**Edge cases**:
- Flat signal (no QRS complexes)
- Pacing spikes (narrow, high amplitude)
- Signal shorter than one beat
- Very high sample rate (5000 Hz)
- Very low sample rate (100 Hz)

#### 2.2.6 Motion Signal Engine (`motion_signal_engine.py`)

**Synthetic test cases**:

| Test | Input Frames | Expected |
|---|---|---|
| Static scene (no motion) | 10 identical frames | Signal near zero, no peaks |
| Periodic translation (simulate cardiac) | Frames with sinusoidal object movement | Peaks at movement frequency |
| Uniform field motion (table pan) | Frames shifted uniformly | Large magnitude, single direction |
| Two objects, different rates | Foreground cardiac + background static | Peaks match foreground rate |

**Property-based tests**:
- Motion signal length = number of frames - 1
- Motion signal values are non-negative (magnitude)
- Peak count is reasonable for given frame rate and expected heart rate range (30-200 BPM)

#### 2.2.7 Hampel Filter (`rws_calculator.py:StenosisAwareHampelFilter`)

**Test cases**:

| Test | Input | Expected |
|---|---|---|
| No outliers | [3.0, 3.1, 2.9, 3.0, 3.1] | Output = input (no changes) |
| Single spike outlier | [3.0, 3.0, 8.0, 3.0, 3.0] | Middle value replaced with median |
| Gradual trend | [3.0, 2.9, 2.8, 2.7, 2.6] | No outliers detected (gradual) |
| Stenotic region preserved | [3.0, 3.0, 1.2, 1.1, 1.2, 3.0] | Stenotic values preserved |
| Physiological bounds | [3.0, -1.0, 3.0, 7.0, 3.0] | Out-of-range values clamped to [0.2, 5.5] |
| Frame-to-frame consistency | [3.0, 3.0, 5.0, 3.0] | 5.0 clamped (>20% change) |

### 2.3 Frontend Test Plan

#### 2.3.1 Zustand Store Unit Tests

Test framework: **Vitest** with `@testing-library/react` for hook testing.

For each store, test:
1. **Initial state**: All fields have expected defaults
2. **Actions**: Each action produces expected state changes
3. **Reset**: `reset()` returns to initial state
4. **Error handling**: Failed API calls set error state, do not corrupt other state
5. **Cross-store effects**: Subscriptions fire correctly

**Priority stores** (test first):
- `studyStore`: DICOM loading, frame caching, metadata
- `segmentationStore`: Segment, cache results per frame, reset
- `rwsStore`: Calculate, accumulate results, delete results
- `qfrStore`: Dual projection workflow state machine
- `timingStore`: ECG peaks, beat boundaries, motion signal

**Mock pattern**:
```typescript
// Mock the API layer
vi.mock('@/lib/api/segmentation', () => ({
  segmentAndExtract: vi.fn().mockResolvedValue({
    mask: 'base64...',
    centerline: [{x: 10, y: 20}],
  }),
}));
```

#### 2.3.2 Component Tests

Test framework: **Vitest** + `@testing-library/react`.

**Critical component tests** (must have):
- `SegmentationPanel`: Engine selection, seed point display, segment button state
- `RWSPanel`: Frame range input, beat selection, result display, color coding
- `QFRPanel`: Projection loading, angle validation warning, QFR result display
- `PlaybackControls`: Play/pause, frame stepping, speed change
- `CalibrationPanel`: Catheter size selection, pixel spacing display
- `SeriesPicker`: Folder selection, series grid rendering, dual selection

**Test patterns**:
- Render with mock store state
- Verify correct elements are rendered for each UI state
- Simulate user interactions (click, type, select)
- Verify store actions are called with correct parameters

#### 2.3.3 Integration Tests (E2E)

Test framework: **Playwright** with Chromium.

**Critical flows** (5 tests):

1. **DICOM Load + View**: Open file -> privacy dialog -> frames display -> playback works
2. **Segment + QCA**: Load DICOM -> draw ROI -> segment -> verify mask overlay -> verify QCA numbers
3. **RWS Calculation**: Load -> segment multiple frames -> set beat range -> calculate RWS -> verify result card
4. **QFR Dual Projection**: Load two series -> segment each -> calibrate -> reconstruct -> verify QFR value
5. **Export**: Calculate RWS -> export CSV -> verify file content

**Test data**: Bundle 2-3 small synthetic DICOM files (8-16 frames each) with known properties in `tests/fixtures/`.

#### 2.3.4 Visual Regression

Tool: **Playwright screenshot comparison** (`toHaveScreenshot()`).

Capture screenshots for:
- Empty state (no DICOM loaded)
- Loaded study with segmentation overlay
- RWS result panel with color-coded values
- QFR result panel with 3D mesh view
- Dark theme variants of the above

#### 2.3.5 Accessibility Tests

Tool: **axe-core** via `@axe-core/playwright`.

Run on every page state:
- Main analysis view
- Settings modal
- Privacy dialog
- Series picker
- QFR dual-projection view

**Required pass criteria**: Zero "critical" or "serious" violations.

### 2.4 API Test Plan

#### 2.4.1 Contract Tests

Test framework: **pytest** + `httpx.AsyncClient` with `TestClient`.

For every endpoint, test:

| Category | Test |
|---|---|
| **Happy path** | Valid request -> 200 + correct response shape |
| **Invalid input** | Missing required field -> 422 with error details |
| **Wrong type** | String where int expected -> 422 |
| **Boundary values** | frame_index=0, frame_index=max, frame_index=-1 |
| **No session** | Request without loaded DICOM -> 400 or 404 with "No study loaded" |
| **Method not allowed** | GET on POST-only endpoint -> 405 |

**Example**:
```python
async def test_segment_requires_study(client):
    response = await client.post("/segmentation/segment", json={
        "frame_index": 0,
        "engine": "nnunet",
    })
    assert response.status_code == 400
    assert "No study loaded" in response.json()["error"]["message"]
```

#### 2.4.2 Error Handling Tests

Verify every error path returns a consistent error format:
```json
{
  "error": {
    "code": "SEGMENTATION_FAILED",
    "message": "Human-readable description",
    "details": {}
  }
}
```

Test that:
- DomainException -> 422
- ValidationError -> 400
- Pydantic RequestValidationError -> 400
- Unhandled Exception -> 500 with "Internal error" (no stack trace in response)

#### 2.4.3 Performance Tests

| Endpoint | Target | Test Method |
|---|---|---|
| `GET /dicom/frame/{N}` | < 50ms p99 | Benchmark 100 sequential requests |
| `POST /segmentation/segment` | < 5s (CPU) | Time with synthetic frame |
| `POST /qca/calculate` | < 500ms | Time with synthetic mask |
| `POST /rws/calculate` (10 frames) | < 3s | Time with pre-loaded study |
| `GET /health` | < 10ms | Benchmark 1000 requests |

#### 2.4.4 Session Tests

- Load two studies in sequence: verify second replaces first
- Concurrent requests to same session: verify no corruption
- Request with unknown session ID: verify 404, not crash
- Session expiry after timeout: verify cleanup

### 2.5 CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     CI Pipeline (GitHub Actions)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trigger: push to any branch, PR to main                   │
│                                                             │
│  Stage 1: Lint + Type Check (parallel)                      │
│  ├── Backend: ruff check + black --check + mypy             │
│  └── Frontend: eslint + tsc --noEmit                        │
│                                                             │
│  Stage 2: Unit Tests (parallel)                             │
│  ├── Backend: pytest tests/unit/ --cov (Python 3.10, 3.12) │
│  └── Frontend: vitest run --coverage                        │
│                                                             │
│  Stage 3: Integration Tests                                 │
│  ├── Backend: pytest tests/integration/                     │
│  └── Frontend: Playwright (Chromium)                        │
│                                                             │
│  Stage 4: Build Verification                                │
│  ├── Backend: pip install . (verify package builds)         │
│  └── Frontend: npm run build (verify Vite build)            │
│                                                             │
│  Stage 5: (main branch only)                                │
│  └── Docker build + push to registry                        │
│                                                             │
│  Required for merge to main:                                │
│  ├── All lint checks pass                                   │
│  ├── All unit tests pass                                    │
│  ├── All integration tests pass                             │
│  ├── Coverage >= 80% backend, >= 70% frontend               │
│  └── Build succeeds                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Branch strategy**:
- `main`: Production-ready. Protected. Requires passing CI + 1 review.
- `develop`: Integration branch. Feature branches merge here first.
- `feature/*`: Short-lived branches for individual features/fixes.
- `release/*`: Release preparation (version bumps, changelog).

**Release process**:
1. Create `release/v2.x.y` branch from `develop`
2. Version bump in `pyproject.toml`, `package.json`, `tauri.conf.json`
3. Generate changelog from commit messages
4. Merge to `main` via PR
5. Tag `v2.x.y`
6. CI builds Docker images and Tauri binaries
7. Publish GitHub release with artifacts

---

## 3. Security & Compliance

### 3.1 Threat Model

**Assets**:
- DICOM files containing patient data (name, ID, DOB, medical images)
- Analysis results (QCA, RWS, QFR measurements)
- Application configuration and credentials (future)

**Threat actors**:
- **Accidental exposure**: User exports data without anonymization, shares results containing patient info
- **Local attacker**: Another user on the same machine accesses DICOM files or analysis data
- **Network attacker** (web deployment only): Intercepts HTTP traffic between frontend and backend

**Attack surfaces**:

| Surface | Threat | Mitigation |
|---|---|---|
| DICOM file upload | Malformed DICOM triggers buffer overflow in pydicom | pydicom handles this; add file size limit |
| `/dicom/load-path` endpoint | Path traversal: attacker reads arbitrary files | Allowlist of directories; validate path prefix |
| `/dicom/scan-directory` endpoint | Directory traversal: enumerate filesystem | Restrict to configured DICOM directory |
| HTTP between frontend/backend | Eavesdropping on patient data | Desktop: localhost only. Web: HTTPS mandatory |
| Base64 image payloads | Large payload DoS | Request size limit (100MB) |
| Tauri FS plugin (scope: `**`) | Any file accessible from frontend | Restrict FS scope to DICOM directories only |
| CSP is null | XSS could execute arbitrary code | Set CSP to `default-src 'self'; script-src 'self'` |
| No input sanitization on metadata display | DICOM metadata with malicious strings rendered as HTML | Sanitize or use text-only rendering |

### 3.2 DICOM Security

#### 3.2.1 Anonymization Pipeline

Fields stripped during anonymization (HIPAA Safe Harbor, 18 identifiers):

| DICOM Tag | Field | Action |
|---|---|---|
| (0010,0010) | PatientName | Replace with "ANONYMOUS" |
| (0010,0020) | PatientID | Replace with random UUID |
| (0010,0030) | PatientBirthDate | Remove |
| (0010,0040) | PatientSex | Retain (non-identifying) |
| (0010,1010) | PatientAge | Retain (non-identifying at >89 threshold) |
| (0008,0080) | InstitutionName | Remove |
| (0008,0081) | InstitutionAddress | Remove |
| (0008,0090) | ReferringPhysicianName | Remove |
| (0008,1050) | PerformingPhysicianName | Remove |
| (0010,1000) | OtherPatientIDs | Remove |
| (0010,2160) | EthnicGroup | Remove |
| (0010,21B0) | AdditionalPatientHistory | Remove |
| (0020,0010) | StudyID | Replace with random |
| (0008,0050) | AccessionNumber | Replace with random |

#### 3.2.2 De-Identification Profiles

| Profile | Description | Use Case |
|---|---|---|
| **Basic** | Remove 18 HIPAA identifiers | Default for export |
| **Clean Pixel Data** | Basic + burn-in text removal | When images have patient info overlaid |
| **Retain Longitudinal** | Basic but keep StudyUID/SeriesUID | For research requiring temporal tracking |

#### 3.2.3 Export Anonymization Verification

Before any export, verify:
1. Patient name does not appear in exported DICOM metadata
2. Patient ID does not appear in exported DICOM metadata
3. Institution name does not appear
4. Filenames do not contain patient identifiers
5. PDF reports do not include patient identifiers (if anonymize option selected)

### 3.3 Medical Device Considerations

#### 3.3.1 Regulatory Classification

**This application is NOT a medical device** under the following reasoning:
- It is a **research tool** for investigating coronary artery mechanics
- It does not make diagnostic or therapeutic recommendations
- It does not replace clinical judgment or validated commercial systems (QAngio QFR, CAAS vFFR)
- QFR values are computed using published but unvalidated algorithms

However, if the application is ever used to influence clinical decisions, it would fall under:
- **FDA**: Class II Medical Device (Software as Medical Device, SaMD)
- **EU MDR**: Class IIa (aids in diagnosis using imaging data)
- **IEC 62304**: Software lifecycle requirements for medical device software

#### 3.3.2 Required Disclaimers

Every QFR result, RWS result, and clinical interpretation must display:

> **FOR RESEARCH USE ONLY**
> This software has not been validated for clinical decision-making. QFR and RWS values are computed using published algorithms that have not been calibrated against an invasive reference standard (FFR/iFR). Do not use these results to guide patient treatment decisions.

This disclaimer must appear:
- In the QFR result panel (below the QFR value)
- In the RWS result panel (below the interpretation)
- In the header of every exported PDF report
- In the first row of every exported CSV/XLSX file
- In the application About dialog

#### 3.3.3 Reproducibility Requirements

Same input must produce same output. Ensure:
1. No randomness in algorithms (no random seeds without documentation)
2. Numpy random state is fixed for any stochastic operations
3. Algorithm parameters are logged with every result
4. Software version is embedded in every export

### 3.4 Security Checklist

- [ ] **Input validation on all endpoints**: Pydantic models with field constraints
- [ ] **File upload size limit**: 500MB enforced in middleware (not just settings)
- [ ] **Path traversal prevention**: Validate all file paths against allowlist
- [ ] **CORS configuration**: Restrict to actual frontend origins (not `allow_origins=["*"]`)
- [ ] **CSP headers**: Set in Tauri config and Nginx config
- [ ] **No secrets in code/config**: API keys, tokens, passwords not committed
- [ ] **Dependency vulnerability scanning**: `pip-audit` for Python, `npm audit` for Node
- [ ] **DICOM anonymization verification**: Automated test that export anonymization actually works
- [ ] **Request rate limiting**: Not needed for desktop; add for web deployment
- [ ] **HTTPS enforcement**: Nginx config redirects HTTP to HTTPS in production
- [ ] **Logging does not contain patient data**: Verify no DICOM identifiers in log output
- [ ] **File upload type validation**: Accept only `.dcm` files, verify DICOM magic bytes

---

## 4. Performance & Reliability

### 4.1 Performance Budget

| Operation | Target | Max Acceptable | Measurement Method |
|---|---|---|---|
| App startup (frontend) | < 2s | < 4s | Lighthouse First Contentful Paint |
| App startup (backend) | < 1s | < 3s | Time from `uvicorn` start to `/health` 200 |
| DICOM load (100 frames) | < 2s | < 5s | Time from upload to first frame displayed |
| DICOM load (500 frames) | < 5s | < 12s | Same |
| Single frame display | < 16ms | < 33ms | Canvas render time per frame |
| Frame preload (batch of 20) | < 1s | < 3s | Time for 20 parallel frame fetches |
| Segmentation (GPU, nnU-Net) | < 2s | < 5s | Total API response time |
| Segmentation (CPU, nnU-Net) | < 8s | < 20s | Same |
| ML model cold start | < 10s | < 20s | First inference includes model loading |
| QCA calculation | < 300ms | < 1s | API response time |
| RWS calculation (20-frame beat) | < 2s | < 5s | API response time |
| QFR 3D reconstruction | < 3s | < 10s | API response time |
| Motion signal (100 frames) | < 3s | < 8s | API response time |
| PDF export | < 3s | < 10s | API response time |
| CSV/XLSX export | < 1s | < 3s | API response time |
| Theme switch | < 50ms | < 100ms | Time to apply dark/light class |

### 4.2 Memory Budget

| Component | Target | Max Acceptable | Notes |
|---|---|---|---|
| Frontend (no study loaded) | < 80MB | < 150MB | Base React app + Three.js (code-split) |
| Frontend (100-frame study) | < 200MB | < 400MB | Frame cache + segmentation cache |
| Frontend (500-frame study) | < 400MB | < 800MB | With LRU eviction (max 200 cached frames) |
| Backend (no ML model, no study) | < 100MB | < 200MB | FastAPI + dependencies |
| Backend (study loaded, 100 frames) | < 400MB | < 800MB | Study entity + frame pixel data |
| Backend (with nnU-Net loaded) | < 1.5GB | < 3GB | PyTorch model in GPU/CPU memory |
| Backend (nnU-Net + AngioPy) | < 2GB | < 4GB | Both models loaded |
| Total system (typical use) | < 2GB | < 4GB | Frontend + backend + 1 ML model |

**Memory management strategies**:
1. **Frontend frame LRU cache**: Keep at most 200 frames. Evict least-recently-used on overflow.
2. **Backend frame lazy loading**: Load frames on-demand from DICOM file, not all at upload time.
3. **Segmentation mask compression**: Store masks as compressed PNG blobs, decompress on access.
4. **ML model unloading**: After 10 minutes of no inference, unload model to free GPU/CPU memory. Reload on next request (cold start penalty is acceptable for desktop use).

### 4.3 Reliability

#### 4.3.1 Crash Recovery

| Scenario | Recovery Strategy |
|---|---|
| Backend crash | Frontend detects via health check failure (5s interval). Shows "Backend disconnected" banner. Auto-retry connection. On reconnect, offer to restore session from SQLite. |
| Frontend crash (tab close) | Session state persisted to SQLite (backend) and IndexedDB (frontend). On reopen, offer "Restore previous session?" |
| Browser tab accidental close | `beforeunload` event shows confirmation if unsaved analysis exists |
| OOM (backend) | ML engines wrapped in try/except with `torch.cuda.empty_cache()`. Returns user-friendly error. Other functionality continues to work. |
| OOM (frontend) | Frame cache eviction prevents unbounded growth. If frame decode fails, show placeholder. |
| Disk full | SQLite persistence fails gracefully (log warning, continue in-memory). Export fails with clear error message. |

#### 4.3.2 Graceful Degradation

| Component Failure | Degraded Behavior |
|---|---|
| ML engine unavailable | Show available engines only. Threshold segmentation fallback. |
| GPU unavailable | Automatic fallback to CPU. Warn user about slower performance. |
| ECG not in DICOM | Motion signal automatically calculated as alternative. |
| matplotlib not installed | PDF export unavailable. CSV/XLSX/JSON export still works. |
| scipy not installed | QCA uses threshold method only (no Gaussian fitting). Motion signal uses simple frame differencing. |
| Backend unreachable | Frontend shows read-only mode with cached data. Playback works. Analysis buttons disabled. |

#### 4.3.3 Data Loss Prevention

1. **Auto-save**: Every significant action (segment, calibrate, calculate) is persisted to SQLite within 1 second.
2. **Confirmation dialogs**: Before overwriting existing segmentation, before clearing session, before closing with unsaved results.
3. **Segmentation history**: Keep the 3 most recent masks per frame (can undo accidental re-segmentation).
4. **Export before close**: If the user has unsaved RWS/QFR results, the close confirmation dialog offers a one-click export option.

#### 4.3.4 Health Monitoring

**Backend health checks**:
```python
@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - START_TIME,
        "active_sessions": session_store.count(),
        "gpu_available": torch.cuda.is_available(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
    }

@router.get("/health/deep")
async def deep_health():
    # Only called periodically, not on every request
    return {
        "status": "healthy",
        "ml_engines": {
            "nnunet": check_model_loadable("nnunet"),
            "angiopy": check_model_loadable("angiopy"),
        },
        "sqlite": check_sqlite_writable(),
        "dicom_dir": check_dicom_dir_accessible(),
    }
```

**Frontend health polling**: Every 5 seconds, `GET /health`. On 3 consecutive failures, show "Backend disconnected" banner. On recovery, attempt session restore.

---

## 5. Deployment & Operations

### 5.1 Desktop Deployment (Tauri)

**Build pipeline**:
```
npm run build         # Vite -> dist/
npm run tauri:build   # Rust compilation + binary packaging
```

**Python backend bundling strategy**:
- **Option A (recommended for research)**: Require Python 3.10+ pre-installed. Tauri shell plugin runs `python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000`. Ship `python-backend/` as a directory alongside the binary. Users `pip install -r requirements.txt` on first run.
- **Option B (standalone)**: Bundle Python via PyInstaller or Nuitka. Produces a single `backend` executable. Larger binary (~500MB+) but no Python dependency.

**Auto-update**:
- Tauri v2 updater plugin with GitHub Releases as the update source
- Check for updates on startup (non-blocking)
- User confirms before downloading and installing

**Crash reporting**:
- Frontend: `window.onerror` and `unhandledrejection` handlers write to `~/.coronary-rws/crash.log`
- Backend: Structured logging to `~/.coronary-rws/backend.log`
- No automatic telemetry (privacy-first for medical data)

**Log collection**:
- Logs written to `~/.coronary-rws/logs/` with daily rotation (keep 7 days)
- "Export Logs" button in Settings for support requests

### 5.2 Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"
services:
  frontend:
    image: nginx:1.25-alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./dist:/usr/share/nginx/html:ro
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      backend:
        condition: service_healthy

  backend:
    build:
      context: ./python-backend
      dockerfile: Dockerfile
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data        # SQLite storage
      - ./dicom:/app/dicom:ro   # DICOM file directory
    environment:
      - APP_ENV=production
      - DEVICE=cpu              # or cuda for GPU
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "4"
```

**Nginx config** (reverse proxy):
```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;  # Long operations
        client_max_body_size 500M;  # DICOM upload
    }
}
```

### 5.3 Configuration Management

**Environment variables catalog**:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment: development, production, test |
| `HOST` | `127.0.0.1` | Backend bind address |
| `PORT` | `8000` | Backend bind port |
| `CORS_ORIGINS` | `http://localhost:1420,http://localhost:5173` | Allowed CORS origins |
| `DEVICE` | `auto` | ML device: auto, cpu, cuda, mps |
| `ML_MODELS_PATH` | `./models` | Directory for ML model weights |
| `DICOM_DIR` | `./dicom` | Allowed DICOM directory for path-based loading |
| `DATA_DIR` | `./data` | SQLite database and session storage |
| `LOG_LEVEL` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum DICOM file upload size |
| `SESSION_TIMEOUT_MINUTES` | `120` | Session expiry after inactivity |
| `MAX_SESSIONS` | `5` | Maximum concurrent sessions |

**Feature flags**: None in v2.0. If needed in the future, use a simple JSON config file, not a feature flag service.

**Runtime vs build-time**:
- Build-time: `__API_BASE__` (Vite define), Tauri config
- Runtime: All backend settings (environment variables + `.env` file)

---

## 6. Observability

### 6.1 Logging Strategy

**Log levels**:

| Level | When to Use | Example |
|---|---|---|
| `ERROR` | Operation failed, user action cannot complete | Segmentation engine crash, DICOM parse failure |
| `WARNING` | Degraded behavior, non-critical issue | GPU unavailable (falling back to CPU), angle < 25 degrees |
| `INFO` | Significant user actions and results | Study loaded, segmentation completed, RWS calculated |
| `DEBUG` | Algorithm internals, performance timing | Hampel filter stats, QFR segment-by-segment pressure |

**Structured logging format** (JSON):
```json
{
  "timestamp": "2026-02-10T14:23:45.123Z",
  "level": "INFO",
  "logger": "app.routes.segmentation",
  "session_id": "abc-123",
  "message": "Segmentation completed",
  "data": {
    "frame_index": 15,
    "engine": "nnunet",
    "inference_time_ms": 1234,
    "mask_pixels": 5678,
    "device": "cuda"
  }
}
```

**What to log**:
- Every API request (method, path, status, duration)
- ML inference (engine, device, inference time, mask size)
- Algorithm results (QFR value, RWS value, QCA metrics)
- Session lifecycle (create, restore, expire, destroy)
- Errors with full context (stack trace at ERROR level only)
- Configuration at startup

**What NOT to log**:
- Patient names, IDs, dates of birth
- Raw DICOM metadata
- Base64 image data
- Full pixel arrays
- Any HIPAA-identifiable information

**Log rotation**: Daily rotation, keep 7 days, max 100MB total.

### 6.2 Metrics

**Key performance indicators**:

| Metric | Collection Method | Alert Threshold |
|---|---|---|
| Segmentation inference time | Timer around `engine.predict()` | > 10s (CPU), > 5s (GPU) |
| QFR calculation time | Timer around `calculator.calculate()` | > 5s |
| Frame delivery latency | Timer around `GET /dicom/frame` | > 200ms p99 |
| Backend memory usage | `psutil.Process().memory_info().rss` | > 3GB |
| GPU memory usage | `torch.cuda.memory_allocated()` | > 80% of available |
| Active sessions | `session_store.count()` | > max_sessions |
| Error rate | Count 5xx responses / total responses | > 5% |

**Frontend performance metrics** (collected via `PerformanceObserver`):
- Time to Interactive (TTI)
- Frame render time (canvas performance)
- API call latency (per endpoint)
- Memory usage (`performance.memory` API)

**Clinical accuracy metrics** (if validation data available):
- QFR vs invasive FFR correlation (Pearson r, Bland-Altman analysis)
- RWS reproducibility (intra-observer coefficient of variation)
- QCA accuracy vs phantom diameters (mean absolute error)

### 6.3 Error Tracking

**Error categorization**:

| Category | Example | Action |
|---|---|---|
| **User error** | Upload non-DICOM file, set negative frame range | Show clear error message, no logging |
| **Input quality** | Low-quality segmentation, insufficient angle separation | Show warning, log at WARNING |
| **System error** | OOM, disk full, backend crash | Show error banner, log at ERROR |
| **Algorithm error** | QCA fitting failed, stereo degenerate | Show "calculation failed" with reason, log at ERROR |

**Error reporting format** (frontend → backend):
```json
{
  "error_id": "uuid",
  "timestamp": "2026-02-10T14:23:45.123Z",
  "category": "algorithm",
  "message": "QCA Gaussian fitting failed for frame 15",
  "stack_trace": "...",
  "context": {
    "session_id": "abc-123",
    "frame_index": 15,
    "engine": "nnunet",
    "app_version": "2.0.0"
  }
}
```

**Critical error escalation**: If the backend returns 3 consecutive 500 errors, the frontend displays a persistent error banner with "Restart Backend" button (Tauri) or "Refresh Page" suggestion (web).

---

## 7. Migration Plan (v1 to v2)

### 7.1 Risk Assessment

**What can go wrong**:

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Algorithm regression (QFR/RWS values change) | High | Critical | Golden test suite before rewriting |
| URL/API contract breaks | High | High | Contract tests for every endpoint |
| Performance regression | Medium | High | Benchmark tests run in CI |
| Feature parity gaps | Medium | High | Feature checklist verified per phase |
| Browser compatibility | Low | Medium | Playwright tests in Chromium + Firefox |
| DICOM format edge cases | Medium | Medium | Test with diverse DICOM files (vendors, modalities) |

**Data migration**: No persistent data exists in v1.1 (all in-memory). No migration needed. Settings stored in `localStorage` may need schema migration (current version 9 via Zustand persist).

**Rollback strategy**: Keep v1.1 branch tagged and deployable. Each phase produces a working application, so rollback means deploying the last successful phase.

### 7.2 Phased Approach

#### Phase 1: Foundation (Weeks 1-3)

**Scope**:
- Backend: SessionStore replacing module-level globals
- Backend: 3-layer architecture (remove application/use_cases)
- Backend: Standardized error format
- Frontend: Split `api.ts` into modules
- Frontend: Consolidate stores (17 -> 11)
- Persistence: SQLite schema + basic save/restore
- CI: Pipeline with lint + unit tests

**Features included**:
- DICOM loading and viewing
- Frame playback
- Basic session management

**Test criteria**:
- All existing backend unit tests pass
- DICOM load + frame display works
- Session persists across backend restart
- CI pipeline green

**Estimated duration**: 3 weeks

#### Phase 2: Analysis Core (Weeks 4-6)

**Scope**:
- Segmentation (nnU-Net, AngioPy, hybrid)
- Centerline extraction
- QCA calculation
- Calibration
- ECG parsing and R-peak detection
- Motion signal

**Features included**:
- Full segmentation workflow
- QCA with diameter profile and chart
- Calibration (catheter, manual, from-mask)
- ECG/motion beat boundaries
- OOM recovery for ML engines

**Test criteria**:
- Golden tests for QCA engine
- Segmentation produces correct masks on test DICOM
- ECG R-peaks detected on synthetic signal
- Contract tests for all Phase 2 endpoints

**Estimated duration**: 3 weeks

#### Phase 3: Advanced Analysis (Weeks 7-9)

**Scope**:
- RWS calculation with all outlier methods
- QFR dual-projection workflow
- Stereo 3D reconstruction
- 3D mesh visualization
- Fix IQR/temporal outlier methods (Issue #42)

**Features included**:
- RWS per-beat with interpretation
- QFR fQFR/cQFR/aQFR modes
- 3D vessel mesh with QFR heatmap
- Research-use disclaimers

**Test criteria**:
- Golden tests for RWS calculator (all outlier methods distinct)
- Golden tests for QFR calculator (known phantom data)
- Stereo reconstruction round-trip test
- IQR and temporal methods produce different results from Hampel

**Estimated duration**: 3 weeks

#### Phase 4: Polish (Weeks 10-12)

**Scope**:
- Vessel tracking (CSRT)
- Mask editing (all tools)
- Export (CSV, XLSX, JSON, PDF)
- Progress reporting (SSE)
- Session restore
- Accessibility improvements
- E2E tests

**Features included**:
- All remaining features from v1.1
- Progress bars for long operations
- Session persistence and restore
- Export with anonymization

**Test criteria**:
- 5 E2E Playwright tests pass
- Accessibility audit: zero critical violations
- Export produces valid CSV/XLSX with correct data
- Session restore works after backend restart

**Estimated duration**: 3 weeks

### 7.3 Validation Plan

#### 7.3.1 Algorithm Verification

For each algorithm (QCA, RWS, QFR, stereo reconstruction, ECG R-peak), verify v2 produces the same results as v1.1 for the same inputs:

1. **Capture v1.1 baselines**: Run v1.1 on 3+ reference DICOM files. Record exact numerical outputs (QCA diameter profiles, RWS values, QFR values) as JSON files.

2. **v2 comparison**: Run v2 on the same DICOM files with the same parameters. Compare numerical outputs against baselines.

3. **Acceptable tolerance**:
   - QCA diameters: < 0.01mm difference (numerical precision)
   - RWS values: < 0.1% absolute difference
   - QFR values: < 0.005 absolute difference
   - Stereo 3D points: < 0.1mm per coordinate

4. **Any difference beyond tolerance**: Investigate and document. If v2 is more correct (e.g., fixing the IQR/temporal fallback bug), document the improvement and update the baseline.

#### 7.3.2 Reference DICOM Datasets

| Dataset | Purpose | Properties |
|---|---|---|
| `test_normal_lad.dcm` | Normal vessel | LAD, 100 frames, 512x512, no stenosis, ECG present |
| `test_stenotic_rca.dcm` | Significant stenosis | RCA, 80 frames, 50% DS, no ECG |
| `test_qfr_pair_a.dcm` + `test_qfr_pair_b.dcm` | QFR dual projection | RAO 30 + LAO 45, same patient, LAD stenosis |
| `test_large_study.dcm` | Performance test | 500 frames, 1024x1024 |
| `test_siemens.dcm` | Vendor-specific | Siemens cath lab, embedded ECG with 50Hz noise |
| `test_philips.dcm` | Vendor-specific | Philips cath lab, different metadata format |

If real DICOM files cannot be distributed, create synthetic DICOM files using `pydicom` with known geometric phantoms (circular vessels of known diameter, known stenosis geometry).

#### 7.3.3 Automated Comparison Pipeline

```python
# tests/validation/test_v1_v2_parity.py

def test_qca_parity():
    """Verify v2 QCA matches v1.1 baseline."""
    baseline = load_json("baselines/test_normal_lad_qca.json")
    result = run_qca(load_dicom("test_normal_lad.dcm"), frame=15, engine="nnunet")

    for i, (expected, actual) in enumerate(zip(baseline["diameter_profile"], result.diameter_profile)):
        assert abs(expected - actual) < 0.01, f"Diameter mismatch at point {i}: {expected} vs {actual}"

    assert abs(baseline["mld"] - result.mld) < 0.01
    assert abs(baseline["ds_percent"] - result.ds_percent) < 0.1

def test_rws_parity():
    """Verify v2 RWS matches v1.1 baseline."""
    baseline = load_json("baselines/test_stenotic_rca_rws.json")
    result = run_rws(load_dicom("test_stenotic_rca.dcm"), start=10, end=30, method="hampel")

    assert abs(baseline["mld_rws"] - result.mld_rws) < 0.1
    assert baseline["interpretation"] == result.interpretation
```

This pipeline runs in CI on every merge to `main` to catch algorithm regressions.

---

*End of SPEC Part 3: Quality, Testing, Security & Decisions. February 2026.*
