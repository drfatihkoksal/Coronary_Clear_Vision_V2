# CONTEXT.md Devil's Advocate Review

> A ruthless, evidence-based critique of the Coronary RWS Analyser v1.1 codebase and its CONTEXT.md rewrite specification. Every issue is backed by file paths and line numbers from the actual code. The goal is not to be discouraging -- it is to prevent a failed rewrite by exposing every weakness now.

---

## A. Technology Choices

### 1. Tauri: A solution looking for a problem

**Problem**: CONTEXT.md specifies Tauri as the desktop shell, but the entire application runs as a web app (Vite dev server + FastAPI). The Tauri integration is minimal -- it launches a Python subprocess and opens a webview. The `src-tauri/` directory contributes nothing to the core functionality.

**Evidence**: The frontend calls `http://127.0.0.1:8000` directly (`vite.config.ts` defines `__API_BASE__`). There is no Tauri IPC, no Rust-side computation, no native file system integration beyond what the browser provides. The `TAURI_ENV_PLATFORM` detection in `vite.config.ts` is the only Tauri-aware code in the entire frontend.

**Risk**: Tauri adds Rust compilation requirements, platform-specific build complexity, and a ~200MB binary size. Hospital IT departments are unlikely to deploy a custom desktop app -- they use PACS-integrated web solutions. Research teams can use the web app directly.

**Recommendation**: Either commit to Tauri by moving compute-intensive operations to Rust (DICOM decoding, image processing), or drop it entirely and ship as a web application with a simple launcher script that starts the Python backend.

---

### 2. Zustand: 17 stores is 17 coordination problems

**Problem**: The codebase has 17 Zustand stores totaling 4826 lines (`src/stores/`). CONTEXT.md frames this as "modular," but the stores are tightly coupled through direct `getState()` calls across store boundaries.

**Evidence**:
- `dicomStore.ts:65-67`: Directly calls `usePlayerStore.getState().setTotalFrames()`, `.setFrameRate()`, `.setCurrentFrame()`
- `segmentationStore.ts:89`: Reads `useSettingsStore.getState().segmentation.preferredEngine`
- `segmentationStore.ts:110`: Writes back to `useSettingsStore.getState().setSegmentationSetting()`
- `calibrationStore.ts:60`: Reads `useDicomStore.getState().metadata`
- `motionStore.ts:84`: Calls `useOverlayStore.getState().setVisibility()`
- `trackingStore.ts:125`: Reads `useSettingsStore.getState().tracking.confidenceThreshold`

**Risk**: This is not modularity -- it is fragmentation with implicit coupling. The dependency graph between stores is undocumented and bidirectional. When Store A calls `StoreB.getState()`, you cannot understand Store A's behavior without also understanding Store B. A Redux-style single store with slices would make these dependencies explicit and traceable.

**Recommendation**: For the rewrite, either (a) consolidate into fewer stores with explicit dependency injection, (b) use a message bus/event pattern for cross-store communication, or (c) accept the coupling and document the dependency graph. Do not repeat the pattern of 17 independently-created stores with hidden `getState()` bridges.

---

### 3. The 2614-line api.ts: not modular, just big

**Problem**: `src/lib/api.ts` is a single 2614-line file that contains every backend API call. CONTEXT.md acknowledges this in section 13.1 as an issue but still describes the architecture as modular.

**Evidence**: `src/lib/api.ts` -- 2614 lines. Contains DICOM, segmentation, QCA, RWS, QFR, motion, calibration, tracking, export, mask-edit, auth, and report API calls all in one file. The migration from v1.0 introduced malformed URL bugs in 22 places (documented in MEMORY.md), precisely because the file is too large for a human to review effectively.

**Risk**: A single 2600-line file is the opposite of modular. It is a maintenance burden and a migration hazard (as proven by the URL space bugs).

**Recommendation**: Split into `api/dicom.ts`, `api/segmentation.ts`, `api/qfr.ts`, etc. This is acknowledged in CONTEXT.md section 13.1 item 4. Do it in v2, not "later."

---

### 4. Three.js for a single tube mesh

**Problem**: The application includes Three.js, @react-three/fiber, and @react-three/drei (substantial bundle) for rendering a single vessel tube with QFR coloring.

**Evidence**: `src/components/Viewer/QFRMesh3DViewer.tsx` is the only Three.js consumer. The mesh is a simple extruded tube with Phong shading and a color map. The vessel mesh itself is generated server-side in `vessel_mesher.py` (lines 1-50: circular cross-sections, 12 radial subdivisions).

**Risk**: Three.js adds ~500KB (gzipped) to the bundle for a feature that could be implemented with raw WebGL or even a 2D SVG projection. The orbit controls and lighting setup are boilerplate.

**Recommendation**: Evaluate whether a lightweight alternative (e.g., `regl`, raw WebGL, or even a Canvas 2D projection for a single tube) would suffice. If 3D interaction is truly needed, keep Three.js but acknowledge the cost-benefit ratio.

---

### 5. Cornerstone.js: @ts-nocheck is a red flag

**Problem**: `src/lib/dicomDecodeWorker.ts` has `@ts-nocheck` because Cornerstone's internal APIs do not play well with TypeScript strict mode.

**Evidence**: `src/lib/dicomDecodeWorker.ts` line 1: `// @ts-nocheck`. The worker also requires SharedArrayBuffer headers (COOP/COEP), which the Vite config must inject via a custom plugin (`vite.config.ts` cross-origin-isolation plugin).

**Risk**: `@ts-nocheck` means the entire DICOM decoding worker is untyped. Any API changes in Cornerstone (which has had breaking changes between versions) will be undetectable at compile time. The SharedArrayBuffer requirement adds deployment complexity (specific HTTP headers needed).

**Recommendation**: Evaluate whether Cornerstone.js is still the right choice. Alternatives like `dwv` (DICOM Web Viewer) or direct `pydicom` server-side rendering (which you already do for base64 frames) might eliminate the worker entirely. If keeping Cornerstone, pin the exact version and document the `@ts-nocheck` scope.

---

### 6. FastAPI + Python: justified, but barely

**Problem**: CONTEXT.md lists Python/FastAPI for the backend. The only irreplaceable reason is ML model inference (PyTorch, nnU-Net). All other computation (QCA, RWS, QFR, stereo reconstruction) is pure NumPy/SciPy that could run in any language.

**Evidence**: The ML engines (`infrastructure/ml_engines/`) are the only PyTorch consumers. Domain services (`domain/services/`) use only NumPy and SciPy. The QFR calculator (`qfr_calculator_3d.py`), stereo reconstructor, QCA engine, and RWS calculator are pure numerical code.

**Risk**: Python's GIL limits concurrent request handling. A segmentation request that takes 2-3 seconds blocks the event loop for other requests (FastAPI's async does not help when calling synchronous NumPy/OpenCV code).

**Recommendation**: For the rewrite, consider: (a) running ML inference in a separate worker process (Celery, RQ, or subprocess), (b) keeping the REST API thin and farming compute to background tasks, or (c) using ONNX Runtime to serve models from a non-Python backend. Python is fine for v2, but structure it to not block on long-running compute.

---

## B. Architecture Flaws

### 7. Module-level `_current_study`: a fundamental design flaw

**Problem**: The backend's session management is a module-level global variable.

**Evidence**: `dicom_routes.py:37-40`:
```python
_current_session_id: Optional[str] = None
_current_study: Any = None
_current_file_path: Optional[str] = None
_ecg_r_peaks: Optional[List[int]] = None
```
All other routes import from this via `_get_current_study()`. The QFR session (`qfr_session.py`) has a separate global dict keyed by session ID.

**Risk**:
- **No concurrency**: Two browser tabs hit the same global. Loading a study in Tab B overwrites Tab A's study silently.
- **No crash recovery**: If the backend crashes, all state is lost. The user must re-upload and re-segment from scratch.
- **Untestable**: Unit tests cannot run in parallel because they share the module-level state. Integration tests must serialize.
- **No horizontal scaling**: Cannot deploy multiple backend instances behind a load balancer.
- **Type erasure**: `_current_study` is typed as `Any`, not `Study`. This defeats type checking entirely.

**Recommendation**: Implement a `SessionStore` (even in-memory with cleanup) keyed by session ID. Every route receives the session ID from the frontend and retrieves the study from the store. This is a ~2 hour refactor that fixes testability, concurrency, and scalability in one step.

---

### 8. DDD cosplay: folders, not bounded contexts

**Problem**: CONTEXT.md claims "DDD / Clean Architecture" but the codebase has none of the actual DDD patterns -- no bounded contexts, no aggregates enforcing invariants, no domain events, no repositories.

**Evidence**:
- `domain/events/__init__.py`: Empty file. Zero domain events in the entire codebase.
- `infrastructure/persistence/models/__init__.py`: Empty. No persistence models.
- `infrastructure/persistence/repositories/__init__.py`: Empty. No repositories.
- `domain/entities/study.py`: The `Study` entity is a data class with helper methods. It does not enforce invariants -- you can set `frame_rate = -1` from outside (the `__post_init__` only validates on construction, not on mutation).
- There is no Unit of Work, no specification pattern, no event sourcing. The "DDD layers" are just import path conventions.

**Risk**: Calling it "DDD" in the CONTEXT.md misleads the rewrite team into thinking they need to preserve complex domain patterns. There are none to preserve. It is a layered CRUD+compute backend with good separation of concerns.

**Recommendation**: Drop the DDD label in the rewrite spec. Call it what it is: a layered architecture with presentation, service, and infrastructure layers. Do not invest in "proper DDD" for a single-user desktop tool -- it is unnecessary complexity.

---

### 9. Base64 everywhere: 33% overhead on every frame

**Problem**: Every frame and mask is transported as base64-encoded PNG over JSON.

**Evidence**: `dicom_routes.py` encodes frames as `numpy -> PNG -> base64`. `segmentation_routes.py` encodes masks the same way. QFR routes return raw base64 without the `data:image/png;base64,` prefix (inconsistency documented in CLAUDE.md).

**Calculation**: A 512x512 grayscale frame is ~262KB raw. PNG compression brings it to ~100-200KB. Base64 adds 33%, so ~130-260KB per frame. At 15fps streaming, that is 2-4MB/s of JSON text through an HTTP response. For the typical 100-frame cine loop, the initial load transfers ~15-25MB of base64 text.

**Risk**: Slow initial load, high memory pressure (base64 strings are 33% larger than binary), and GC pressure from allocating/deallocating large strings. This is the single biggest performance bottleneck for a fluid user experience.

**Recommendation**: Use binary transport (ArrayBuffer responses with `Content-Type: application/octet-stream`) for frames. The frontend already uses Cornerstone.js which works with raw pixel data. Keep base64 only for masks (which are small binary images). Consider WebSocket streaming for playback.

---

### 10. No WebSocket: polling in 2026

**Problem**: CONTEXT.md explicitly states "No WebSocket -- all communication is request/response." Long operations (segmentation, tracking propagation, 3D reconstruction) have no progress reporting mechanism.

**Evidence**: `motion_signal_engine.py:104` has a `progress_callback` parameter, but the HTTP route that calls it (`motion_routes.py`) cannot stream progress to the frontend. The frontend has no SSE or WebSocket listener. Long operations appear to hang.

**Risk**: Users will think the application has frozen during:
- ML segmentation (2-10 seconds depending on engine and hardware)
- Tracking propagation across 100+ frames (10-30 seconds)
- QFR 3D reconstruction (5-15 seconds)
- Motion signal calculation (5-20 seconds for dense optical flow)

**Recommendation**: Add WebSocket or SSE for progress events. FastAPI supports both natively. This is especially important for tracking propagation where the user needs to see frame-by-frame progress.

---

### 11. Session ID mismatch between frontend and backend

**Problem**: CONTEXT.md section 13.1 item 6 acknowledges this: "Frontend and backend generate session IDs independently."

**Evidence**: The frontend generates a session ID in `src/lib/sessionUtils.ts`. The backend generates its own in `dicom_routes.py:37`. The `X-Session-ID` header is sent but rarely validated. The QFR session store (`qfr_session.py`) keys by the backend-generated ID, which the frontend may not know about.

**Risk**: Session operations silently fail when IDs do not match. This was already the root cause of the motion routes bug (documented in MEMORY.md: "session stores 'study' not 'dicom_handler', and frontend/backend session_ids differ").

**Recommendation**: Backend generates session ID on DICOM upload, returns it, frontend stores and sends it with every request. Simple, deterministic, no ambiguity.

---

## C. Algorithm Validity

### 12. QFR Gould/Young-Tsai: Kt=1.52 needs justification

**Problem**: The QFR calculator uses Kt=1.52 as the turbulent loss coefficient, citing Young & Tsai (1973). The code does not validate this against clinical QFR implementations.

**Evidence**: `qfr_calculator_3d.py:45`:
```python
KT = 1.52    # Turbulent/expansion loss coefficient (Young-Tsai)
```

The original Young & Tsai paper (JBiomech 1973) studied stenosis models in rigid tubes, not pulsatile flow in compliant coronary arteries. Commercial QFR systems (Medis QAngio QFR, Pulse Medical AngioPLUS) use proprietary coefficients calibrated against invasive FFR measurements in large clinical trials (FAVOR II, FAVOR III).

**Risk**: An uncalibrated Kt value directly affects QFR accuracy. If Kt is 10% too high, QFR will be systematically lower, leading to over-diagnosis of significant stenosis. Clinical decisions (PCI vs. medical therapy) hinge on QFR > 0.80. Getting this wrong has direct patient harm potential.

**Recommendation**: (a) Validate QFR output against published phantom or clinical datasets (e.g., FAVOR II), (b) document that this is a research implementation not validated for clinical use, (c) add prominent disclaimers in the UI and exports.

---

### 13. Resting-to-hyperemic velocity conversion: power law is ad hoc

**Problem**: The cQFR mode converts resting flow velocity to hyperemic using a power-law model with alpha=0.5, calibrated to a single reference point.

**Evidence**: `qfr_calculator_3d.py:219-266`:
```python
ALPHA = 0.5
V_REST_REF = 0.15   # typical resting velocity (m/s)
V_HYP_REF = 0.35    # typical hyperemic velocity (m/s)
K = V_HYP_REF / (V_REST_REF ** ALPHA)
v_hyperemic = K * (v_rest ** ALPHA)
```

This is a single-parameter model calibrated to one (V_rest, V_hyp) pair. The Tu et al. (JACC CI 2016) paper uses a population-derived regression model with multiple covariates (vessel type, patient demographics). The power law with alpha=0.5 is not from any published model -- the code comments say "recommended" without citing a source.

**Risk**: The conversion from resting to hyperemic velocity is the most sensitive parameter in cQFR. A 20% error in hyperemic velocity translates to a ~10% error in QFR for moderate stenoses (where clinical decisions are made).

**Recommendation**: Either implement the Tu et al. empirical model (published coefficients exist) or clearly document that the power-law is a simplification with unknown accuracy relative to clinical cQFR.

---

### 14. Pan-Tompkins for angiography ECG: wrong tool for the job

**Problem**: Pan-Tompkins is designed for clean 12-lead ECG at 200-360 Hz sampling. Cath lab DICOM ECG is a single lead, often noisy, sometimes at 500-1000 Hz, and frequently has pacing artifacts.

**Evidence**: `ecg_analyzer.py:28-50` implements Pan-Tompkins as the primary R-peak detector. The `siemens_ecg_filter.py` adds a 50/60 Hz notch filter, but this does not address the fundamental issue: cath lab ECG has:
- Baseline wander from respiration and patient movement
- High-frequency noise from catheter manipulation
- Pacing spikes that Pan-Tompkins detects as R-peaks
- Variable signal quality (wet electrodes, poor contact)

**Risk**: Missed R-peaks lead to incorrect beat boundaries, which lead to incorrect RWS calculations. False peaks (pacing spikes) lead to non-physiological beat durations. The manual R-peak editing feature (which works correctly) exists precisely because automated detection is unreliable.

**Recommendation**: Add pacing spike detection (narrow high-amplitude pulse), improve baseline wander removal (high-pass at 0.5 Hz), and consider a more robust detector like Hamilton-Tompkins or wavelet-based detection for noisy single-lead signals.

---

### 15. Farneback optical flow: measures everything, not just cardiac motion

**Problem**: The motion signal engine uses dense optical flow to detect cardiac phases, but optical flow captures ALL motion -- table movement, breathing, contrast injection, and C-arm rotation.

**Evidence**: `motion_signal_engine.py:265-278` computes `np.mean(magnitude)` over the entire frame. There is no ROI restriction, no motion decomposition, and no artifact rejection.

**Risk**: Table panning (common during procedures) creates a massive motion spike that dwarfs cardiac motion. Contrast injection creates flow-related motion unrelated to cardiac phase. The peak detector will find these artifacts instead of systolic peaks.

**Recommendation**: (a) Restrict optical flow calculation to a cardiac ROI (e.g., heart silhouette), (b) use a bandpass filter on the motion signal to isolate cardiac frequency (0.7-2.5 Hz = 40-150 BPM), (c) detect and exclude table motion events (uniform flow field = table movement, divergent flow field = cardiac motion).

---

### 16. Gaussian subpixel diameter fitting: breaks on calcified lesions

**Problem**: QCA uses Gaussian fitting on the perpendicular intensity profile to measure vessel diameter. This assumes the vessel edge has a Gaussian roll-off, which is only true for normal soft-tissue vessels.

**Evidence**: `qca_engine.py:542-588` fits `A * exp(-((t - mu)^2) / (2 * sigma^2)) + offset` to the intensity profile. The fallback is threshold-based measurement (`_threshold_diameter`).

**Risk**: Calcified lesions have sharp, bright edges (calcium appears bright on fluoroscopy). Stents are metallic with distinct edge profiles. Overlapping vessels create bimodal profiles. In all these cases, the Gaussian fit either fails (falls back to threshold) or produces incorrect FWHM values.

**Recommendation**: The threshold fallback is adequate for most cases. Consider adding: (a) edge detection (gradient-based boundary finding), (b) multi-modal profile detection (for overlapping vessels), (c) quality metrics on the Gaussian fit (R-squared, residual analysis) to flag unreliable measurements.

---

### 17. Stereo reconstruction with only 2 views: fragile geometry

**Problem**: The stereo reconstructor uses two angiographic views to triangulate 3D points. With only 2 views, the reconstruction is highly sensitive to calibration errors and foreshortening.

**Evidence**: `stereo_reconstructor.py` requires >= 25 degrees angular separation. But even at 25 degrees, foreshortened vessel segments create near-degenerate triangulation (the memory documents a case where both projections got 0 degrees, producing a "ball-shaped" mesh with Z stddev=3454mm).

**Risk**: Calibration errors (SID, SOD, pixel spacing, gantry angles) compound quadratically in triangulation. A 1-degree error in angle at 750mm SOD translates to ~13mm position error at the isocenter. For a 3mm vessel, this is a 400% error in diameter.

**Recommendation**: (a) Add reprojection error reporting -- if the triangulated 3D point does not reproject close to the observed 2D point, the reconstruction is unreliable, (b) add a confidence metric per 3D point based on triangulation angle, (c) consider bundle adjustment to refine camera parameters from the matched points.

---

### 18. Hampel filter for RWS: MAD assumes symmetry

**Problem**: The RWS calculator uses a Hampel filter (MAD-based outlier detection) on diameter time series. MAD assumes a symmetric distribution around the median, but cardiac diameter variation is inherently asymmetric (rapid contraction, slow relaxation).

**Evidence**: `rws_calculator.py:64-255` implements `StenosisAwareHampelFilter` with stenosis-aware adaptive thresholding. The filter uses `k = 1.4826` (MAD-to-sigma conversion for normal distribution). But physiological diameter variation is NOT normally distributed -- systole is faster than diastole, creating a skewed distribution.

**Risk**: The asymmetry means the filter may clip genuine diastolic peaks (slow, gradual increase) less aggressively than systolic dips (rapid, sharp decrease). This could systematically underestimate Dmax and overestimate Dmin, biasing RWS downward.

**Recommendation**: Consider a phase-aware filter that uses different thresholds for the contraction and relaxation phases of the cardiac cycle. Or use percentile-based extrema (which the code already does -- 95th/5th percentile in `_calculate_position_rws_hampel` at line 676) and drop the Hampel filter entirely, since the percentiles are already robust to outliers.

---

## D. What is Missing

### 19. Zero tests for the new QFR calculator

**Problem**: The rewritten QFR calculator (`qfr_calculator_3d.py`) has ZERO unit tests.

**Evidence**: `tests/unit/domain/test_qfr_calculator.py` imports from `qfr_calculator.py` (the OLD 2D calculator), NOT from `qfr_calculator_3d.py`. A grep for `qfr_calculator_3d` in the tests directory returns no results. The new 554-line calculator that implements the Gould/Young-Tsai model -- the most clinically critical component -- is completely untested.

**Risk**: Any change to the QFR calculator could silently break it. The power-law velocity conversion, the Borda-Carnot expansion loss calculation, and the segmental pressure model are all complex numerical code that needs regression tests with known-good values.

**Recommendation**: Before any rewrite, create test cases with known phantom data: (a) uniform vessel (QFR should be ~1.0), (b) 50% stenosis at known length (compare against published QFR values), (c) edge cases (zero-length vessel, single-point stenosis, extreme velocities).

---

### 20. Zero frontend tests

**Problem**: There are zero frontend test files.

**Evidence**: `find src -name "*.test.*" -o -name "*.spec.*"` returns 0 results. The `vitest.config.ts` and `playwright.config.ts` exist but are never used. CONTEXT.md lists "Vitest (unit) + Playwright (E2E)" in the tech stack without mentioning that no tests exist.

**Risk**: The entire frontend is untested. The 17 Zustand stores, the 2614-line API client, the canvas layer system, the QFR dual-projection workflow -- all of it is verified only by manual testing. Any rewrite has no safety net.

**Recommendation**: Before the rewrite, write at minimum: (a) unit tests for each Zustand store's core actions, (b) integration tests for the critical workflows (DICOM load, segment, QCA, RWS calculate), (c) a single E2E test that runs the happy path. This gives you a reference to verify the rewrite against.

---

### 21. No undo/redo beyond mask editing

**Problem**: Only mask editing has undo/redo. Overwriting a segmentation, changing calibration, or deleting an RWS result is irreversible.

**Evidence**: `maskEditStore.ts` has an edit history (527 lines, the largest store). No other store has any history mechanism. `segmentationStore.ts` overwrites previous segmentation results on re-segment. `calibrationStore.ts` overwrites pixel spacing on any calibration change.

**Risk**: Users will accidentally overwrite segmentations (e.g., running nnU-Net on a frame that had a manually refined mask). This is especially frustrating because manual mask editing is time-consuming.

**Recommendation**: At minimum, add confirmation dialogs before destructive operations ("This will overwrite the existing segmentation. Continue?"). Ideally, add a per-frame segmentation history (keep the last N masks).

---

### 22. No audit trail for a medical application

**Problem**: This is a medical imaging tool used for clinical decisions (RWS interpretation, QFR-based PCI decisions). There is no audit trail of what the user did, what parameters were used, or what results were generated.

**Evidence**: No logging of user actions to any persistent store. Backend logging goes to stdout/stderr (transient). No record of: which frames were segmented, which engine was used, what calibration was applied, which peaks were manually edited, etc.

**Risk**: Regulatory risk. If this tool is ever used for clinical decisions (even in a research context), there needs to be a record of the analysis parameters and results. This is a requirement for ISO 13485 and IEC 62304 compliance.

**Recommendation**: Add a simple action log (JSON file or SQLite) that records timestamped entries: study loaded, frame N segmented with engine X, calibration set to Y mm/px, RWS calculated for beat N with result Z%, etc.

---

### 23. No input validation for non-coronary DICOM

**Problem**: What happens when someone uploads a chest CT, a brain MRI, or a non-medical DICOM file?

**Evidence**: `dicom_routes.py` and `load_study.py` parse any valid DICOM file. There is no check for modality (should be "XA" for X-ray angiography), no check for NumberOfFrames (should be > 1 for cine), no check for image dimensions (coronary angiograms are typically 512x512 or 1024x1024).

**Risk**: The segmentation engines will produce garbage masks on non-coronary images. The QCA engine will measure "diameters" of whatever the mask contains. The user will get meaningless RWS/QFR values with no warning.

**Recommendation**: Add validation on upload: check modality is "XA", check NumberOfFrames > 1, check image dimensions are reasonable (256-2048 square). Warn but allow override for non-standard inputs.

---

### 24. No error recovery for ML OOM

**Problem**: If an ML model runs out of GPU memory during inference, the backend crashes or returns an opaque 500 error.

**Evidence**: `nnunet_engine.py` and `angiopy_engine.py` do not catch `torch.cuda.OutOfMemoryError`. The global exception handler in `error_handler.py` catches generic exceptions but does not clean up GPU memory.

**Risk**: After an OOM, the GPU memory remains allocated (PyTorch does not automatically free it). Subsequent segmentation attempts will also OOM. The only recovery is restarting the backend.

**Recommendation**: Wrap inference calls in try/except for `RuntimeError` (PyTorch OOM), call `torch.cuda.empty_cache()` on failure, and return a meaningful error message ("Insufficient GPU memory. Try a smaller ROI or switch to CPU.").

---

### 25. No cancellation for long-running operations

**Problem**: Once started, segmentation, tracking propagation, and 3D reconstruction cannot be cancelled.

**Evidence**: No endpoint provides a cancellation mechanism. The tracking propagation (`tracking_routes.py`) iterates through frames synchronously. The frontend has no "Cancel" button for any long-running operation.

**Risk**: If a user accidentally starts tracking propagation across 200 frames, they must wait 30+ seconds or reload the page (losing all state).

**Recommendation**: Use background tasks (FastAPI `BackgroundTasks` or asyncio tasks) with cancellation tokens. Add a `/cancel` endpoint per long-running operation. Show a progress bar with a cancel button.

---

### 26. No persistent storage: close the app, lose everything

**Problem**: All analysis results are stored in memory. Closing the browser tab or restarting the backend loses everything.

**Evidence**: Backend state is in module-level globals (`dicom_routes.py:37-40`). Frontend state is in Zustand stores (memory only, except `settingsStore` and `calibrationStore` which use localStorage). No database, no file-based persistence.

**Risk**: A user who has spent 30 minutes segmenting, tracking, and calculating RWS across multiple beats loses all results if:
- The browser tab is accidentally closed
- The backend crashes
- The system runs out of memory
- The user navigates away from `/app`

**Recommendation**: Add SQLite persistence for session data. On every significant action (DICOM load, segmentation, calibration, RWS calculation), save to disk. On page load, offer to restore the previous session.

---

## E. Over-Engineering

### 27. SAM2 engine: fully implemented, cannot possibly work

**Problem**: `sam2_engine.py` (329 lines) is a complete SAM2 inference implementation -- not a stub as CONTEXT.md claims. But it references internal SAM2 APIs (`model.forward_image`, `model._prepare_backbone_features`) that are private and version-specific.

**Evidence**: `sam2_engine.py:221-258` calls `self.model.forward_image()`, `self.model._prepare_backbone_features()`, `self.model.sam_prompt_encoder()`, `self.model.sam_mask_decoder()` -- all private API methods with leading underscores. These will break on any SAM2 version update.

**Risk**: The SAM2 engine looks functional but is fragile and untested. It gives the impression of a working feature while being practically unusable (requires a fine-tuned `best_model.pth` that does not exist in the repository, and depends on private API stability).

**Recommendation**: Either remove the SAM2 engine entirely (it is dead code) or promote it to a properly tested, version-pinned integration. Do not ship dead code that looks functional.

---

### 28. Docker + PostgreSQL + Redis: for a single-user desktop app

**Problem**: `docker-compose.yml` provisions 4 containers including PostgreSQL 15 and Redis 7. The backend does not use either service -- there are no database models, no Redis connections.

**Evidence**:
- `infrastructure/persistence/models/__init__.py`: Empty
- `infrastructure/persistence/repositories/__init__.py`: Empty
- `requirements.txt:35-39`: SQLAlchemy, Alembic, Redis are commented out
- `docker-compose.yml` configures `DATABASE_URL` and `REDIS_URL` environment variables that no code reads

**Risk**: Over-engineered infrastructure that wastes time configuring, maintaining, and debugging container orchestration for services that are not used. A new developer will spend time understanding the Docker setup only to discover it is aspirational.

**Recommendation**: Remove PostgreSQL and Redis from Docker Compose. Add them back when there is actual code that uses them. Keep a single-container Docker setup (backend only) for the common deployment case.

---

### 29. Auth system: for a single-user local app

**Problem**: `auth_routes.py` has login/logout/me endpoints. `src/lib/AuthContext.tsx` implements JWT token management with auto-refresh. This is for a desktop app that processes local DICOM files.

**Evidence**: `auth_routes.py` returns `{"message": "Authentication not implemented"}` for all endpoints. Meanwhile, `src/lib/api.ts` has full token refresh logic, 401 interception, and session management. The frontend auth code is substantial and completely non-functional.

**Risk**: Dead code that confuses developers. The auth token infrastructure adds complexity to the API client without providing any security.

**Recommendation**: Remove auth entirely from v2 if it is a single-user desktop app. If multi-user is a future goal, add it when you add the database and user model -- not before.

---

### 30. DDD layers: 4 layers for what is a computation service

**Problem**: The backend has 4 layers (presentation, application, domain, infrastructure) with empty event, persistence, and repository directories.

**Evidence**:
- `application/use_cases/`: 6 files, mostly thin wrappers around domain services
- `application/services/__init__.py`: Empty
- `application/interfaces/`: 3 interface files that are only implemented once each
- `domain/events/__init__.py`: Empty (no domain events)

**Risk**: The layered architecture adds import path complexity and file navigation overhead without providing the benefits that justify it (testability via interfaces, event-driven decoupling, persistence abstraction). The "use cases" are one-liners that call domain services.

**Recommendation**: Flatten the backend for v2. Routes call services directly. Keep the domain services as-is (they are well-structured). Remove the application layer unless you actually need orchestration (e.g., a use case that coordinates multiple services with transactional guarantees).

---

## F. Under-Engineering

### 31. No progress indication for any operation

**Problem**: Segmentation, tracking, motion calculation, and 3D reconstruction provide no progress feedback to the user.

**Evidence**: The backend services accept `progress_callback` parameters (e.g., `motion_signal_engine.py:104`) but the HTTP routes do not propagate progress to the frontend. The frontend shows a loading spinner with no percentage or ETA.

**Risk**: Users will think the application has crashed during long operations.

**Recommendation**: Even without WebSocket, you can use HTTP streaming (chunked transfer encoding) or SSE to send progress updates. FastAPI supports `StreamingResponse` natively.

---

### 32. No file locking or concurrent access protection

**Problem**: Two instances of the application (or two browser tabs) can corrupt state silently.

**Evidence**: Module-level globals (`dicom_routes.py:37-40`) are not protected by any locking mechanism. Loading a study in one tab while another tab is mid-analysis will silently replace the study object.

**Risk**: Data corruption and incorrect results without any error message.

**Recommendation**: At minimum, add a mutex on the module-level state. Return HTTP 409 (Conflict) if a study load is attempted while another is in progress.

---

### 33. requirements.txt includes matplotlib but it is not installed

**Problem**: `requirements.txt:53` lists `matplotlib>=3.7.0` but MEMORY.md documents that matplotlib is NOT installed in the venv, and `PDFReporter` import is wrapped in try/except to prevent crashes.

**Evidence**: `requirements.txt:53`: `matplotlib>=3.7.0`. But the MEMORY.md states: "Backend matplotlib not installed in venv -> PDFReporter import made lazy (try/except)." The `infrastructure/file_handlers/__init__.py` wraps the import.

**Risk**: Inconsistency between requirements.txt and the actual venv. Running `pip install -r requirements.txt` would install matplotlib. Not running it means PDF export is silently broken.

**Recommendation**: Either install matplotlib properly and test PDF export, or remove it from requirements.txt and document that PDF export requires `pip install matplotlib reportlab`.

---

## G. The Rewrite Itself

### 34. v1.0 to v1.1 migration introduced 22+ bugs

**Problem**: The migration from v1.0 to v1.1 (documented in MEMORY.md) introduced malformed URLs in 22 places, broken session management, incorrect metadata access patterns, wrong focal length formulas, degenerate 3D reconstruction, and silent Pydantic default overrides.

**Evidence**: MEMORY.md "Bugs Found & Fixed" section documents 12 distinct categories of bugs introduced by the migration. The URL space bugs alone affected 22 API calls.

**Risk**: A full rewrite (v2.0) will introduce MORE bugs, not fewer. The v1.1 migration was a relatively conservative port -- mostly copying code with minor adjustments. A rewrite changes everything simultaneously, making it impossible to isolate regressions.

**Recommendation**: Instead of a rewrite, consider incremental improvements: (a) fix the module-level state issue, (b) split api.ts, (c) add persistence, (d) add tests. Each change can be verified independently. A "rewrite" should be reserved for when the architecture fundamentally cannot support a required feature.

---

### 35. Test coverage is insufficient to verify a rewrite

**Problem**: The test suite covers only the old calculators, not the actual production code.

**Evidence**:
- `test_qfr_calculator.py`: Tests `qfr_calculator.py` (OLD), not `qfr_calculator_3d.py` (PRODUCTION)
- Zero frontend tests (0 files)
- `test_rws_calculator.py`: 449 lines (adequate for RWS)
- `test_qca_calculator.py`: 471 lines (adequate for QCA)
- No integration tests for the HTTP API (the integration test files exist but test at the use-case level, not the route level)

**Risk**: You cannot verify that a rewrite produces the same results as v1.1 because there is no test suite that defines "correct." You will be comparing manual observations against manual observations.

**Recommendation**: Before starting the rewrite, create a "golden test" suite: (a) load a known DICOM, (b) segment frame N with engine X, (c) calculate QCA, (d) calculate RWS for beat range [A, B], (e) assert specific numerical results. This gives you a ground truth to verify v2 against.

---

### 36. Who maintains v2?

**Problem**: CONTEXT.md is written as a specification for a rewrite, but does not address who will implement and maintain v2.

**Evidence**: The current codebase has a single contributor (based on git history). The CONTEXT.md document is ~1100 lines of specification. The actual codebase is ~15000+ lines of Python and ~10000+ lines of TypeScript.

**Risk**: A single-developer rewrite of a 25000+ line codebase is a 6-12 month project. During that time, v1.1 receives no maintenance. If the developer leaves, the project has an unfinished rewrite AND an unmaintained v1.1.

**Recommendation**: Do not rewrite. Improve incrementally. The codebase is functional, the algorithms work (with documented limitations), and the architecture, while imperfect, supports the current feature set. Address the critical issues (module-level state, persistence, tests) without rewriting everything.

---

## H. Contradictions Between CONTEXT.md and Code

### 37. CONTEXT.md says SAM2/SegFormer/HRNet are "stubs"

**Contradiction**: CONTEXT.md section 9.3 says SAM2 is "Not yet integrated." But `sam2_engine.py` (329 lines) is a fully implemented inference pipeline with preprocessing, point prompting, and post-processing. It is NOT a stub. However, it depends on a model file (`best_model.pth`) that does not exist in the repository, making it a Schrodinger's engine -- simultaneously implemented and non-functional.

**Evidence**: `sam2_engine.py` has complete `predict()` method (lines 146-299). `segformer_engine.py` and `hrnet_engine.py` are actual stubs (minimal placeholder classes). The localization route (`localization_routes.py:170`) explicitly labels itself as a stub.

---

### 38. CONTEXT.md says 16 route files, actual count is 18

**Contradiction**: CONTEXT.md section 3.3 says "16+ route files." The actual count is 18 route files in `presentation/routes/`.

**Evidence**: `ls presentation/routes/*.py | wc -l` = 18 files (including `__init__.py`). The 17 actual route files are: annotation, auth, calibration, dicom, export, health, localization, mask_edit, motion, qca, qfr, reconstruction, report, rws, segmentation, tracking, training_export.

---

### 39. CONTEXT.md says "12 Panel components," actual count is 11

**Contradiction**: CONTEXT.md section 3.2 says "Panels/ (12 components)". The actual `src/components/Panels/` directory has 11 components: CalibrationPanel, ECGPanel, ExportPanel, MetadataDisplay, QCAPanel, QFRPanel, ReportPanel, RWSPanel, SegmentationPanel, SeriesPicker, TrainingExportPanel.

---

### 40. CONTEXT.md lists auth endpoints that do not exist

**Contradiction**: CONTEXT.md section 5.14 describes "Register, login, logout, verify" endpoints. The actual `auth_routes.py` only has login, logout, and me (get current user) -- no register and no verify.

---

### 41. Docker references Dockerfiles that may not exist

**Contradiction**: `docker-compose.yml` references `docker/frontend/Dockerfile` and `docker/backend/Dockerfile`. The directories exist but the Dockerfiles were not verified.

---

### 42. CONTEXT.md claims IQR and temporal outlier methods for RWS

**Contradiction**: CONTEXT.md section 5.9 lists "IQR" and "temporal" outlier methods. The `rws_calculator.py:55-61` defines these enum values but `_calculate_position_rws_robust` (line 580-586) maps IQR, temporal, and everything else to the Hampel filter implementation:
```python
else:
    # HAMPEL is the default for all other methods
    return self._calculate_position_rws_hampel(diameters, frame_indices, position)
```
IQR and temporal are listed as options but secretly run Hampel. The user thinks they are choosing a different algorithm.

**Risk**: This is a silent feature lie. The UI presents options that do not work as described.

---

### 43. Missing endpoint: stereo reconstruction

**Contradiction**: CONTEXT.md section 7.2 does not list the `/reconstruction/` routes, but `reconstruction_routes.py` exists and is registered in `main.py`. This is an undocumented API surface.

---

## Summary: Top 5 Actions Before Any Rewrite

1. **Add tests for the production QFR calculator** (`qfr_calculator_3d.py`) with known-good phantom data. This is the most clinically critical untested code.

2. **Fix module-level state** in `dicom_routes.py`. Replace with a proper session store. This makes the backend testable, concurrent-safe, and prepares for persistence.

3. **Add persistence** (SQLite). Save segmentation results, calibration, and RWS results to disk. This is the number one user complaint waiting to happen.

4. **Do not rewrite**. The v1.0-to-v1.1 migration introduced 22+ bugs. A full rewrite will introduce more. Improve incrementally: split api.ts, consolidate stores, add WebSocket, add tests.

5. **Add disclaimers** throughout the application: "For research use only. Not validated for clinical decision-making." The QFR and RWS algorithms are unvalidated against clinical gold standards.

---

*End of Devil's Advocate Review. February 2026.*
