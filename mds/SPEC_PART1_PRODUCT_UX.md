# SPEC Part 1: Product & UX Specification

> Definitive specification for Coronary RWS Analyser v2.0 frontend. A developer should be able to build the entire user-facing application from this document without asking questions.

---

## Table of Contents

1. [Product Vision & Scope](#1-product-vision--scope)
2. [Feature Priority Matrix](#2-feature-priority-matrix)
3. [User Flows (Definitive)](#3-user-flows-definitive)
4. [UI Layout Specification](#4-ui-layout-specification)
5. [Component Specification](#5-component-specification)
6. [Interaction Design](#6-interaction-design)
7. [Design System](#7-design-system)
8. [State Management Specification](#8-state-management-specification)
9. [Accessibility Specification](#9-accessibility-specification)
10. [Responsive Design Specification](#10-responsive-design-specification)
11. [Error & Edge Case Handling (UI)](#11-error--edge-case-handling-ui)

---

## 1. Product Vision & Scope

### 1.1 Vision Statement

Coronary RWS Analyser is a desktop research tool that enables cardiovascular researchers and interventional cardiologists to quantify coronary artery wall strain (RWS) and estimated flow ratio (QFR) from standard X-ray angiography DICOM cine loops, replacing manual measurement with reproducible, algorithm-driven analysis -- all within a single application, without requiring external PACS integration or cloud services.

### 1.2 Target Users

**Persona 1: Interventional Cardiologist (Dr. Yilmaz)**
- Age 38-55. Works in a catheterization lab. Performs 400+ PCIs per year.
- Goal: Quickly assess whether a lesion is hemodynamically significant to decide PCI vs. medical therapy.
- Pain point: Invasive FFR requires a pressure wire, adds 10-15 minutes to the procedure. Wants a non-invasive screen.
- Usage pattern: 5-10 minute sessions. Loads a cine loop, segments, measures QFR. Exports a result to paste into clinical notes.
- Technical skill: Low. Expects single-click workflows. Will not debug errors.

**Persona 2: Cath Lab Technician (Ayse)**
- Age 25-40. Operates the angiography system, archives DICOM files.
- Goal: Pre-process angiograms before the physician reviews. Segment vessels, check calibration, generate a preliminary report.
- Pain point: Manual frame-by-frame analysis in legacy QCA software is tedious.
- Usage pattern: 15-30 minute sessions. Loads multiple studies sequentially. Uses tracking to auto-analyze multiple beats.
- Technical skill: Moderate. Comfortable with DICOM tools. Will use keyboard shortcuts.

**Persona 3: Cardiovascular Researcher (Mehmet)**
- Age 25-35. PhD student or postdoc. Validating RWS as a biomarker for vulnerable plaque.
- Goal: Analyze 50-200 studies with consistent parameters. Export structured data for statistical analysis.
- Pain point: Needs reproducible, batch-capable analysis with full parameter control. Needs to export raw data (diameter profiles, frame-level measurements) not just summary statistics.
- Usage pattern: 1-2 hour sessions. Deep analysis with mask editing, outlier method comparison, per-beat RWS.
- Technical skill: High. Will use every feature. Will want to understand algorithm parameters.

### 1.3 Core Value Proposition

1. **RWS measurement** is the primary differentiator. No existing commercial software calculates Radial Wall Strain from angiographic cine loops. This is a novel biomarker.
2. **QFR estimation** from dual-projection angiography, without an invasive pressure wire, using the Gould/Young-Tsai hemodynamic model.
3. **Integrated pipeline**: DICOM load, ECG extraction, segmentation, QCA, RWS, QFR, and export in one application -- no switching between tools.
4. **ECG-gated analysis**: Cardiac phase boundaries from embedded ECG or motion signal ensure physiologically meaningful measurements.

### 1.4 Non-Goals

- **NOT a PACS viewer**. Does not integrate with hospital PACS systems, DICOM networking (C-FIND, C-MOVE, WADO), or HL7/FHIR interfaces.
- **NOT a clinical device**. This is a research tool. It is NOT FDA-cleared, CE-marked, or validated for clinical decision-making. All exports must carry a "For Research Use Only" disclaimer.
- **NOT multi-user**. The backend serves a single user at a time. No user accounts, no role-based access, no concurrent sessions. Authentication is out of scope for v2.
- **NOT real-time**. Does not process live fluoroscopy feeds. Works only with saved DICOM files.
- **NOT a training platform**. The training export feature supports ML model development but the application does not train models or manage datasets.
- **NOT mobile-first**. Mobile/tablet layouts are a P2 concern. The primary target is a 1920x1080 desktop display.

### 1.5 Success Metrics

| Metric | v1.1 Baseline | v2 Target |
|--------|--------------|-----------|
| Time to first RWS result (from DICOM load) | ~5 min (manual) | < 2 min (guided workflow) |
| Time to QFR result (from two projections) | ~10 min | < 5 min |
| Crash/data loss incidents per 100 sessions | Unknown (no persistence) | 0 (all state persisted) |
| Frame-to-frame analysis latency (tracking+seg+QCA) | ~2-3s/frame | < 1.5s/frame |
| Test coverage (frontend) | 0% | > 60% (stores + critical paths) |
| Test coverage (backend domain services) | ~40% | > 80% |
| Accessibility conformance | Non-compliant | WCAG 2.1 AA (except canvas) |

---

## 2. Feature Priority Matrix

### 2.1 P0 -- Must-Have for Launch

Every feature listed here must be fully functional before v2 ships.

| # | Feature | Description | Acceptance Criteria |
|---|---------|-------------|---------------------|
| P0-1 | DICOM Load | Upload or open a multi-frame DICOM XA file. | Loads any valid multi-frame XA DICOM. Shows privacy dialog. Extracts frames, metadata, ECG, and frame rate. Displays first frame within 3 seconds. |
| P0-2 | Frame Viewer | Display DICOM frames with zoom, pan, and playback. | Renders at native DICOM resolution. Zoom 0.1x-10x centered on cursor. Pan via drag. Play/pause/step at 0.25x-2x speed. Frame counter visible. |
| P0-3 | nnU-Net Segmentation | Segment coronary arteries using nnU-Net (ROI, wide, fullframe). | Returns binary mask within 5 seconds on GPU, 15 seconds on CPU. Mask overlaid on viewer. Supports ROI-based and fullframe modes. |
| P0-4 | AngioPy Segmentation | Seed-guided segmentation using AngioPy U-Net. | Requires 2-10 seed points. Returns binary mask + probability map. Supports hybrid roi+angiopy mode. |
| P0-5 | Centerline Extraction | Extract vessel centerline from segmentation mask. | Returns ordered proximal-to-distal point sequence. Supports skeleton and distance transform methods. |
| P0-6 | QCA Calculation | Quantitative Coronary Analysis along centerline. | Gaussian subpixel fitting. Returns 50-point diameter profile, MLD, proximal/distal reference diameters, %DS, lesion length. All values in mm when calibrated. |
| P0-7 | Calibration | Pixel-to-mm spatial calibration. | From DICOM header, catheter (4-8 French), manual entry, or automatic from mask. Persists across frames. Yellow warning when uncalibrated. |
| P0-8 | ECG Extraction | Parse ECG from DICOM WaveformSequence. | Extracts signal, detects R-peaks (Pan-Tompkins), computes beat boundaries as frame indices, shows heart rate. |
| P0-9 | R-Peak Editing | Manual correction of R-peak positions. | Add, remove, move peaks. Recalculates beat boundaries immediately. Visual feedback on ECG display. |
| P0-10 | Motion Signal | Farneback optical flow cardiac phase detection. | Alternative to ECG. Returns motion magnitude signal, detects peaks, computes beat boundaries. Peak editing (add, remove, move). |
| P0-11 | RWS Calculation | Radial Wall Strain for a cardiac beat. | Input: frame range + outlier method. Segments + QCA each frame. Returns MLD/proximal/distal RWS with interpretation (normal/intermediate/elevated/high-risk). Color-coded display. |
| P0-12 | RWS Results Management | Track multiple RWS results. | Accumulate per-beat results. Delete individual results. Show summary statistics. Vessel label per result. |
| P0-13 | Mask Editing | Edit segmentation masks manually. | Brush, eraser, smart brush, flood fill. Undo/redo (20 levels). Save/cancel. Re-extract centerline and re-run QCA after edit. |
| P0-14 | QFR Dual-Projection | 3D QFR from two angiographic views. | Load two projections. Calibrate each. Segment each. TIMI frame count. 3D reconstruction. Gould/Young-Tsai QFR. Three modes (fQFR, cQFR, aQFR). |
| P0-15 | 3D Mesh Viewer | Visualize reconstructed vessel in 3D. | Triangle mesh with QFR heatmap coloring. Orbit, zoom, pan controls. Auto-fit camera. |
| P0-16 | Data Export | Export analysis results. | CSV, JSON formats for QCA and RWS data. Includes frame indices, diameter values, RWS per beat. |
| P0-17 | Session Persistence | Save/restore analysis state. | All segmentation masks, QCA results, RWS results, calibration persisted to disk (SQLite or file). Restore on app reopen. |
| P0-18 | Privacy Dialog | HIPAA-compliant anonymization prompt. | Hard gate before any DICOM upload. Anonymize or keep original. Cannot be skipped. |
| P0-19 | Research Disclaimer | "For Research Use Only" notice. | Displayed on launch, on every export/report, and in the status bar. Persistent. |
| P0-20 | Keyboard Shortcuts | Full keyboard control for power users. | All tool switches, playback, frame navigation, zoom accessible via keyboard. Help overlay with `?`. |
| P0-21 | Theme System | Light/dark/system theme. | Persisted preference. Instant switch. Correct contrast ratios in both modes. |
| P0-22 | Error Recovery | Graceful handling of all failures. | ML OOM: catch, free GPU, show message. Backend crash: detect, show reconnect prompt. No silent failures. |

### 2.2 P1 -- Should-Have (First Major Update)

| # | Feature | Description | Acceptance Criteria |
|---|---------|-------------|---------------------|
| P1-1 | ROI Tracking (CSRT) | Track vessel ROI across frames. | Initialize from ROI. Propagate forward/backward. Confidence threshold. Auto seg+QCA per tracked frame. |
| P1-2 | PDF Report | Generate clinical summary PDF. | Patient info (anonymized if requested), frame images, QCA charts, RWS results, ECG timeline. |
| P1-3 | Series Picker | Browse and select DICOM series from folder. | Parse DICOM headers client-side. Show thumbnails, angles, frame counts. Dual-selection for QFR. |
| P1-4 | XLSX Export | Excel format export with multiple sheets. | QCA sheet, RWS sheet, metadata sheet. Formatted for clinical review. |
| P1-5 | WebSocket Progress | Real-time progress for long operations. | Progress bar with percentage for segmentation, tracking, reconstruction. Cancel button for all. |
| P1-6 | Morphological Mask Ops | Dilate, erode, fill holes, remove islands, smooth. | Each operation processes the current mask server-side. Preview before apply. |
| P1-7 | QCA Marker Dragging | Adjust reference diameter positions on viewer. | Drag proximal/distal markers along centerline. Snap to nearest centerline point. Live diameter update. |
| P1-8 | Multi-Series Support | Handle DICOM files with multiple series. | Detect multi-series on load. Show series picker. Allow switching between series. |
| P1-9 | Audit Trail | Log all analysis actions. | JSON file logging: timestamp, action, parameters, results. One log per session. Exportable. |

### 2.3 P2 -- Nice-to-Have (Future)

| # | Feature | Description |
|---|---------|-------------|
| P2-1 | SAM2 Integration | Segment Anything Model 2 for interactive segmentation. Requires fine-tuned model weights and stable API. Currently dead code -- remove from v2 and reintegrate when model and API are production-ready. |
| P2-2 | SegFormer Integration | Transformer semantic segmentation. Stub only -- defer until model is trained and validated. |
| P2-3 | HRNet Vessel Classification | Vessel type identification (LAD, LCx, RCA). Stub only -- defer until model is trained. |
| P2-4 | Mobile/Tablet Layout | Responsive design for screens < 1024px. |
| P2-5 | Training Dataset Export | Frame + mask pairs for ML training. |
| P2-6 | Study Browser | Previously loaded studies list with metadata. Requires persistence layer. |
| P2-7 | Batch Analysis | Process multiple studies sequentially with consistent parameters. |
| P2-8 | Advanced Mask Tools | Contour deformation, edge snapping, region growing, magic wand, temporal interpolation. |
| P2-9 | Segmentation History | Per-frame mask version history. Revert to previous segmentation. |
| P2-10 | Docker/Web Deployment | Multi-container deployment with Nginx reverse proxy. |

### 2.4 Decisions on Devil's Advocate Concerns

**Stub Engines (SAM2, SegFormer, HRNet)**:
- **Decision**: P2. Remove from v2 launch. SAM2 (329 lines) depends on private APIs and a missing model file. SegFormer and HRNet are empty stubs. Including non-functional engines in the UI misleads users and wastes engineering effort. Re-introduce each engine only when: (a) model weights exist and are tested, (b) the engine has unit tests with known-good outputs, (c) the engine UI is distinct from existing engines.

**Auth System**:
- **Decision**: Remove entirely from v2. This is a single-user desktop/research tool. The JWT/refresh/session infrastructure in v1.1 is dead code (all endpoints return "not implemented"). Adding auth before adding a user model and database is premature. If multi-user becomes a requirement, it should be designed alongside the persistence layer, not grafted on.

**17 Zustand Stores**:
- **Decision**: Consolidate to 8 stores. See Section 8.

**api.ts (2600 lines)**:
- **Decision**: Split into per-domain modules. See Section 8 (API client architecture).

**IQR/Temporal Outlier Methods**:
- **Decision**: In v1.1, IQR and temporal methods secretly run Hampel. In v2, either implement them properly or remove them from the UI. Do not present options that do not work as described.

---

## 3. User Flows (Definitive)

### Flow 1: First-Time App Launch through First Segmentation

**Trigger**: User launches the application for the first time.

**Steps**:
1. Application opens. Main window renders (1400x900 minimum).
2. Viewer area shows the empty state: centered text "Coronary RWS Analyser" + subtitle "Open a DICOM file to begin analysis" on a dark background.
3. Status bar shows "For Research Use Only" disclaimer on the left. Connection status "Backend: Connected" (green dot) or "Backend: Disconnected" (red dot) on the right.
4. If backend is disconnected: show a non-dismissible banner at the top of the viewer: "Cannot connect to analysis backend. Please ensure the Python server is running on port 8000." with a "Retry" button.
5. User clicks "Open File" button in the toolbar (folder icon) OR presses Ctrl+O OR drags a `.dcm` file onto the viewer.
6. **Privacy Dialog** appears (modal, z-index 100, blocks all interaction):
   - Title: "Patient Privacy"
   - Body: "This DICOM file may contain patient identifiers. Would you like to anonymize the data before processing?"
   - Three buttons: [Cancel] (gray, left) | [Keep Original] (gray, center) | [Anonymize] (blue, right with shield icon)
   - Escape key maps to Cancel.
7. User clicks [Keep Original] or [Anonymize].
8. System resets all analysis stores (segmentation, QCA, RWS, ECG, calibration, tracking, motion).
9. Loading indicator appears: spinner overlaid on viewer center + "Loading DICOM..." text.
10. Backend processes DICOM file: parses, extracts frames, metadata, ECG.
11. On success:
    - Viewer renders frame 0.
    - Header updates: shows "512x512 | 120 frames | 15 fps".
    - Right panel auto-selects "Seg + QCA" tab.
    - If DICOM has pixel spacing: calibration auto-set, badge shows "Cal: DICOM".
    - If DICOM has ECG: ECG panel populates, R-peaks shown, beat boundaries computed.
    - Status bar: "Ready -- Frame 0/119".
    - Motion signal calculation begins automatically in background.
12. User selects segmentation engine in the right panel (default: nnU-Net).
13. If engine requires ROI: user presses [B] to activate ROI tool, clicks on vessel in viewer to place a 160x160 fixed ROI.
14. If engine requires seeds: user presses [S] to activate Seed tool, clicks 2+ points along the vessel (proximal to distal).
15. User clicks [Segment] button (or presses Enter while panel is focused).
16. Spinner appears on button. "Segmenting..." text.
17. Backend returns mask + centerline. QCA auto-calculates.
18. Viewer shows: semi-transparent mask overlay (green-ish) + centerline (thin line) + diameter markers.
19. Right panel shows QCA results: MLD, DS%, reference diameters, diameter profile mini-chart.

**Error Paths**:
- Step 5 (file selection): If file is not a valid DICOM, show toast: "Invalid file format. Please select a DICOM (.dcm) file." Severity: error. Recovery: user selects a different file.
- Step 10 (DICOM parsing): If DICOM has 0 frames or is single-frame: show toast: "This DICOM file has only 1 frame. Coronary analysis requires a multi-frame cine loop (NumberOfFrames > 1)." Severity: warning. Recovery: user loads a different file.
- Step 10: If DICOM modality is not "XA": show toast: "This DICOM file has modality '{modality}'. This tool is designed for X-ray Angiography (XA). Results may be unreliable." Severity: warning. Allow user to proceed.
- Step 17 (segmentation): If ML engine fails (OOM, model not found): show toast with specific message. "Segmentation failed: insufficient GPU memory. Try CPU mode or a smaller ROI." Severity: error. Recovery: user adjusts parameters or switches engine.
- Step 17: If mask is empty (no vessel detected): show toast "No vessel detected in the selected region. Try adjusting the ROI or seed points." Severity: warning. Recovery: user re-positions ROI/seeds.

**Exit Points**:
- Step 6: Cancel in Privacy Dialog aborts the entire load. Returns to empty state.
- Step 13-14: User can press Escape to deselect the tool without completing.
- Any step: User can load a new file (triggers Privacy Dialog again, resets all state).

---

### Flow 2: Complete RWS Analysis

**Trigger**: DICOM is loaded (Flow 1 complete or resumed from persistence).

**Steps**:
1. **ECG/Motion Verification**: User navigates to the right panel. If ECG is available, the ECG mini-display shows the waveform with R-peak markers. If no ECG, the motion signal panel shows the optical flow signal with detected peaks.
2. **R-Peak/Motion Peak Editing** (optional): User clicks "Edit" on the ECG panel to enter R-peak editing mode. Left-click adds a peak; right-click removes; drag moves. User verifies beat boundaries look correct. Clicks "Done" to exit editing.
3. **Navigate to Target Frame**: User plays the video or uses arrow keys to navigate to a frame showing the target lesion clearly (typically end-diastole).
4. **Segment Target Frame**: If not already segmented, user segments the current frame (see Flow 1, steps 12-18).
5. **Verify QCA**: User checks QCA results in the right panel. If MLD or reference diameters look wrong, user edits the mask (Flow 4) or adjusts QCA marker positions by dragging on the viewer.
6. **Set Calibration**: If not auto-calibrated from DICOM, user opens the Calibration section in the right panel. Selects catheter size (5F-8F) and clicks "Calibrate from Mask" (auto-measures catheter width) or enters pixel spacing manually.
7. **Switch to RWS Tab**: User clicks the "RWS" tab in the right panel.
8. **Select Frame Range**: Two options:
   a. **From ECG beat**: Click a beat button (B1, B2, B3...) to auto-fill start/end frames from R-peak boundaries.
   b. **Manual**: Enter start and end frame numbers in the input fields. Or right-click on the viewer at the desired start frame and select "Set as RWS Start Frame," then navigate to end frame and right-click "Set as RWS End Frame."
9. **Select Outlier Method**: Dropdown: None, Hampel (default), Double Hampel. (IQR and Temporal removed unless properly implemented.)
10. **Select Vessel**: Dropdown: LAD, LCX, RCA, Diagonal, OM, IM, PLA, PDA, or unset.
11. **Calculate**: Click the green [Calculate RWS] button.
12. **Processing**: Button shows spinner. Status bar shows "Calculating RWS... Frame X/Y". For each frame in the range, the backend: segments (if not cached), extracts centerline, calculates QCA, collects diameter series.
13. **Result Displayed**: A result card appears:
    - Large MLD RWS value (e.g., "10.4%") color-coded:
      - Green (<8%): "Normal"
      - Amber (8-12%): "Intermediate"
      - Orange (12-14%): "Elevated"
      - Red (>14%): "High Risk"
    - Smaller values: Proximal RWS, Distal RWS, Average RWS.
    - Frame links: "Dmin: Frame 45" and "Dmax: Frame 52" (clickable -- navigates viewer to that frame).
    - Beat number, vessel label, frame range, outlier method.
14. **Repeat for Additional Beats**: User selects a different beat (B2, B3...) and clicks Calculate again. New result cards accumulate in a scrollable list.
15. **Summary**: After 2+ results, a summary section appears at the bottom: mean MLD RWS, std, min, max across all beats.
16. **Export**: User switches to Export tab and exports RWS + QCA data as CSV or JSON.

**Error Paths**:
- Step 8: If start frame >= end frame: disable Calculate button, show inline text "End frame must be after start frame."
- Step 8: If frame range < 5 frames: show warning "Frame range is very short (N frames). RWS may be unreliable with fewer than 10 frames."
- Step 12: If segmentation fails on any frame in the range: show toast "Segmentation failed on frame X. RWS calculation aborted. Try segmenting frame X manually first." Severity: error. Recovery: user segments problematic frames manually before re-running.
- Step 12: If centerline extraction fails (empty mask): skip that frame, note in results as "N frames skipped."
- Step 13: If all diameters are identical (RWS = 0%): show result with note "Zero strain detected. This may indicate a rigid stent or segmentation issue."

**Exit Points**:
- Any step: User can switch to a different tab or load a new file.
- Step 11: Cancel button appears during calculation. Clicking it aborts (partial results discarded).

---

### Flow 3: Complete QFR Analysis (Dual-Projection)

**Trigger**: User has access to two DICOM cine loops of the same coronary vessel from different viewing angles (minimum 25 degrees separation).

**Steps**:
1. **Enter QFR Mode**: User clicks the QFR mode toggle in the toolbar (or clicks the "QFR" tab in the right panel and toggles "Dual-Projection Mode" on).
2. **Layout Change**: The main viewer is replaced by a dual-projection layout: two side-by-side viewer panels (Projection 1 on the left, Projection 2 on the right).
3. **Load Projection 1**: User clicks [Load P1] button in the QFR panel. The Series Picker modal opens.
4. **Series Picker**: User clicks "Select Folder," chooses a folder containing DICOM files. The picker scans the folder (client-side DICOM header parsing), displays a 3-column grid of series with thumbnails, angle labels (RAO 30 / CRA 15), and frame count badges.
5. **Select P1**: User clicks a series. It is assigned to P1 (blue outline). The series is uploaded to the backend. P1 viewer populates with frame 0.
6. **Load Projection 2**: User clicks [Load P2]. Same picker, this time in P2-selection mode (teal outline). User selects a series with a sufficiently different viewing angle.
7. **Angle Validation**: System displays the angular separation between P1 and P2 (e.g., "Angular separation: 42 degrees"). If < 25 degrees: yellow warning "Insufficient angular separation. Reconstruction may be unreliable. Minimum recommended: 25 degrees."
8. **Segment P1**: User places 2+ seed points on P1's vessel in the P1 viewer. Clicks [Segment P1] in the QFR panel. Mask + centerline overlay appear on P1.
9. **Segment P2**: Same process for P2.
10. **Calibrate P1**: User selects catheter size (default 6F) and clicks [Calibrate P1]. Or the system auto-calibrates from DICOM pixel spacing if available.
11. **Calibrate P2**: Same for P2.
12. **TIMI Frame Count (for cQFR/aQFR)**: User navigates P1's frames to find where contrast first enters the vessel. Clicks [Set T-Start]. Then navigates to where contrast reaches the distal landmark. Clicks [Set T-End]. The TIMI frame count is displayed (T-end minus T-start).
13. **3D Reconstruct**: All prerequisites met (both projections segmented, calibrated, >= 25 degree separation). User clicks the purple gradient [3D Reconstruct] button.
14. **Processing**: Button shows spinner. "Reconstructing... Matching epipolar points... Triangulating... Computing QFR..." Status updates via progress bar.
15. **Result Display**:
    - Large QFR value: e.g., "QFR = 0.76" color-coded:
      - Green (>= 0.80): "Not significant"
      - Yellow (0.75-0.79): "Gray zone"
      - Red (< 0.75): "Hemodynamically significant"
    - Mode selector: fQFR | cQFR | aQFR (radio buttons). Switching mode recalculates QFR with different flow assumptions.
    - Metrics: vessel length (mm), pressure drop (mmHg), flow velocity (m/s), stenosis (%).
16. **3D View Toggle**: User clicks [View 3D] to switch from dual-projection view to the Three.js 3D mesh viewer. The vessel is displayed as a tube with QFR heatmap coloring (red = low QFR, green = normal). Orbit with left-drag, zoom with scroll, pan with right-drag.
17. **Export**: QFR results included in the standard export.

**Error Paths**:
- Step 4 (folder scan): If no valid DICOM files found: show message in picker "No DICOM files found in this folder."
- Step 7 (angle validation): If angles are identical (0 degree separation): block reconstruction, show error "Both projections have identical viewing angles. 3D reconstruction requires different angles."
- Step 8-9: If segmentation fails: standard segmentation error handling (see Flow 1).
- Step 13: If reconstruction fails (degenerate geometry, insufficient matched points): show error "3D reconstruction failed: {reason}. Check that both projections show the same vessel segment and have correct calibration."
- Step 13: If reprojection error is high (> 5 pixels): show warning "Reconstruction quality is low (reprojection error: X px). Results may be inaccurate."

**Exit Points**:
- Step 1: Toggle QFR mode off to return to normal viewer.
- Step 3-6: Close Series Picker with X button.
- Step 13: Cancel button during reconstruction.
- Any step: Loading a new file in normal mode exits QFR mode.

---

### Flow 4: Mask Refinement Cycle

**Trigger**: A segmentation mask exists for the current frame but needs manual correction.

**Steps**:
1. **Enter Edit Mode**: User clicks [Edit Mask] button in the Segmentation panel. This button is only enabled when a mask exists for the current frame.
2. **Mode Transition**: The system cursor is hidden. A floating toolbar appears centered at the top of the viewer (glass-morphism style). The toolbar contains: Brush, Quick-Brush, Eraser, Smart-Brush, Smart-Eraser, Lasso, Polygon, Magic Wand, Contour Edit + separator + Undo, Redo + separator + Morphological Ops dropdown + separator + size slider + separator + [Save] (green), [Cancel] (red).
3. **Select Tool**: User clicks Brush (or presses B). A circle cursor follows the mouse, sized to the brush radius.
4. **Paint Mask**: User clicks and drags on the viewer to add pixels to the mask. Painted area appears immediately as a colored overlay.
5. **Erase**: User switches to Eraser (E key). Click and drag removes mask pixels.
6. **Adjust Size**: User presses `]` to increase or `[` to decrease brush/eraser size by 2px.
7. **Smart Brush** (optional): User switches to Smart Brush (D key). Painting snaps to image edges (gradient-aware). Edge sensitivity adjustable in toolbar settings.
8. **Undo**: User presses Ctrl+Z to undo the last stroke. Up to 20 undo levels.
9. **Morphological Ops** (optional): User clicks the Morph dropdown and selects "Fill Holes" to close small gaps in the mask. Or "Smooth" to soften jagged edges.
10. **Re-Extract Centerline**: User clicks [Extract Centerline] in the toolbar (or it auto-runs on save). The system extracts a new centerline from the edited mask.
11. **Re-Calculate QCA**: User clicks [Recalculate QCA] (or it auto-runs). New diameter profile, MLD, and DS% are computed from the edited mask and new centerline.
12. **Save**: User clicks [Save] (green). The edited mask replaces the original in the segmentation store. Edit mode exits. Normal viewer interaction resumes.

**Error Paths**:
- Step 4: If the user paints outside the image bounds: strokes are clipped to image dimensions.
- Step 9: If morphological operation makes the mask empty: show warning "Operation removed all mask pixels. Undo to recover."
- Step 10: If centerline extraction fails (mask too small or fragmented): show warning "Could not extract centerline. The mask may be too small or fragmented."

**Exit Points**:
- Step 12: [Cancel] restores the original mask and exits edit mode.
- Any step: If the user changes frame during edit mode, the current working mask is auto-saved before loading the new frame's mask for editing.
- Pressing Escape exits edit mode (prompts: "Save changes?" with [Save], [Discard], [Cancel]).

---

### Flow 5: ROI Tracking with Auto-Analysis

**Trigger**: User wants to analyze vessel motion across multiple frames automatically.

**Steps**:
1. **Set ROI**: User activates ROI tool (B key), clicks on the vessel. A 160x160 fixed ROI appears.
2. **Segment Reference Frame**: User segments the current frame to establish the baseline.
3. **Enable Track Mode**: User clicks [Track] toggle in the playback controls. Controls turn orange, "TRACK MODE" badge appears with pulse animation.
4. **Enable Auto-Seg**: User checks "Auto Seg+QCA" checkbox in the tracking section.
5. **Set Confidence Threshold**: Slider in tracking settings (default 0.6).
6. **Propagate**: User clicks [Propagate Forward] (or uses playback to step frame-by-frame). For each frame:
   a. CSRT tracker updates ROI position.
   b. Confidence score displayed (0.0-1.0).
   c. If Auto-Seg enabled: segmentation + QCA runs automatically.
   d. Results cached per frame.
7. **Progress Display**: Status bar shows "Tracking: Frame X/Y | Confidence: 0.85". Progress bar visible.
8. **Tracking Loss**: If confidence drops below threshold, tracking stops. Toast: "Tracking lost at frame X (confidence: 0.42). Re-initialize to continue." User can re-position ROI and re-initialize.
9. **Review Results**: After propagation, user reviews per-frame segmentation and QCA by stepping through frames. Can edit individual masks if needed.
10. **RWS from Tracked Data**: User switches to RWS tab. The frame range is auto-populated from the tracked range. Calculate RWS uses the cached per-frame QCA data.

**Error Paths**:
- Step 6a: If tracker initialization fails: toast "Tracker initialization failed. Ensure the ROI contains a visible vessel."
- Step 6c: If segmentation fails on a frame: log warning, skip that frame, continue tracking.
- Step 7: If propagation is taking too long (>60s): show "Still working..." message with option to cancel.

**Exit Points**:
- Step 3: Toggle Track Mode off at any time.
- Step 6: [Cancel] button stops propagation (partial results retained).
- Step 8: After tracking loss, user decides whether to re-initialize or accept partial results.

---

### Flow 6: Export & Report Generation

**Trigger**: User has completed analysis (QCA, RWS, and/or QFR results exist).

**Steps**:
1. **Open Export Panel**: User clicks the "Export" tab in the right panel.
2. **Select Format**: Radio buttons: CSV, JSON, XLSX (P1), PDF Report (P1).
3. **Select Data**: Checkboxes: Include QCA data, Include RWS data, Include metadata, Include QFR data (if available).
4. **Anonymize Option**: Checkbox: "Anonymize patient data in export" (default: checked).
5. **Click Export**: [Export] button.
6. **File Save Dialog**: Native OS save dialog appears. Default filename: `{PatientID}_{StudyDate}_analysis.{format}` (anonymized if selected: `ANON_{StudyDate}_analysis.{format}`).
7. **Export Processing**: Brief spinner. For CSV: single file with columns for frame index, diameter values, RWS per beat. For JSON: structured output matching the data model.
8. **Success**: Toast: "Export saved to {path}." Severity: success. Duration: 3 seconds.
9. **Footer in Export**: Every export file includes: "Generated by Coronary RWS Analyser v2.0 -- For Research Use Only. Not validated for clinical decision-making."

**Error Paths**:
- Step 6: If user cancels the save dialog: no export, no error message.
- Step 7: If export fails (disk full, permission denied): toast "Export failed: {reason}." Severity: error.
- Step 2: If no analysis data exists for the selected options: disable the Export button, show inline text "No {type} data available to export."

---

## 4. UI Layout Specification

### 4.1 Main Analysis View (Normal Mode)

```
+--------------------------------------------------------------+
| Header (48px h)                                               |
| [Logo] Coronary RWS Analyser  v2.0  | 512x512 120fr 15fps  |
|                                   [Theme] [Settings] [Full]  |
+----+-----------------------------------------------+---------+
| T  |                                               | Right   |
| o  |                                               | Panel   |
| o  |            Main Viewer                        | (320px) |
| l  |        (DICOM canvas layers)                  |         |
| b  |                                               | +-----+ |
| a  |                                               | |Tabs:| |
| r  |   Zoom: 100%                  Frame: 45/119   | |Seg  | |
|    |                                               | |RWS  | |
|(56 |-----------------------------------------------| |QFR  | |
| px)|  Chart Area (collapsible, 120-200px)          | |Info  | |
| w  |  [ECG] [QCA] [RWS] [Motion]  tab bar         | |Exprt| |
|    |  +------------------------------------------+ | |     | |
|    |  | Active chart content (e.g., ECG waveform)| | |     | |
|    |  +------------------------------------------+ | +-----+ |
|    |-----------------------------------------------| Panel   |
|    | Playback Controls (64px h)                    | content |
|    | [|<] [<] [>||] [>] [>|]  ---[]---  1.0x Loop | area    |
+----+-----------------------------------------------+---------+
| Status Bar (24px h)                                           |
| [*] For Research Use Only  |  Backend: Connected  |  ? Help  |
+--------------------------------------------------------------+
```

**Dimensions**:
- Total minimum window: 1200 x 800 px
- Default window: 1400 x 900 px
- Header: full width, 48px height, fixed
- Toolbar: 56px width, fills height between header and status bar, fixed left
- Right Panel: 320px width, fills height between header and status bar, fixed right
- Viewer: flex-1 (takes remaining width between toolbar and right panel), fills height minus header, chart area, playback controls, and status bar
- Chart Area: full viewer width, collapsible (0px when hidden, 120-200px when visible), below viewer
- Playback Controls: full viewer width, 64px height, below chart area
- Status Bar: full width, 24px height, fixed bottom

**Resize Behavior**:
- Right panel: fixed 320px, not resizable in v2 (resizable handle is P2)
- Chart area: collapsible via tab click (click active tab to collapse), minimum 120px when visible, maximum 200px
- Viewer: absorbs all remaining space, maintains aspect ratio of DICOM image (letterboxed with dark background)

### 4.2 QFR Dual-Projection Mode

```
+--------------------------------------------------------------+
| Header (48px h)                                               |
+----+-----------------------------------------------+---------+
| T  | +---------------------+---------------------+ | Right   |
| o  | |  Projection 1       |  Projection 2       | | Panel   |
| o  | |                     |                     | | (320px) |
| l  | |  [DICOM + overlay]  |  [DICOM + overlay]  | |         |
| b  | |                     |                     | | QFR Tab |
| a  | |  RAO 30 / CRA 15    |  LAO 45 / CRA 0    | | active  |
| r  | |  Frame: 15/120      |  Frame: 22/150      | |         |
|    | +---------------------+---------------------+ |         |
|(56 |-----------------------------------------------| P1 info |
| px)| Synchronized Playback Controls (64px h)       | P2 info |
|    | [|<] [<] [>||] [>] [>|]  ---[]---  1.0x      | Results |
+----+-----------------------------------------------+---------+
| Status Bar (24px h)                                           |
+--------------------------------------------------------------+
```

- The two projection viewers split the available width equally (50/50).
- Each projection viewer has its own canvas layers (video + segmentation + annotation overlays).
- Angle labels and frame counters are overlaid on each projection viewer's top-left and top-right corners.
- Playback controls synchronize both projections: play/pause/step advances both. Individual frame offset possible via per-projection frame inputs in the QFR panel.

### 4.3 QFR 3D Mesh Mode

```
+--------------------------------------------------------------+
| Header (48px h)                                               |
+----+-----------------------------------------------+---------+
| T  |                                               | Right   |
| o  |         3D Vessel Mesh (Three.js)             | Panel   |
| l  |                                               | (320px) |
| b  |    [Orbit: drag  |  Zoom: scroll  |  Pan: R] |         |
| a  |                                               | QFR Tab |
| r  |         Vessel with QFR heatmap coloring      | active  |
|    |                                               |         |
|(56 |                                               | QFR val |
| px)|                                               | Metrics |
|    |                                               | Mode    |
|    |  [Back to Projections]                        | selector|
+----+-----------------------------------------------+---------+
| Status Bar (24px h)                                           |
+--------------------------------------------------------------+
```

- The 3D viewer fills the entire viewer area (no chart area, no playback controls).
- A [Back to Projections] button at the bottom-left returns to dual-projection mode.
- The QFR panel on the right shows all QFR metrics, mode selector, and vessel details.

### 4.4 Study Browser (P2)

```
+--------------------------------------------------------------+
| Header (48px h)                                               |
| [Logo] Coronary RWS Analyser  |  [New Analysis] [Settings]  |
+--------------------------------------------------------------+
|                                                               |
|  Recent Studies                                [Scan Folder]  |
|  +---+---------------------+--------+--------+-----------+   |
|  | # | Patient / Study     | Date   | Frames | Status    |   |
|  +---+---------------------+--------+--------+-----------+   |
|  | 1 | ANON_12345          | 2026-01| 120    | Analyzed  |   |
|  | 2 | ANON_67890          | 2026-01| 95     | Loaded    |   |
|  +---+---------------------+--------+--------+-----------+   |
|                                                               |
+--------------------------------------------------------------+
| Status Bar (24px h)                                           |
+--------------------------------------------------------------+
```

### 4.5 Z-Index Layer Map

| Z-Index | Element | Description |
|---------|---------|-------------|
| 0 | Video canvas | DICOM frame rendering |
| 1 | Segmentation canvas | Mask overlay + centerline |
| 2 | Annotation canvas | Seed points, ROI, measurements |
| 3 | Overlay canvas | ECG trace, frame info, diameter markers |
| 10 | Chart area | Below viewer, above playback |
| 20 | Toolbar tooltips | Appear on hover |
| 30 | Context menus | Appear on right-click |
| 40 | Mobile panel overlay | Backdrop + sliding panel |
| 50 | Mask edit toolbar | Floating toolbar during mask edit |
| 60 | Settings modal | Settings dialog |
| 70 | Series picker modal | DICOM series selection |
| 100 | Privacy dialog | Highest priority modal |
| 110 | Toast notifications | Top-right corner, above everything |

### 4.6 Focus Flow (Tab Order)

1. Header: Theme toggle, Settings button, Fullscreen button
2. Toolbar: Tool buttons (top to bottom)
3. Viewer: Tab-focusable (receives keyboard shortcuts when focused)
4. Chart area tab bar: ECG, QCA, RWS, Motion tabs
5. Playback controls: Previous, Play/Pause, Next, Speed, Loop
6. Right panel tab bar: Seg, RWS, QFR, Info, Export tabs
7. Right panel content: Form fields, buttons in reading order

Skip-to-content link: First focusable element in the DOM. Hidden until focused. Jumps to the viewer area.

---

## 5. Component Specification

### 5.1 Viewer Components

#### VideoPlayer

- **Props/Inputs**: `frameData: Map<number, ImageBitmap>`, `currentFrame: number`, `viewTransform: ViewTransform`, `overlays: OverlayConfig`, `annotationMode: AnnotationMode`
- **States**:
  - **Empty**: No DICOM loaded. Dark background, centered placeholder text.
  - **Loading**: Spinner centered on viewer. Text: "Loading DICOM..."
  - **Displaying**: Frame rendered. Optional overlays visible.
  - **Zoomed**: Zoom percentage indicator in top-right corner (visible for 2s after zoom change).
  - **Pan mode**: Cursor changes to grab (idle) / grabbing (dragging).
  - **ROI mode**: Crosshair cursor. ROI rectangle with dashed border and drag handles.
  - **Seed mode**: Crosshair cursor. Seed points as numbered colored circles (radius 6px, white border, sequential color).
  - **Mask edit**: System cursor hidden, custom circle cursor matching tool size. "MASK EDIT" indicator top-left.
  - **Track mode**: "TRACK MODE" pulsing badge, confidence percentage.
  - **Error**: Red banner across the top of the viewer with error message.
- **Outputs/Events**: `onFrameChange(index)`, `onViewTransformChange(transform)`, `onSeedPointAdd(point)`, `onSeedPointRemove(index)`, `onROIChange(bbox)`, `onContextMenu(event, imagePoint)`
- **Accessibility**: `role="application"`, `aria-label="DICOM image viewer, frame {current} of {total}"`, `tabindex="0"` for keyboard focus. Arrow key navigation for frame stepping when focused.

#### QFRModeViewer

- **Props/Inputs**: `projection1: ProjectionState`, `projection2: ProjectionState`, `syncPlayback: boolean`
- **States**:
  - **No projections**: Both panels show "Load Projection" placeholder.
  - **One projection**: Loaded panel shows frames, other shows placeholder.
  - **Both loaded**: Side-by-side viewers with independent overlays.
  - **Thumbnail mode**: When one projection is transferred to main viewer, a small thumbnail (120x120px) of the other projection appears in the main viewer's top-right corner.
- **Outputs/Events**: `onProjectionFrameChange(projectionId, frameIndex)`, `onSeedPointAdd(projectionId, point)`
- **Accessibility**: `role="group"`, `aria-label="Dual-projection QFR viewer"`. Each projection sub-viewer has `aria-label="Projection {1|2}, angle {angle}"`.

#### QFR3DViewer

- **Props/Inputs**: `mesh: { vertices: Float32Array, faces: Uint32Array, colors?: Float32Array }`, `qfrValue: number`
- **States**:
  - **Empty**: Dark background, "No 3D data" text.
  - **Loading**: Spinner + "Generating mesh..."
  - **Displaying**: Three.js canvas with vessel mesh. Orbit controls active.
  - **Heatmap on**: Mesh colored by QFR (red-yellow-green gradient).
  - **Heatmap off**: Uniform gray mesh.
- **Outputs/Events**: None (read-only visualization).
- **Accessibility**: `role="img"`, `aria-label="3D vessel reconstruction. QFR value: {value}. Use mouse to orbit, scroll to zoom."`. No keyboard alternative for orbit (acceptable for P0, P1 will add keyboard orbit with arrow keys).

#### ReportViewer (P1)

- **Props/Inputs**: `reportData: ReportData`, `format: 'pdf' | 'svg'`
- **States**: Empty, Loading, Displaying, Error.
- **Outputs/Events**: `onExport(format)`.

### 5.2 Panel Components

#### SegmentationPanel

- **Props/Inputs**: `currentFrame: number`, `segmentationData: FrameSegmentationData | null`, `availableEngines: EngineInfo[]`
- **States**:
  - **No file loaded**: Engine grid visible but [Segment] button disabled, muted text "Load a DICOM file first."
  - **Engine selected**: Selected engine card highlighted with engine-specific color border.
  - **Prerequisites unmet**: [Segment] button disabled, text below explains what is needed: "nnU-Net requires an ROI. Press B to draw one." or "AngioPy requires 2+ seed points. Press S to place them."
  - **Segmenting**: [Segment] button shows spinner, text "Segmenting...", button disabled.
  - **Segmentation complete**: Overlay toggle switches appear. Mask info shown (mask pixel count, centerline point count).
  - **Error**: Red text below button: "{error message}".
- **Outputs/Events**: `onSegment(engine, options)`, `onEngineChange(engine)`, `onOverlayToggle(layer, visible)`, `onEditMask()`, `onClearSegmentation()`
- **Accessibility**: Engine grid uses `role="radiogroup"`, each engine is `role="radio"` with `aria-checked`. [Segment] button has `aria-label="Segment current frame using {engine}"`. `aria-live="polite"` region for segmentation status.

Engine color mapping (badge accent):
| Engine | Color | Tailwind Class |
|--------|-------|---------------|
| nnunet | Blue | `blue-600` |
| nnunet-wide | Indigo | `indigo-600` |
| nnunet-fullframe | Emerald | `emerald-600` |
| angiopy | Blue | `blue-600` |
| roi+angiopy | Cyan | `cyan-600` |

#### QCAPanel

- **Props/Inputs**: `metrics: QCAMetrics | null`, `calibration: Calibration | null`
- **States**:
  - **No segmentation**: "Segment the current frame to see QCA results."
  - **No calibration**: Yellow warning bar: "No spatial calibration set. Values shown in pixels, not millimeters."
  - **Calculating**: Spinner.
  - **Results**: MLD value (red text, bold), DS% with severity badge, proximal reference (cyan text), distal reference (yellow text), interpolated RD, lesion length, diameter profile mini-chart.
  - **Error**: Error message text.
- **DS% Severity Badges**:
  - < 50%: "Mild" -- green badge
  - 50-69%: "Moderate" -- yellow badge
  - 70-89%: "Severe" -- orange badge
  - >= 90%: "Critical" -- red badge
- **Outputs/Events**: `onRecalculate(method, numPoints)`, `onMarkerDrag(markerType, newPosition)`
- **Accessibility**: Results announced via `aria-live="polite"` region. Each metric has `aria-label`: "Minimum lumen diameter: 1.23 millimeters."

#### RWSPanel

- **Props/Inputs**: `results: RWSResult[]`, `beats: BeatBoundary[]`, `ecgAvailable: boolean`
- **States**:
  - **No data**: "Segment and run QCA across a frame range, then calculate RWS."
  - **Frame range set**: Start/end inputs filled, [Calculate] enabled (green).
  - **ECG beats available**: Beat buttons (B1, B2, ...) visible above frame range inputs.
  - **Calculating**: Spinner, button disabled, status text "Calculating RWS... Frame X/Y".
  - **Single result**: Result card with color-coded MLD RWS.
  - **Multiple results**: Scrollable list of cards + summary section.
- **RWS Color Coding**:
  - < 8%: Green (#22c55e) -- "Normal"
  - 8-12%: Amber (#f59e0b) -- "Intermediate"
  - 12-14%: Orange (#f97316) -- "Elevated"
  - > 14%: Red (#ef4444) -- "High Risk"
- **Outputs/Events**: `onCalculate(startFrame, endFrame, options)`, `onDeleteResult(resultId)`, `onNavigateToFrame(frameIndex)`
- **Accessibility**: Result values announced: "MLD radial wall strain: 10.4 percent. Interpretation: intermediate."

#### QFRPanel

- **Props/Inputs**: `isQfrMode: boolean`, `projection1: ProjectionState`, `projection2: ProjectionState`, `qfrResult: QFRResult | null`
- **States**:
  - **QFR mode off**: Mode toggle showing "Standard" selected. Quick single-view QFR controls (P1 feature).
  - **QFR mode on, no projections**: Step-by-step instruction list: "1. Load Projection 1, 2. Load Projection 2, 3. Segment both, 4. Calibrate, 5. Reconstruct."
  - **P1 loaded**: P1 status card (green border) with angle, frame count, calibration status.
  - **Both loaded**: Both cards, angular separation display.
  - **Angle warning**: Yellow alert for < 25 degree separation.
  - **Both segmented + calibrated**: [3D Reconstruct] button enabled (purple gradient).
  - **Reconstructing**: Button spinner, "Reconstructing..." text.
  - **Result available**: Large QFR value, mode selector (fQFR/cQFR/aQFR), metrics table, [View 3D] button.
  - **3D view active**: [Back to Projections] button visible.
- **QFR Color Coding**:
  - >= 0.80: Green -- "Not significant"
  - 0.75-0.79: Yellow -- "Gray zone"
  - < 0.75: Red -- "Hemodynamically significant"
- **Outputs/Events**: `onToggleQfrMode()`, `onLoadProjection(id)`, `onSegmentProjection(id)`, `onCalibrateProjection(id)`, `onReconstruct()`, `onToggle3DView()`, `onChangeQfrMode(mode)`
- **Accessibility**: QFR result announced: "Quantitative flow ratio: 0.76. Interpretation: hemodynamically significant."

#### ECGPanel

- **Props/Inputs**: `ecgData: ECGData | null`, `currentFrame: number`, `isEditMode: boolean`
- **States**:
  - **No ECG**: "No ECG data in this DICOM file. Use motion signal for cardiac phase detection."
  - **ECG loaded**: Canvas (full panel width x 80px) with green waveform, red R-peak markers, white frame cursor line.
  - **Edit mode on**: Blue border highlight. Guide popup: "Click to add peak, right-click to remove, drag to move."
  - **Heart rate**: "HR: 72 bpm" text overlay.
- **Outputs/Events**: `onToggleEditMode()`, `onAddPeak(sampleIndex)`, `onRemovePeak(sampleIndex)`, `onMovePeak(fromSample, toSample)`, `onScrubToFrame(frameIndex)`
- **Accessibility**: `aria-label="ECG waveform. Heart rate: {hr} beats per minute. {n} R-peaks detected."`. Edit mode changes announced via `aria-live`.

#### MetadataDisplay

- **Props/Inputs**: `metadata: DicomMetadata | null`
- **States**: No data (placeholder text) | Loaded (metadata table).
- **Displayed Fields**: Patient ID (anonymized if applicable), Study Date, Modality, Manufacturer, Dimensions, Frame Count, Frame Rate, Pixel Spacing, Primary Angle, Secondary Angle, ECG Type.
- **Accessibility**: `role="table"` with labeled rows.

#### CalibrationPanel

- **Props/Inputs**: `calibration: Calibration | null`, `dicomPixelSpacing: [number, number] | null`
- **States**:
  - **Not calibrated**: Yellow badge "Uncalibrated." Catheter size buttons (5F, 6F, 7F, 8F). Manual input field.
  - **DICOM calibrated**: Green badge "Calibrated (DICOM)." Pixel spacing displayed.
  - **Catheter calibrated**: Green badge "Calibrated (Catheter {size})." Pixel spacing displayed.
  - **Manual calibrated**: Green badge "Calibrated (Manual)." Pixel spacing displayed.
- **Outputs/Events**: `onCalibrateFromCatheter(size)`, `onCalibrateManual(pixelSpacing)`, `onCalibrateFromMask()`, `onReset()`
- **Accessibility**: Current calibration state announced. Size buttons use `role="radiogroup"`.

#### ExportPanel

- **Props/Inputs**: `hasQcaData: boolean`, `hasRwsData: boolean`, `hasQfrData: boolean`
- **States**: No data available (all exports disabled) | Data available (relevant exports enabled) | Exporting (spinner) | Success (checkmark) | Error (error text).
- **Outputs/Events**: `onExport(format, options)`

#### SeriesPicker (Modal)

- **Props/Inputs**: `mode: 'single' | 'dual'`, `onSelect: (series) => void`
- **States**:
  - **Closed**: Not rendered.
  - **Open, no folder**: "Select Folder" button prominent.
  - **Scanning**: Progress text "Scanning DICOM files..."
  - **Grid populated**: 3-column grid. Each cell: thumbnail (128x128), angle label (RAO 30 / CRA 15), frame count badge, dimensions.
  - **Single selection**: Selected cell highlighted with blue border.
  - **Dual selection**: P1 = blue border + "P1" label, P2 = teal border + "P2" label.
- **Outputs/Events**: `onSelectSeries(series, projectionId?)`, `onClose()`
- **Accessibility**: Grid uses `role="listbox"`, each series is `role="option"`. Focus trap within modal.

### 5.3 Control Components

#### Toolbar (Left Sidebar)

- **Props/Inputs**: `annotationMode: AnnotationMode`, `isQfrMode: boolean`, `hasFile: boolean`
- **States**: Per-button states: active (blue background), inactive (surface-tertiary), disabled (50% opacity), hover (lightened background + tooltip).
- **Buttons** (top to bottom):
  1. Open File (Folder icon) -- always enabled
  2. ROI Tool (Square icon) -- disabled when no file
  3. Seed Tool (Crosshair icon) -- disabled when no file
  4. Pan Tool (Hand icon) -- disabled when no file
  5. --- separator ---
  6. QFR Mode Toggle (3D icon) -- disabled when no file
  7. --- separator ---
  8. Settings (Gear icon) -- always enabled
  9. Shortcuts Help (? icon) -- always enabled
- **Outputs/Events**: `onToolSelect(tool)`, `onOpenFile()`, `onToggleQfrMode()`, `onOpenSettings()`, `onShowHelp()`
- **Accessibility**: `role="toolbar"`, `aria-label="Analysis tools"`. Each button has `aria-label="{tool name} ({shortcut key})"` and `aria-pressed` for toggle buttons. Arrow key navigation between buttons.

#### PlaybackControls

- **Props/Inputs**: `currentFrame: number`, `totalFrames: number`, `playbackState: PlaybackState`, `speed: number`, `isLooping: boolean`, `isTrackMode: boolean`, `trackingConfidence: number`
- **States**:
  - **No file**: All disabled, skeleton placeholders.
  - **Stopped**: Play icon on play button.
  - **Playing**: Pause icon on play button, frame counter updating.
  - **Track mode**: Orange tint on transport area, "TRACK MODE" pulsing badge, confidence percentage display.
- **Outputs/Events**: `onPlay()`, `onPause()`, `onNextFrame()`, `onPrevFrame()`, `onSeek(frame)`, `onSpeedChange(speed)`, `onToggleLoop()`, `onToggleTrackMode()`
- **Accessibility**: `role="toolbar"`, `aria-label="Playback controls"`. Frame slider: `role="slider"`, `aria-valuemin="0"`, `aria-valuemax="{total}"`, `aria-valuenow="{current}"`, `aria-label="Frame position"`.

### 5.4 Chart Components

#### RWSChart

- **Props/Inputs**: `results: RWSResult[]`, `currentFrame: number`
- **States**: No data (empty chart with axis labels) | Data (line/bar chart of RWS values across beats with color-coded thresholds).
- **Accessibility**: `role="img"`, `aria-label="RWS chart showing {n} beats. Mean MLD RWS: {value}%."`. Data table alternative accessible via hidden table.

#### DiameterChart (QCA)

- **Props/Inputs**: `diameterProfile: number[]`, `mldIndex: number`, `proximalRdIndex: number`, `distalRdIndex: number`
- **States**: No data | Data (line chart of diameter profile with marked points for MLD and reference diameters).
- **Accessibility**: `role="img"`, `aria-label="Diameter profile along vessel centerline. Minimum lumen diameter: {mld} mm at position {index}."`.

#### ECGChart

- **Props/Inputs**: `signal: number[]`, `rPeaks: number[]`, `currentSample: number`, `isEditMode: boolean`
- **States**: No data | Waveform displayed | Edit mode (interactive).
- **Accessibility**: `aria-label="ECG waveform. {n} R-peaks detected. Heart rate: {hr} bpm."`.

#### MotionChart

- **Props/Inputs**: `signal: number[]`, `peaks: number[]`, `currentFrame: number`
- **States**: No data | Signal displayed | Edit mode.
- **Accessibility**: `aria-label="Motion signal from optical flow. {n} peaks detected."`.

### 5.5 Common Components

#### Button

- **Variants**: primary (blue), secondary (gray), destructive (red), ghost (transparent), outline (bordered).
- **Sizes**: sm (28px h), md (36px h), lg (44px h).
- **States**: default, hover, focus (ring-2 ring-blue-500 ring-offset-2), active (pressed), disabled (50% opacity, cursor-not-allowed), loading (spinner replacing icon/text).
- **Accessibility**: Native `<button>` element. `aria-disabled` when disabled. `aria-busy="true"` when loading.

#### Badge

- **Variants**: default (gray), success (green), warning (amber), error (red), info (blue).
- **Sizes**: sm (text-xs, px-1.5 py-0.5), md (text-sm, px-2 py-1).
- **Accessibility**: `role="status"` for dynamic badges. Static badges need no role.

#### Toast / Notification

- **Position**: Top-right corner, stacked vertically.
- **Variants**: success (green left border), warning (amber), error (red), info (blue).
- **Duration**: 3 seconds default, errors persist until dismissed.
- **Anatomy**: Icon | Message text | Dismiss X button.
- **Animation**: Slide in from right (200ms ease-out), fade out on dismiss.
- **Accessibility**: `role="alert"` for errors, `role="status"` for others. Auto-read by screen reader.

#### Modal / Dialog

- **Anatomy**: Backdrop (black/50, blur-sm) + centered content card.
- **Sizes**: sm (400px), md (600px), lg (900px).
- **Behavior**: Focus trap within modal. Escape closes. Click backdrop closes (unless blocking modal like Privacy Dialog).
- **Accessibility**: `role="dialog"`, `aria-modal="true"`, `aria-labelledby` pointing to title element. Focus trap via inert attribute on background content.

#### Tooltip

- **Trigger**: Hover (300ms delay) or focus.
- **Position**: Prefer top, fallback to bottom/left/right if clipped.
- **Appearance**: Dark background (slate-800), white text, 6px padding, 4px radius, max-width 200px.
- **Accessibility**: `role="tooltip"`, linked via `aria-describedby`.

#### Slider (Range)

- **Anatomy**: Track (4px h) + Thumb (16px circle) + Value label.
- **States**: default, hover (thumb grows to 20px), focus (ring), disabled.
- **Accessibility**: `role="slider"`, `aria-valuemin`, `aria-valuemax`, `aria-valuenow`, `aria-label`.

### 5.6 Mask Edit Components

#### MaskEditToolbar (Floating)

- **Position**: Centered at top of viewer, floating with 12px margin from top.
- **Appearance**: `bg-gray-900/95 backdrop-blur-sm rounded-lg shadow-xl` (glass morphism).
- **Tool buttons** (left to right):
  1. Brush (B) -- circle icon
  2. Quick-Brush (Q) -- fast-circle icon
  3. Eraser (E) -- eraser icon
  4. Smart-Brush (D) -- sparkle-circle icon
  5. Smart-Eraser (G) -- sparkle-eraser icon
  6. Lasso (L) -- lasso icon
  7. Polygon (P) -- pentagon icon
  8. Magic Wand (W) -- wand icon
  9. Contour Edit (C) -- pen-tool icon
  10. | separator |
  11. Undo (Ctrl+Z) -- undo icon
  12. Redo (Ctrl+Y) -- redo icon
  13. | separator |
  14. Morphological Ops (dropdown) -- layers icon
  15. | separator |
  16. Size slider (2-100px)
  17. | separator |
  18. [Save] -- green, checkmark icon
  19. [Cancel] -- red, X icon
- **States per tool**: active (highlighted accent), inactive (muted), disabled (for undo/redo when unavailable).
- **Accessibility**: `role="toolbar"`, `aria-label="Mask editing tools"`. Each button has shortcut key in aria-label.

---

## 6. Interaction Design

### 6.1 Canvas Interaction Matrix

| Tool Mode | Left Click | Left Drag | Right Click | Scroll | Double Click | Middle Drag |
|-----------|-----------|-----------|-------------|--------|-------------|------------|
| **Select** (default) | No action | No action | Context menu | Zoom in/out | No action | Pan |
| **ROI** | Place fixed 160x160 ROI | Draw freeform ROI box | Context menu | Zoom | No action | Pan |
| **Seed** | Add seed point | No action | Remove nearest seed point | Zoom | No action | Pan |
| **Pan** | Start pan (grab cursor) | Pan viewport | Context menu | Zoom | Reset view (fit) | Pan |
| **Mask Brush** | Start stroke (paint) | Continue painting stroke | No action (exit via Escape) | Change brush size | No action | Pan |
| **Mask Eraser** | Start stroke (erase) | Continue erasing stroke | No action | Change eraser size | No action | Pan |
| **Mask Smart Brush** | Start stroke (edge-aware paint) | Continue painting | No action | Change size | No action | Pan |
| **Mask Flood Fill** | Fill from click point | No action | No action | Adjust tolerance | No action | Pan |
| **Mask Contour Edit** | Select contour point | Drag contour point (Gaussian deformation) | No action | No action | No action | Pan |
| **QCA Marker Drag** | Grab marker | Drag marker along centerline | No action | Zoom | No action | Pan |
| **Seed Drag** | Grab seed | Drag seed to new position | No action | Zoom | No action | Pan |
| **ECG Scrub** | Navigate to clicked sample position | Scrub through timeline | ECG context menu | No action | No action | No action |
| **R-Peak Edit** | Add R-peak at position | Drag existing peak | Remove nearest peak | No action | No action | No action |

**Priority Chain** (when multiple interactions overlap):
1. R-Peak Edit (if ECG edit mode on and click in ECG area)
2. ECG Scrub (if click in ECG area)
3. Pan Tool (if pan mode active)
4. Mask Edit (if mask edit mode active)
5. QCA Marker Drag (if click near marker, hit radius 8px)
6. Seed Point Drag (if click near seed point, hit radius 10px)
7. Fixed ROI (if ROI tool active, fixed mode)
8. Seed Mode (if seed tool active)
9. ROI Mode (if ROI tool active, draw mode)
10. ROI Drag/Resize (if click on existing ROI)

### 6.2 Keyboard Shortcut Map

#### Global Shortcuts (active when viewer focused)

| Key | Action | Context Sensitivity |
|-----|--------|-------------------|
| Space | Play/Pause | Viewer mode only. In dialogs: activates focused button. |
| ArrowLeft | Previous frame | Viewer mode only. In inputs: cursor movement. |
| ArrowRight | Next frame | Viewer mode only. |
| ArrowUp | Increase playback speed | Viewer mode only. |
| ArrowDown | Decrease playback speed | Viewer mode only. |
| Home | Go to first frame | Viewer mode only. |
| End | Go to last frame | Viewer mode only. |
| B | Select ROI tool | Switches to Brush in mask edit mode. |
| S | Select Seed tool | No effect in mask edit mode. |
| H | Select Pan tool | No effect in mask edit mode. |
| R | Reset view (fit to window) | No effect in mask edit mode. |
| C | Cycle annotation modes | Switches to Contour Edit in mask edit mode. |
| + / = | Zoom in | Always. |
| - | Zoom out | Always. |
| Escape | Deselect tool / Exit mode | In mask edit: prompt save/discard. In dialogs: close. |
| ? | Show shortcuts help overlay | Always. |
| Ctrl+O | Open file | Always. |
| Ctrl+S | Export current data | Always (when data available). |
| Ctrl+Z | Undo | Mask edit mode only. |
| Ctrl+Y / Ctrl+Shift+Z | Redo | Mask edit mode only. |

#### Mask Edit Shortcuts (active only during mask editing)

| Key | Action |
|-----|--------|
| B | Brush tool |
| Q | Quick Brush tool |
| E | Eraser tool |
| D | Smart Brush tool |
| G | Smart Eraser tool |
| L | Lasso tool |
| P | Polygon tool |
| W | Magic Wand tool |
| C | Contour Edit tool |
| [ | Decrease brush/eraser size by 2px |
| ] | Increase brush/eraser size by 2px |

**Shortcut suppression**: All keyboard shortcuts are suppressed when focus is inside `<input>`, `<textarea>`, or `<select>` elements. Shortcuts are registered on `window` with capture-phase listeners.

### 6.3 Drag-and-Drop

| Source | Target | Action |
|--------|--------|--------|
| .dcm file from OS file manager | Viewer area (or anywhere on app window) | Triggers Privacy Dialog, then DICOM load. |
| .dcm file from OS file manager | Series Picker modal (when open) | Adds file to the series scan results. |

**Visual Feedback**: When dragging a file over the application, a drop-zone overlay appears on the viewer: dashed blue border + "Drop DICOM file here" text, semi-transparent blue background. Overlay disappears on `dragleave` or `drop`.

### 6.4 Touch / Trackpad Support (P2)

| Gesture | Action |
|---------|--------|
| Single tap | Same as left-click |
| Long press (500ms) | Same as right-click (context menu) |
| Pinch | Zoom in/out |
| Two-finger drag | Pan |
| Swipe left/right | Previous/next frame |

---

## 7. Design System

### 7.1 Color Tokens

#### Semantic Surface/Content Tokens

| Token Name | Light Value | Dark Value | Usage |
|-----------|-------------|------------|-------|
| `--surface-primary` | `#ffffff` | `#0f172a` (slate-900) | Main background (viewer, panels) |
| `--surface-secondary` | `#f1f5f9` (slate-100) | `#1e293b` (slate-800) | Header, footer, panel backgrounds |
| `--surface-tertiary` | `#e2e8f0` (slate-200) | `#334155` (slate-700) | Buttons, inputs, cards, hover states |
| `--content-primary` | `#0f172a` (slate-900) | `#f1f5f9` (slate-100) | Headings, primary text |
| `--content-secondary` | `#475569` (slate-600) | `#cbd5e1` (slate-300) | Labels, secondary text |
| `--content-muted` | `#64748b` (slate-500) | `#94a3b8` (slate-400) | Disabled, placeholder, hints |
| `--brand` | `#2563eb` (blue-600) | `#3b82f6` (blue-500) | Primary actions, active states |
| `--border` | `#e2e8f0` (slate-200) | `#334155` (slate-700) | Dividers, borders |
| `--error` | `#ef4444` (red-500) | `#ef4444` | Error text, badges |
| `--warning` | `#f59e0b` (amber-500) | `#f59e0b` | Warning text, badges |
| `--success` | `#22c55e` (green-500) | `#22c55e` | Success text, badges |

#### Clinical Color Tokens

| Token Name | Value | Usage |
|-----------|-------|-------|
| `--clinical-normal` | `#22c55e` (green-500) | RWS <8%, QFR >=0.80, DS% <50% |
| `--clinical-intermediate` | `#f59e0b` (amber-500) | RWS 8-12%, QFR 0.75-0.79 |
| `--clinical-elevated` | `#f97316` (orange-500) | RWS 12-14%, DS% 70-89% |
| `--clinical-high-risk` | `#ef4444` (red-500) | RWS >14%, QFR <0.75, DS% >=90% |

#### Vessel Color Tokens

| Token Name | Value | Vessel |
|-----------|-------|--------|
| `--vessel-lad` | `#3b82f6` (blue-500) | Left Anterior Descending |
| `--vessel-lcx` | `#8b5cf6` (violet-500) | Left Circumflex |
| `--vessel-rca` | `#ef4444` (red-500) | Right Coronary Artery |
| `--vessel-lm` | `#f59e0b` (amber-500) | Left Main |
| `--vessel-other` | `#6b7280` (gray-500) | Other / unspecified |
| `--vessel-stenosis` | `#dc2626` (red-600) | Stenosis region marker |

#### QCA Marker Colors

| Marker | Color | Usage |
|--------|-------|-------|
| MLD | `#ef4444` (red-500) | Minimum Lumen Diameter position |
| Proximal RD | `#06b6d4` (cyan-500) | Proximal reference diameter marker |
| Distal RD | `#eab308` (yellow-500) | Distal reference diameter marker |
| Interpolated RD | `#a855f7` (purple-500) | Reference line on diameter chart |

### 7.2 Typography

| Element | Font Family | Size | Weight | Line Height |
|---------|-------------|------|--------|-------------|
| App title | System sans-serif (Inter, -apple-system, Segoe UI) | 18px (`text-lg`) | 600 (`font-semibold`) | 1.75 |
| Version badge | Same | 12px (`text-xs`) | 400 (`font-normal`) | 1.5 |
| Panel heading | Same | 14px (`text-sm`) | 600 (`font-semibold`) | 1.5 |
| Section heading | Same | 16px (`text-base`) | 500 (`font-medium`) | 1.5 |
| Body text | Same | 14px (`text-sm`) | 400 (`font-normal`) | 1.5 |
| Small text / labels | Same | 12px (`text-xs`) | 500 (`font-medium`) | 1.5 |
| Metric value (large) | Same | 24px (`text-2xl`) | 700 (`font-bold`) | 1.25 |
| Metric value (medium) | Same | 20px (`text-xl`) | 600 (`font-semibold`) | 1.25 |
| Status bar text | Same | 12px (`text-xs`) | 400 (`font-normal`) | 1.5 |
| Keyboard hint | Monospace (ui-monospace, Menlo, Monaco) | 12px | 400 | 1.5 |
| Code / values | Monospace | 13px | 400 | 1.5 |

### 7.3 Spacing

Base unit: 4px. All spacing follows a 4px grid.

| Token | Value | Usage |
|-------|-------|-------|
| `space-0` | 0px | No spacing |
| `space-0.5` | 2px | Tight inline gaps |
| `space-1` | 4px | Minimum gap |
| `space-2` | 8px | Default gap between related elements |
| `space-3` | 12px | Panel padding, card padding |
| `space-4` | 16px | Section spacing |
| `space-5` | 20px | Group spacing |
| `space-6` | 24px | Large section spacing |
| `space-8` | 32px | Major layout divisions |
| `space-10` | 40px | Page-level spacing |
| `space-12` | 48px | Header height |
| `space-14` | 56px | Toolbar width |
| `space-16` | 64px | Playback controls height |
| `space-80` | 320px | Right panel width |

### 7.4 Elevation (Shadows & Z-Index)

| Level | Shadow | Usage |
|-------|--------|-------|
| 0 (flat) | none | Default elements |
| 1 (raised) | `0 1px 2px rgba(0,0,0,0.05)` | Cards, buttons |
| 2 (overlay) | `0 4px 6px -1px rgba(0,0,0,0.1)` | Dropdowns, tooltips |
| 3 (modal) | `0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1)` | Mask edit toolbar |
| 4 (dialog) | `0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)` | Modals |
| 5 (system) | `0 25px 50px -12px rgba(0,0,0,0.25)` | Mobile panel |

Z-index layers: see Section 4.5.

### 7.5 Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `radius-sm` | 4px | Small buttons, badges, inputs |
| `radius-md` | 6px | Standard buttons, cards |
| `radius-lg` | 8px | Panels, modals, large cards |
| `radius-xl` | 12px | Floating toolbars |
| `radius-full` | 9999px | Circular buttons, avatar, dots |

### 7.6 Iconography

- **Library**: Lucide React (consistent line-weight icons)
- **Default size**: 18px (toolbar icons), 16px (inline icons), 14px (badge icons)
- **Stroke width**: 2px (default), 1.5px for dense UIs
- **Color**: Inherits from parent text color (`currentColor`)
- **Custom icons**: None needed for v2 -- Lucide covers all use cases
- **Medical icons**: Use abstract representations, not realistic anatomy (avoids regulatory issues)

### 7.7 Animation

| Animation | Duration | Timing | Usage |
|-----------|----------|--------|-------|
| Button hover | 150ms | ease | Background color transition |
| Tab switch | 150ms | ease | Active tab highlight |
| Toast enter | 200ms | ease-out | Slide in from right |
| Toast exit | 150ms | ease-in | Fade out |
| Mobile panel slide | 300ms | ease-in-out | Panel slide from right |
| Accordion open | 200ms | ease-out | Content height expansion |
| Accordion close | 200ms | ease-out | Content height collapse |
| Loading spinner | 1000ms | linear | Continuous rotation (animate-spin) |
| Track mode pulse | 2000ms | ease-in-out | Pulse animation on badge (animate-pulse) |
| Modal backdrop | 150ms | ease | Fade in/out |
| Modal content | 200ms | ease-out | Scale from 95% + fade in |
| Zoom indicator | 2000ms | ease | Appear, then fade out after 2s idle |

**Reduced Motion**: All animations respect `@media (prefers-reduced-motion: reduce)`. Under reduced motion:
- All transitions set to 0ms duration
- animate-spin replaced with static spinner icon
- animate-pulse replaced with static display
- Slide animations replaced with instant show/hide

### 7.8 QFR Heatmap Color Scale

For 3D mesh coloring and QFR profile charts:

| QFR Value | Color (hex) | RGB |
|-----------|------------|-----|
| 1.00 | `#22c55e` | (34, 197, 94) -- green |
| 0.90 | `#84cc16` | (132, 204, 22) -- lime |
| 0.80 | `#eab308` | (234, 179, 8) -- yellow |
| 0.75 | `#f97316` | (249, 115, 22) -- orange |
| 0.70 | `#ef4444` | (239, 68, 68) -- red |
| 0.60 | `#dc2626` | (220, 38, 38) -- dark red |

Linear interpolation between these stops.

---

## 8. State Management Specification

### 8.1 Consolidated Store Architecture

The v1.1 pattern of 17 independent stores with hidden `getState()` coupling is replaced with 8 consolidated stores with explicit dependency injection via selectors and action parameters.

| Store | Replaces | Persistence | Description |
|-------|----------|-------------|-------------|
| `viewerStore` | playerStore + overlayStore | None (session) | Playback, zoom/pan, overlay visibility, annotation mode |
| `studyStore` | dicomStore + ecgStore + motionStore | SQLite (via backend) | DICOM metadata, ECG data, motion signal, beat boundaries |
| `analysisStore` | segmentationStore + qcaStore + annotationStore | SQLite (via backend) | Per-frame segmentation, QCA results, seed points, ROIs |
| `rwsStore` | rwsStore (unchanged) | SQLite (via backend) | RWS results, outlier settings, vessel labels |
| `qfrStore` | qfrModeStore | SQLite (via backend) | QFR dual-projection state, reconstruction results |
| `editStore` | maskEditStore | None (session) | Mask edit state, undo history, tool settings |
| `trackingStore` | trackingStore (unchanged) | None (session) | CSRT tracking state, propagation progress |
| `settingsStore` | settingsStore + calibrationStore | localStorage | User preferences (theme, defaults, brush sizes), calibration |

### 8.2 Store Interfaces

```typescript
// ---- viewerStore ----
interface ViewerState {
  // Playback
  currentFrame: number;
  totalFrames: number;
  frameRate: number;
  playbackState: 'stopped' | 'playing' | 'paused';
  playbackSpeed: number; // 0.25, 0.5, 1.0, 2.0
  isLooping: boolean;

  // View transform
  viewTransform: { scale: number; x: number; y: number };
  viewTransformVersion: number; // incremented on change, for render optimization

  // Annotation mode
  annotationMode: 'select' | 'roi' | 'seed' | 'pan' | null;

  // Overlay visibility
  overlays: {
    mask: boolean;
    centerline: boolean;
    seedPoints: boolean;
    roi: boolean;
    diameterMarkers: boolean;
    crossSections: boolean;
    ecgSignal: boolean;
    motionSignal: boolean;
  };

  // Actions
  setCurrentFrame: (frame: number) => void;
  play: () => void;
  pause: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  setSpeed: (speed: number) => void;
  toggleLoop: () => void;
  setAnnotationMode: (mode: AnnotationMode | null) => void;
  setViewTransform: (transform: ViewTransform) => void;
  resetView: () => void;
  setOverlayVisibility: (layer: string, visible: boolean) => void;
  reset: () => void;
}

// ---- studyStore ----
interface StudyState {
  // DICOM
  metadata: DicomMetadata | null;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  sessionId: string | null; // Backend-generated, stored on load

  // ECG
  ecgData: ECGData | null;
  rPeaks: number[] | null;
  beatBoundaries: BeatBoundary[];
  heartRate: number | null;
  isEcgEditMode: boolean;

  // Motion
  motionSignal: MotionSignalData | null;
  motionPeaks: number[] | null;
  motionBeatBoundaries: MotionBeatBoundary[];

  // Actions
  loadFile: (file: File, anonymize: boolean) => Promise<void>;
  loadFromPath: (path: string, anonymize: boolean) => Promise<void>;
  clearStudy: () => void;
  loadEcg: () => Promise<void>;
  setRPeaks: (peaks: number[]) => void;
  addRPeak: (sampleIndex: number) => void;
  removeRPeak: (sampleIndex: number) => void;
  moveRPeak: (fromSample: number, toSample: number) => void;
  toggleEcgEditMode: () => void;
  calculateMotionSignal: () => Promise<void>;
  setMotionPeaks: (peaks: number[]) => void;
  reset: () => void;
}

// ---- analysisStore ----
interface AnalysisState {
  // Segmentation (Map keyed by frame index)
  frameData: Map<number, {
    mask: Uint8Array | null;
    probabilityMap: Float32Array | null;
    centerline: Point[];
    seedPoints: Point[];
    width: number;
    height: number;
    engine: SegmentationEngine;
    detectedVessel: string | null;
  }>;
  selectedEngine: SegmentationEngine;
  isSegmenting: boolean;
  segmentationError: string | null;

  // QCA (Map keyed by frame index)
  qcaResults: Map<number, QCAMetrics>;
  isCalculatingQca: boolean;

  // Annotations (Map keyed by frame index)
  annotations: Map<number, {
    seedPoints: Point[];
    roi: BoundingBox | null;
  }>;

  // Available engines
  availableEngines: Map<string, SegmentationEngineInfo>;

  // Actions
  segment: (frameIndex: number, options: SegmentOptions) => Promise<void>;
  segmentAndExtract: (frameIndex: number, options: SegmentOptions) => Promise<void>;
  calculateQca: (frameIndex: number, options: QcaOptions) => Promise<void>;
  setEngine: (engine: SegmentationEngine) => void;
  addSeedPoint: (frameIndex: number, point: Point) => void;
  removeSeedPoint: (frameIndex: number, index: number) => void;
  clearSeedPoints: (frameIndex: number) => void;
  setRoi: (frameIndex: number, roi: BoundingBox | null) => void;
  clearFrameData: (frameIndex: number) => void;
  loadAvailableEngines: () => Promise<void>;
  reset: () => void;
}

// ---- rwsStore ----
interface RwsState {
  results: RWSResult[];
  summary: RWSSummary | null;
  isCalculating: boolean;
  error: string | null;

  // Input state
  startFrame: number | null;
  endFrame: number | null;
  selectedBeat: number | null;
  outlierMethod: 'none' | 'hampel' | 'double_hampel';
  selectedVessel: CoronaryVessel | null;

  // Actions
  calculate: (startFrame: number, endFrame: number, options: RwsOptions) => Promise<void>;
  deleteResult: (resultId: string) => void;
  setStartFrame: (frame: number | null) => void;
  setEndFrame: (frame: number | null) => void;
  setOutlierMethod: (method: string) => void;
  setVessel: (vessel: CoronaryVessel | null) => void;
  reset: () => void;
}

// ---- qfrStore ----
interface QfrState {
  isQfrMode: boolean;
  projection1: ProjectionState | null;
  projection2: ProjectionState | null;
  angularSeparation: number | null;
  isReconstructing: boolean;
  reconstructionError: string | null;

  // Results
  qfrResult: QFR3DResult | null;
  mesh3D: VesselMesh | null;
  qfrMode: 'fQFR' | 'cQFR' | 'aQFR';
  viewMode: 'projections' | '3d';

  // Sync playback
  isSyncPlayback: boolean;

  // Actions
  toggleQfrMode: () => void;
  loadProjection: (projectionId: 1 | 2, file: File) => Promise<void>;
  segmentProjection: (projectionId: 1 | 2, options: SegmentOptions) => Promise<void>;
  calibrateProjection: (projectionId: 1 | 2, method: CalibrationMethod) => Promise<void>;
  setTimiFrameCount: (projectionId: 1 | 2, tStart: number, tEnd: number) => void;
  reconstruct: () => Promise<void>;
  setQfrMode: (mode: 'fQFR' | 'cQFR' | 'aQFR') => void;
  setViewMode: (mode: 'projections' | '3d') => void;
  clearProjection: (projectionId: 1 | 2) => void;
  reset: () => void;
}

// ---- editStore ----
interface EditState {
  isEditMode: boolean;
  currentFrameIndex: number | null;
  workingMask: Uint8Array | null;
  originalMask: Uint8Array | null;
  activeTool: MaskEditTool;
  undoStack: Uint8Array[]; // max 20
  redoStack: Uint8Array[];
  maskWidth: number;
  maskHeight: number;

  // Actions
  enterEditMode: (frameIndex: number, mask: Uint8Array, width: number, height: number) => void;
  exitEditMode: (save: boolean) => void;
  setTool: (tool: MaskEditTool) => void;
  applyBrushStroke: (points: Point[], isErasing: boolean) => Promise<void>;
  applyFloodFill: (point: Point, tolerance: number) => Promise<void>;
  applyMorphologicalOp: (op: MorphOp) => Promise<void>;
  undo: () => void;
  redo: () => void;
  reset: () => void;
}

// ---- trackingStore ----
interface TrackingState {
  isInitialized: boolean;
  isTrackMode: boolean;
  isPropagating: boolean;
  autoSegQcaEnabled: boolean;
  currentRoi: BoundingBox | null;
  confidence: number;
  propagationProgress: PropagationProgress;
  frameResults: Map<number, TrackingResult>;

  // Actions
  initialize: (frameIndex: number, roi: BoundingBox) => Promise<void>;
  trackFrame: (frameIndex: number) => Promise<TrackingResult>;
  propagate: (startFrame: number, endFrame: number, direction: 'forward' | 'backward') => Promise<void>;
  cancelPropagation: () => void;
  toggleTrackMode: () => void;
  toggleAutoSegQca: () => void;
  resetTracker: () => void;
  reset: () => void;
}

// ---- settingsStore ----
// (Same as v1.1, see existing settingsStore.ts interface, with calibration merged in)
interface SettingsState {
  maskEdit: MaskEditSettings;
  player: PlayerSettings;
  overlay: OverlaySettings;
  segmentation: SegmentationSettings;
  tracking: TrackingSettings;
  calibration: CalibrationSettings & {
    currentPixelSpacing: [number, number] | null;
    currentSource: CalibrationSource | null;
  };
  paths: PathSettings;
  appearance: AppearanceSettings;
  qfr: QfrSettings;
  // ... setters and reset (same pattern as v1.1)
}
```

### 8.3 Store Dependency Graph

```
studyStore ──> viewerStore (sets totalFrames, frameRate on load)
            ──> analysisStore.reset() (on new file load)
            ──> rwsStore.reset() (on new file load)
            ──> qfrStore.reset() (on new file load)
            ──> trackingStore.reset() (on new file load)
            ──> editStore.reset() (on new file load)

analysisStore ──> settingsStore (reads preferred engine)

rwsStore ──> studyStore (reads beat boundaries)
          ──> analysisStore (reads per-frame QCA data)

qfrStore ──> (self-contained, interacts via API only)

trackingStore ──> analysisStore (triggers segment+QCA when autoSegQca enabled)

editStore ──> analysisStore (reads/writes mask for frame)

settingsStore ──> (no dependencies, leaf node)
viewerStore ──> (no dependencies, leaf node)
```

**Key rule**: Stores NEVER call `otherStore.getState()` directly. Instead:
- Cross-store data is passed as action parameters by the component layer.
- The `studyStore.loadFile()` action calls reset on other stores via a `resetAllAnalysis()` helper that the component invokes.
- Components use `useXxxStore(selector)` to read from multiple stores and pass data down.

### 8.4 Persistence Rules

| Store | Persisted? | Method | What is saved |
|-------|-----------|--------|---------------|
| `viewerStore` | No | -- | Session-only. Reset on page load. |
| `studyStore` | Yes | SQLite via backend | DICOM metadata, ECG data, R-peaks, motion signal. (Frames stored as files, not in SQLite.) |
| `analysisStore` | Yes | SQLite via backend | Segmentation masks (as PNG files), centerlines, QCA results per frame. |
| `rwsStore` | Yes | SQLite via backend | All RWS results with parameters. |
| `qfrStore` | Yes | SQLite via backend | Projection metadata, reconstruction results, QFR values. |
| `editStore` | No | -- | Session-only. Undo stack lost on page close. |
| `trackingStore` | No | -- | Session-only. Tracking state not persisted. |
| `settingsStore` | Yes | localStorage | User preferences, calibration, theme, tool defaults. |

### 8.5 Reset / Cleanup Strategy

| Event | Stores Reset | Stores Preserved |
|-------|-------------|-----------------|
| New DICOM load | studyStore, analysisStore, rwsStore, qfrStore, editStore, trackingStore, viewerStore (partial: frame/playback reset, keep zoom/overlays) | settingsStore |
| Session end (close tab) | All in-memory stores | settingsStore (localStorage), persisted data (SQLite) |
| Clear study (explicit) | Same as new DICOM load | settingsStore |
| Enter mask edit mode | editStore initialized from analysisStore | All others unchanged |
| Exit mask edit mode (save) | editStore reset, analysisStore updated with new mask | All others unchanged |
| Exit mask edit mode (cancel) | editStore reset | All others unchanged (original mask preserved) |

### 8.6 API Client Architecture

Split `api.ts` into per-domain modules:

```
src/lib/api/
  index.ts          # Re-exports all modules, shared axios instance, interceptors
  dicom.ts          # /dicom/* endpoints
  segmentation.ts   # /segmentation/* endpoints
  qca.ts            # /qca/* endpoints
  rws.ts            # /rws/* endpoints
  qfr.ts            # /qfr/* endpoints
  motion.ts         # /motion/* endpoints
  calibration.ts    # /calibration/* endpoints
  tracking.ts       # /tracking/* endpoints
  maskEdit.ts       # /mask-edit/* endpoints
  export.ts         # /export/* endpoints
  report.ts         # /report/* endpoints
  health.ts         # /health, /status, /version
  types.ts          # Shared request/response types
```

Each module exports named functions matching the API contract. The shared axios instance in `index.ts` handles:
- Base URL configuration (`__API_BASE__`)
- Session ID header (`X-Session-ID`)
- Error response transformation
- Request/response logging (development only)

---

## 9. Accessibility Specification

### 9.1 WCAG 2.1 AA Compliance Checklist

| Criterion | Level | Status | Implementation |
|-----------|-------|--------|---------------|
| 1.1.1 Non-text Content | A | Partial | Canvas-based viewer cannot provide text alternatives for DICOM images. All UI icons have aria-labels. Chart components have aria-label descriptions. |
| 1.3.1 Info and Relationships | A | Full | Semantic HTML: headings, lists, tables. ARIA roles for custom widgets. |
| 1.3.2 Meaningful Sequence | A | Full | DOM order matches visual order. Tab order follows logical flow. |
| 1.4.1 Use of Color | A | Full | All color-coded indicators (RWS severity, QFR thresholds, DS% badges) include text labels alongside color. |
| 1.4.3 Contrast (Minimum) | AA | Full | All text meets 4.5:1 ratio. Muted text meets 3:1 for large text. |
| 1.4.11 Non-text Contrast | AA | Full | UI controls (buttons, inputs, sliders) have 3:1 contrast against background. |
| 2.1.1 Keyboard | A | Partial | All UI controls keyboard-accessible. Canvas interactions (ROI drawing, seed placement) NOT keyboard-accessible -- P1 will add numeric coordinate input as alternative. |
| 2.1.2 No Keyboard Trap | A | Full | Focus trapping only in modals (which have Escape exit). Tab does not trap anywhere else. |
| 2.4.1 Bypass Blocks | A | Full | Skip-to-content link. |
| 2.4.3 Focus Order | A | Full | Tab order matches visual/logical order (header > toolbar > viewer > charts > controls > panel). |
| 2.4.7 Focus Visible | AA | Full | All focusable elements have `ring-2 ring-blue-500 ring-offset-2` focus style. |
| 3.3.1 Error Identification | A | Full | All errors identified in text (not just color) via toast messages and inline error text. |
| 4.1.2 Name, Role, Value | A | Full | All custom widgets have ARIA roles, names, and states. |

### 9.2 Screen Reader Behavior for Medical Data

**QFR Announcement**: When QFR result is calculated, the `aria-live="polite"` region announces: "QFR calculation complete. Quantitative flow ratio: 0.76. Interpretation: hemodynamically significant. Pressure drop: 12.4 millimeters of mercury."

**RWS Announcement**: "RWS calculation complete. Beat 1, LAD vessel. MLD radial wall strain: 10.4 percent. Interpretation: intermediate. Proximal RWS: 8.2 percent. Distal RWS: 11.1 percent."

**QCA Announcement**: "QCA analysis complete. Minimum lumen diameter: 1.23 millimeters. Diameter stenosis: 62 percent, moderate severity."

**Calibration Change**: "Calibration updated. Pixel spacing: 0.15 millimeters per pixel. Source: catheter 6 French."

**Frame Navigation**: "Frame 45 of 119." (Announced on arrow key navigation, debounced to 500ms to avoid rapid-fire announcements during playback.)

### 9.3 High Contrast Mode

When the OS has high-contrast mode enabled (`@media (forced-colors: active)`):
- All custom colors replaced with system colors (CanvasText, Canvas, LinkText, etc.)
- Custom borders become visible (1px solid CanvasText)
- Focus indicators use system highlight color
- Vessel colors fall back to system colors with pattern differentiation (dashed for LCx, dotted for RCA, solid for LAD)

### 9.4 Reduced Motion Mode

When `@media (prefers-reduced-motion: reduce)`:
- All `transition-*` set to `duration: 0ms`
- `animate-spin` replaced with static icon
- `animate-pulse` removed (static display)
- Mobile panel appears/disappears instantly (no slide)
- Modal appears/disappears instantly (no scale/fade)
- Toast appears/disappears instantly

### 9.5 Keyboard-Only Navigation Map

```
Tab Order:
  [Skip-to-content link]
  |
  v
  Header: [Theme] -> [Settings] -> [Fullscreen]
  |
  v
  Toolbar: [Open File] -> [ROI Tool] -> [Seed Tool] -> [Pan Tool] -> [QFR Toggle] -> [Settings] -> [Help]
  |
  v
  Viewer: [Focusable area - receives shortcuts: Space, Arrows, B, S, H, R, +, -]
  |
  v
  Chart Tab Bar: [ECG] -> [QCA] -> [RWS] -> [Motion]
  |
  v
  Playback: [Prev] -> [Play/Pause] -> [Next] -> [Speed select] -> [Loop toggle] -> [Frame slider]
  |
  v
  Panel Tab Bar: [Seg] -> [RWS] -> [QFR] -> [Info] -> [Export]
  |
  v
  Panel Content: [Interactive elements in reading order]
```

### 9.6 Focus Management Rules

| Event | Focus Moves To |
|-------|----------------|
| Dialog open | First focusable element inside dialog |
| Dialog close (X or Escape) | Element that triggered the dialog |
| Tab switch in right panel | First focusable element in the new tab content |
| Segmentation complete | [Segment] button (remains focused) |
| RWS calculate complete | First result card (for screen reader announcement) |
| Mask edit mode enter | First tool button in floating toolbar (Brush) |
| Mask edit mode exit | [Edit Mask] button in Segmentation panel |
| File loaded | Viewer (for immediate keyboard navigation) |
| Error toast | Toast is announced via aria-live but focus does NOT move (non-disruptive) |

---

## 10. Responsive Design Specification

### 10.1 Breakpoint Table

| Breakpoint | Name | Width | Priority |
|------------|------|-------|----------|
| Default | Mobile | 0 - 767px | P2 (future) |
| `md` | Tablet | 768px - 1023px | P2 (future) |
| `lg` | Desktop (small) | 1024px - 1279px | P0 (supported) |
| `xl` | Desktop (standard) | 1280px - 1535px | P0 (primary target) |
| `2xl` | Desktop (large) | 1536px+ | P0 (supported) |

**Minimum supported resolution**: 1024 x 768 px (with scrolling).
**Optimal resolution**: 1920 x 1080 px.
**Target pixel density**: 1x and 2x (Retina). Canvas renders at `devicePixelRatio` for crisp medical images.

### 10.2 Per-Breakpoint Layout Changes

#### Desktop (lg: 1024px - 1279px)
- Right panel: 280px width (reduced from 320px)
- Chart area: 100px height (reduced from 120-200px)
- Header: version text hidden
- Toolbar: unchanged (56px)

#### Desktop (xl: 1280px+, primary)
- Full layout as specified in Section 4.1
- Right panel: 320px
- Chart area: 120-200px
- All header elements visible

#### Tablet (md: 768px - 1023px) -- P2
- Right panel: overlay mode (slides in from right, 320px, z-40)
- Hamburger menu button in header
- Chart area: collapsed by default, expandable
- Viewer: takes full width

#### Mobile (< 768px) -- P2
- Right panel: full-width bottom sheet
- Toolbar: horizontal strip below header (icons only)
- Chart area: hidden, accessible via panel tab
- Viewer: full width, reduced height
- Touch targets: minimum 44px

### 10.3 Touch Target Sizes

All interactive elements must have a minimum touch target of:
- **Buttons**: 44px x 44px minimum (for medical accuracy -- larger than the standard 40px)
- **Toolbar icons**: 44px x 44px (current 56px width toolbar exceeds this)
- **Tab buttons**: 44px height minimum
- **Slider thumbs**: 44px hit area (visual thumb can be smaller, hit area extends)
- **Seed points on canvas**: 44px hit radius for selection (visual radius 6px, hit radius 22px)
- **QCA markers**: 44px hit radius

### 10.4 Mobile-Specific Interactions (P2)

| Gesture | Action |
|---------|--------|
| Swipe left | Next frame |
| Swipe right | Previous frame |
| Pinch | Zoom in/out |
| Two-finger drag | Pan |
| Long press (500ms) | Context menu |
| Double tap | Reset zoom to fit |
| Single tap | Tool-specific action (same as left-click) |

---

## 11. Error & Edge Case Handling (UI)

### 11.1 Error Message Catalog

Every user-facing error message, with exact text, severity, and recovery action.

#### DICOM Loading Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| DICOM_INVALID | "Invalid file format. Please select a DICOM (.dcm) file." | Error | Select a different file. |
| DICOM_NO_FRAMES | "This DICOM file has only {n} frame(s). Coronary analysis requires a multi-frame cine loop." | Warning | Load a different file. |
| DICOM_WRONG_MODALITY | "This DICOM file has modality '{modality}'. This tool is designed for X-ray Angiography (XA). Results may be unreliable." | Warning | Allow user to proceed or load a different file. |
| DICOM_CORRUPT | "Failed to parse DICOM file. The file may be corrupted or use an unsupported transfer syntax." | Error | Try a different file. |
| DICOM_TOO_LARGE | "This DICOM file ({size}MB) exceeds the recommended maximum ({max}MB). Loading may be slow." | Warning | Proceed or choose a smaller file. |
| DICOM_UPLOAD_FAILED | "Failed to upload DICOM file. Check that the backend server is running." | Error | Retry or restart backend. |

#### Segmentation Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| SEG_OOM | "Segmentation failed: insufficient GPU memory. Try using CPU mode or a smaller ROI." | Error | Switch device or resize ROI. |
| SEG_MODEL_NOT_FOUND | "Segmentation model not found for engine '{engine}'. Ensure model weights are installed." | Error | Check model installation. |
| SEG_NO_ROI | "The selected engine requires an ROI. Press B to draw a bounding box on the vessel." | Warning | Draw ROI. |
| SEG_NO_SEEDS | "AngioPy requires at least 2 seed points. Press S and click on the vessel." | Warning | Place seed points. |
| SEG_EMPTY_MASK | "No vessel detected in the selected region. Try adjusting the ROI or seed points." | Warning | Reposition ROI/seeds. |
| SEG_TIMEOUT | "Segmentation timed out after {seconds}s. The image may be too large for the current device." | Error | Try CPU mode or smaller ROI. |

#### QCA Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| QCA_NO_MASK | "No segmentation mask available for frame {n}. Segment the frame first." | Warning | Segment the frame. |
| QCA_NO_CENTERLINE | "No centerline available. The segmentation mask may be too small or fragmented." | Warning | Edit mask or re-segment. |
| QCA_FIT_FAILED | "Gaussian fitting failed on {n} of {total} cross-sections. Results may be less accurate." | Warning | Review results. Consider re-segmenting. |
| QCA_NO_CALIBRATION | "No spatial calibration set. Diameter values are in pixels, not millimeters. Set calibration for accurate measurements." | Warning | Set calibration. |

#### RWS Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| RWS_NO_RANGE | "Select a frame range for RWS analysis. Set start and end frames or click a beat button." | Info | Set frame range. |
| RWS_RANGE_TOO_SHORT | "Frame range is very short ({n} frames). RWS may be unreliable with fewer than 10 frames." | Warning | Expand range. |
| RWS_SEG_FAILED | "Segmentation failed on frame {n}. RWS calculation aborted at that frame." | Error | Segment problematic frame manually, then retry. |
| RWS_ALL_ZERO | "Zero strain detected across all frames. This may indicate a rigid stent or consistent segmentation artifact." | Warning | Check segmentation quality. |

#### QFR Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| QFR_ANGLE_INSUFFICIENT | "Angular separation between projections is {n} degrees. Minimum recommended: 25 degrees." | Warning | Use projections with larger angular difference. |
| QFR_ANGLE_IDENTICAL | "Both projections have identical viewing angles ({n} degrees). 3D reconstruction requires different angles." | Error | Load a different projection for P2. |
| QFR_RECON_FAILED | "3D reconstruction failed: {reason}. Check that both projections show the same vessel segment." | Error | Verify projections, recalibrate, re-segment. |
| QFR_RECON_LOW_QUALITY | "Reconstruction quality is low (reprojection error: {n}px). QFR result may be inaccurate." | Warning | Review calibration and segmentation. |
| QFR_NO_TIMI | "TIMI frame count not set. Set T-start and T-end frames for cQFR/aQFR modes. Using fQFR (fixed velocity) as default." | Info | Set TIMI frames or accept fQFR. |

#### Backend Errors

| Code | Message | Severity | Recovery |
|------|---------|----------|----------|
| BACKEND_UNREACHABLE | "Cannot connect to analysis backend. Ensure the Python server is running on port 8000." | Error | Start backend, click Retry. |
| BACKEND_TIMEOUT | "Request timed out. The backend may be processing a large operation." | Warning | Wait and retry. |
| BACKEND_500 | "Internal server error: {detail}." | Error | Retry. If persistent, restart backend. |
| BACKEND_CRASH | "Lost connection to the backend. Attempting to reconnect..." | Error | Auto-retry with exponential backoff (1s, 2s, 4s, max 30s). |

### 11.2 Empty States

| Component | Empty State Text | Visual |
|-----------|-----------------|--------|
| VideoPlayer (no file) | "Coronary RWS Analyser" / "Open a DICOM file to begin analysis" | Centered text on dark background. Folder icon above text. |
| SegmentationPanel (no file) | "Load a DICOM file to begin segmentation." | Muted text. Engine grid visible but all buttons disabled. |
| QCAPanel (no segmentation) | "Segment the current frame to see QCA results." | Muted text. Empty chart placeholder. |
| RWSPanel (no data) | "Calculate QCA for multiple frames, then run RWS analysis." | Muted text. Numbered steps list. |
| QFRPanel (no projections) | "1. Load Projection 1  2. Load Projection 2  3. Segment both projections  4. Calibrate  5. 3D Reconstruct" | Numbered instruction list with muted text. |
| ECGPanel (no ECG) | "No ECG data in this DICOM file. Use the motion signal for cardiac phase detection." | Muted text with arrow pointing to motion tab. |
| MetadataDisplay (no file) | "No study loaded." | Single line muted text. |
| ExportPanel (no data) | "No analysis data to export. Complete an analysis first." | Muted text. All export buttons disabled. |
| RWSChart (no results) | Empty chart area with axis labels and grid lines, no data points. | Gray placeholder. |
| DiameterChart (no QCA) | Empty chart area with axis labels. | Gray placeholder. |
| 3D Viewer (no mesh) | "No 3D reconstruction data." | Centered muted text on dark background. |
| SeriesPicker (no folder) | "Click 'Select Folder' to browse DICOM files." | Large folder icon + text. |
| TrackingResults (no tracking) | "Initialize tracking by drawing an ROI on the vessel." | Muted text. |

### 11.3 Edge Cases

| Scenario | Behavior |
|----------|----------|
| 1-frame DICOM | Load succeeds. Warning: "Single-frame DICOM. Playback and multi-frame analysis are not available." Playback controls disabled. Segmentation and QCA work on the single frame. RWS disabled (requires frame range). |
| No ECG in DICOM | ECG panel shows empty state. Motion signal auto-calculated as alternative. Beat buttons use motion peaks. |
| Corrupt/truncated DICOM | Load fails with DICOM_CORRUPT error. No partial load. |
| Very large DICOM (>500 frames) | DICOM_TOO_LARGE warning. Load proceeds. Frame loading is lazy (frames fetched on demand, not all at once). Playback pre-fetches N+5 and N-5 frames around current position. |
| Very large DICOM (>2048x2048 pixels) | Resize to max 1024x1024 on backend before transport. Note in metadata display: "Downsampled from {original} to {current}." |
| Tiny image (<128x128) | Warning: "Image resolution is very low ({W}x{H}). Measurements may be inaccurate." |
| All-black or all-white frame | Segmentation returns empty mask. SEG_EMPTY_MASK warning. |
| Frame rate = 0 or missing | Default to 15 fps. Warning: "Frame rate not found in DICOM metadata. Defaulting to 15 fps. TIMI calculations may be incorrect." |
| Pixel spacing missing from DICOM | No auto-calibration. QCA_NO_CALIBRATION warning persists until user calibrates manually. All measurements shown in pixels. |
| User closes browser tab during analysis | Session state persisted to SQLite if persistence is implemented. On reopen, offer restore. If no persistence yet, state is lost (warning on close: "Analysis data will be lost. Are you sure?"). |
| Backend crashes mid-segmentation | Frontend detects lost connection. BACKEND_CRASH error. Auto-retry. On reconnect, user must reload DICOM (module-level state in v1 is lost; v2 with proper session store can recover from persistence). |
| GPU out of memory | SEG_OOM error. Backend catches `RuntimeError`, calls `torch.cuda.empty_cache()`, returns 503 with message. Frontend suggests CPU fallback. |
| Network timeout (>30s) | BACKEND_TIMEOUT warning. Retry button. Do not auto-retry (operation may still be running server-side). |
| Drag-and-drop non-DICOM file | DICOM_INVALID error immediately after Privacy Dialog. |
| Multiple rapid Segment clicks | Debounce: ignore clicks while `isSegmenting` is true. Button is disabled during segmentation. |
| Frame change during mask edit | Auto-save current working mask to the current frame's slot. Load the new frame's mask into the editor (or blank if no mask exists). |
| RWS with identical Dmax and Dmin | RWS = 0%. Display result with note: "Zero strain detected." |
| QFR reconstruction with extremely short vessel (<5mm) | Warning: "Vessel length is very short ({length}mm). QFR may not be meaningful for vessels shorter than 10mm." |
| QFR with very high stenosis (>99%) | QFR approaches 0. Display normally -- the model handles it mathematically. |
| Zoom beyond 10x | Clamped to 10x. No error. |
| Zoom below 0.1x | Clamped to 0.1x. No error. |
| Pan beyond image bounds | Allowed (user sees dark background). Double-click to reset view. |
| Screen resolution below 1024x768 | ResolutionWarningModal: "Your screen resolution ({W}x{H}) may be insufficient for accurate medical image analysis. Recommended minimum: 1024x768." Dismissible (shown once per session). |

---

*End of SPEC Part 1: Product & UX Specification. This document defines every feature, interaction, state, layout, design token, accessibility requirement, and error scenario for the Coronary RWS Analyser v2.0 frontend.*
