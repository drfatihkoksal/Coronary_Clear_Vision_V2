# CONTEXT_UX.md -- UX Deep-Dive for Frontend Rewrite

This document is a comprehensive UX specification derived from auditing every component, store, hook, canvas layer, stylesheet, and configuration file in the existing `src/` codebase. It is intended to serve as the single source of truth for recreating every interaction, visual state, animation, and design token in a new frontend implementation.

---

## Table of Contents

- [A. User Flows (Step-by-Step)](#a-user-flows-step-by-step)
- [B. Interaction Patterns](#b-interaction-patterns)
- [C. UI States for Every Major Component](#c-ui-states-for-every-major-component)
- [D. Responsive Behavior](#d-responsive-behavior)
- [E. Animation & Transition Specs](#e-animation--transition-specs)
- [F. Design Token System](#f-design-token-system)
- [G. Accessibility Audit](#g-accessibility-audit)
- [H. Medical UX Considerations](#h-medical-ux-considerations)

---

## A. User Flows (Step-by-Step)

### A1. File Load Flow

1. **Trigger**: User clicks "Open File" in Toolbar, presses file-open shortcut, or drags a `.dcm` file onto the application window.
2. **File Input**: Hidden `<input type="file" accept=".dcm,application/dicom">` is programmatically clicked (Web API). For drag-and-drop, `onDrop` reads `e.dataTransfer.files[0]`.
3. **Privacy Dialog**: `PrivacyDialog` modal appears immediately (z-100 fixed overlay, 500px max-width). Three choices:
   - "Cancel" -- aborts load, clears `pendingLoad` state.
   - "No, Keep Original" -- calls `loadFile(file, false)`.
   - "Yes, Anonymize" -- calls `loadFile(file, true)` which strips patient identifiers server-side.
4. **Store Reset**: `resetAllStores()` clears ECG, Calibration, Segmentation, QCA, RWS, Tracking, Annotation stores. Backend also receives `resetAllAnalysis()`.
5. **DICOM Upload**: `dicomApi.load(file, anonymize)` POSTs the file to `/dicom/upload`. Returns `DicomMetadata` (numFrames, rows, columns, frameRate, pixelSpacing, seriesInstanceUid, etc.).
6. **Player Initialization**: `playerStore` receives totalFrames, frameRate; currentFrame set to 0.
7. **ECG Load**: `loadEcg(totalFrames, frameRate, seriesUid)` calls `/dicom/ecg`. If ECG exists in DICOM, R-peaks are detected and beat boundaries (in frame indices) are computed.
8. **Calibration Bootstrap**: `setFromDicom()` reads pixel spacing from the loaded metadata for spatial calibration.
9. **Motion Signal**: VideoPlayer auto-calculates motion signal on DICOM load via `motionApi.calculate()`.
10. **UI Update**: Header shows "Loading..." badge during upload, then displays frame count and resolution. Viewer renders frame 0. Status bar changes from "No file loaded" to "Ready".

### A2. Segmentation Flow

1. **Engine Selection**: User selects one of 8 engines in the SegmentationPanel grid: nnunet, nnunet-wide, angiopy, roi+angiopy, sam2, nnunet-fullframe, segformer, hrnet. Each engine has a distinct color badge.
2. **Preconditions by Engine**:
   - nnunet / nnunet-wide / segformer / hrnet: Need ROI drawn (optional but recommended).
   - angiopy: Need >= 2 seed points placed on the vessel.
   - roi+angiopy (Hybrid): Runs nnunet-fullframe -> centerline extraction -> generates 3 seed points -> AngioPy refinement. ROI optional.
   - sam2: Uses first seed point or ROI center as prompt.
   - nnunet-fullframe: No ROI needed, processes full 512x512 frame.
3. **ROI Drawing**: User selects ROI tool (B key or Toolbar button). Click-and-drag on canvas creates bounding box. ROI is stored per-frame in `annotationStore`.
4. **Seed Placement**: User selects Seed tool (S key or Toolbar). Click on vessel places a seed point (max 10). Seed points rendered as colored circles with index labels.
5. **Segment Button**: Clicking "Segment" in SegmentationPanel calls `segmentAndExtract()`. This:
   - Segments the current frame via the selected engine.
   - Extracts centerline via skeletonization.
   - Caches mask, probabilityMap, centerline, seedPoints, detectedVessel in `segmentationStore.frameData` Map keyed by frame index.
6. **Auto-QCA**: After segmentation, QCA analysis is automatically triggered if a centerline is available. QCA measures MLD, DS%, reference diameters along the centerline.
7. **Overlay Display**: Six overlay toggles in SegmentationPanel control visibility: Mask, Centerline, ROI, Seeds, Diameter markers, Cross-sections. Overlays render on the Segmentation canvas layer (z-index 1).
8. **Engine Auto-Select**: Changing the engine auto-selects the appropriate annotation mode (ROI mode for ROI-based engines, Seed mode for seed-based engines).

### A3. Mask Editing Flow

1. **Enter Edit Mode**: User clicks "Edit Mask" (available when a segmentation mask exists for the current frame). This calls `maskEditStore.enterEditMode(frameIndex, maskBase64, width, height)`.
2. **Floating Toolbar**: `MaskEditToolbar` appears centered at top (z-50, glass-morphism: `bg-gray-900/95 backdrop-blur-sm`). Contains 9 tools with keyboard shortcuts:
   - Brush (B), Quick-Brush (Q), Eraser (E), Smart-Brush (D), Smart-Eraser (G), Lasso (L), Polygon (P), Magic Wand (W), Contour Edit (C).
3. **Brush/Eraser Interaction**: Mouse cursor is replaced with a circle indicator matching brush/eraser size. Drawing applies to the working mask in real-time. Stroke points are collected for server-side mask application.
4. **Undo/Redo**: Ctrl+Z / Ctrl+Y. Up to 20 undo steps stored as base64 mask snapshots.
5. **Morphological Operations**: Dropdown menu with: Dilate, Erode, Fill Holes, Remove Islands, Smooth. Each sends the current mask to the backend for processing.
6. **Centerline Extraction**: Button to re-extract centerline from the edited mask.
7. **QCA Analysis**: Button to re-run QCA on the edited mask.
8. **Size Adjustment**: `[` / `]` keys decrease/increase brush or eraser size by 2px. Range: 2-100px.
9. **Save/Cancel**: Save commits the working mask back to `segmentationStore`. Cancel restores `originalMask`. Both exit edit mode.
10. **Auto-Save on Frame Change**: If frame changes during edit mode, the current working mask is automatically saved before loading the new frame's mask.

### A4. QCA Analysis Flow

1. **Prerequisites**: Segmentation mask + centerline must exist for the current frame.
2. **Method Selection**: Dropdown in QCAPanel: gaussian, parabolic, or threshold.
3. **Points Selection**: 30, 50, or 70 measurement points along the centerline.
4. **Calculate**: Calls QCA backend. Returns diameter profile, MLD (Minimum Lumen Diameter), DS% (Diameter Stenosis), reference diameters (proximal in cyan, distal in yellow), interpolated reference diameter, lesion length.
5. **Marker Dragging**: On the viewer, proximal and distal reference markers appear on the centerline as draggable points. User can drag them to adjust the reference segment. Markers snap to the nearest centerline point. Tooltip shows current diameter value during drag.
6. **Results Display**: QCAPanel shows MLD (red), DS% with severity badge (mild/moderate/severe/critical), proximal reference (cyan), distal reference (yellow), interpolated RD, lesion length, pixel spacing.
7. **Calibration Warning**: Yellow warning banner if no spatial calibration is set -- values shown in pixels, not mm.

### A5. RWS Calculation Flow

1. **Frame Range Selection**: User sets start and end frame for analysis. Can type frame numbers or click "Cur" buttons to use current frame. Can also right-click on the viewer and select "Set as RWS Start Frame" / "Set as RWS End Frame".
2. **ECG Beat Selection**: If ECG is available, beat boundary buttons (B1, B2, B3...) appear. Clicking a beat button auto-fills start/end frames from R-peak boundaries.
3. **Outlier Method**: Dropdown with 5 options: none, hampel, double (Hampel), IQR, temporal. Default: hampel.
4. **Vessel Selection**: Dropdown: LAD, LCX, RCA, LM, Other, or None. Labels results for multi-vessel analysis.
5. **Calculate**: Calls `rwsApi.calculate(startFrame, endFrame, { beatNumber, outlierMethod, qcaData })`. Backend segments + QCA each frame in range, computes MLD RWS = (Dmax - Dmin) / Dmax * 100%.
6. **Results Display**: Per-beat card showing:
   - MLD RWS value (large, color-coded: green < 8%, amber 8-12%, red > 12%).
   - Proximal, distal, average RWS values.
   - Dmin/Dmax frame links (clickable to navigate to that frame).
   - Beat number, vessel label, frame range.
7. **Multiple Results**: Results accumulate in a list. Summary statistics shown when multiple beats calculated.
8. **Dynamic Tabs**: Each calculated beat creates a tab in `VideoPlayerWithTabs` for quick navigation.

### A6. QFR Analysis Flow (Dual-Projection)

1. **Enable QFR Mode**: Click the 3D QFR toggle in Toolbar or QFRPanel mode toggle. This sets `qfrModeStore.isQfrMode = true`.
2. **Mode Switch**: The main viewer is replaced entirely by `QFRModeViewer` (dual-projection layout) or stays in single-view for single-projection QFR.
3. **Load Projections**: Click "Load P1" / "Load P2" in QFRPanel. Opens `SeriesPicker` modal (900px width).
   - **Series Picker**: User clicks "Select Folder" which opens browser's native folder picker (`webkitdirectory`). Client-side JavaScript parses DICOM headers from the folder's files (reads tags from ArrayBuffer without full decode). Displays 3-column grid with thumbnails, angle labels (RAO/LAO/CRA/CAU), frame count badges. User clicks a series to assign it to P1 or P2.
   - Dual-selection mode: P1 outlined in blue, P2 in teal.
4. **Per-Projection Workflow**: For each projection independently:
   - Place seed points on the vessel (2+ required).
   - Segment the vessel.
   - Calibrate (catheter-based or manual pixel spacing).
   - Set TIMI frame count: T-start (dye first appears) and T-end (dye reaches distal landmark).
5. **Angle Validation**: System checks that the angular separation between P1 and P2 is >= 25 degrees. Warning displayed if insufficient.
6. **Reconstruction**: Click "3D Reconstruct" button (gradient purple-to-blue). Backend performs:
   - Epipolar geometry matching between projections.
   - Stereo triangulation for 3D centerline.
   - Diameter measurement at matched points.
   - QFR calculation via Young-Tsai/Gould pressure model.
7. **View Mode Toggle**: After reconstruction, user can switch between "Projections" view (side-by-side) and "3D" view (Three.js mesh).
8. **QFR Result Display**: Large QFR value, color-coded (green >= 0.80, yellow 0.75-0.79, red < 0.75). Three QFR modes: fQFR (fixed), cQFR (contrast), aQFR (adenosine). Vessel metrics: length (mm), pressure drop (mmHg), flow velocity.
9. **Transfer to Main Viewer**: Projections can be "transferred" to the main viewer for advanced editing (mask editing, detailed segmentation). Data syncs back via `qfrModeStore.syncFromMainViewer()`.
10. **Thumbnail Preview**: When a projection is transferred to the main viewer, a thumbnail of the other projection appears in the top-right corner of the viewer.

### A7. Tracking Flow (CSRT)

1. **Initialize**: Draw ROI on a frame, then tracking auto-initializes via `trackingApi.initialize({ frameIndex, roi, roiMode, confidenceThreshold })`.
2. **ROI Modes**: `fixed_160x160` (default), `draw` (freeform), or configurable fixed sizes.
3. **Track Mode**: Toggle "Track" button in PlaybackControls. When enabled:
   - Playback triggers tracking on each frame advance.
   - ROI position updated from CSRT tracker result.
   - Confidence displayed (0-1 scale, color-coded).
   - If `autoSegQcaEnabled`, segmentation + QCA auto-run after each tracked frame.
4. **Visual Indicators**: Track mode shows orange-colored controls, "TRACK MODE" badge with pulse animation, confidence percentage.
5. **Propagation**: Batch-track across a frame range. Progress bar shows current/total frames.
6. **Loss of Tracking**: If confidence drops below threshold (default 0.6), tracking stops. User can re-initialize.

### A8. ECG Interaction Flow

1. **Display**: ECGPanel renders a 300x80px canvas with the ECG waveform. R-peaks shown as red vertical markers.
2. **Frame Sync**: A vertical cursor line moves along the ECG as the video frame changes, showing the temporal position.
3. **R-Peak Editing Mode**: Toggle "Edit" button. When enabled:
   - Left-click on ECG canvas adds an R-peak at the clicked sample position.
   - Right-click on an R-peak opens context menu with "Remove R-peak" option.
   - Drag an existing R-peak to move it to a new sample position.
   - Editing guide popup appears with instructions.
4. **Context Menu**: Right-click on ECG shows "Go to frame [N]" (navigates viewer to that frame) and "Remove R-peak" (if near a peak).
5. **Beat Recalculation**: After any R-peak edit, beat boundaries are recalculated via `recalculateBeatBoundaries()`.
6. **Signal Toggle**: Visibility toggles for ECG signal and Motion signal (if calculated).

### A9. Export Flow

1. **Quick Export (CSV)**: Toolbar button exports current QCA/RWS data as CSV.
2. **Training Export**: TrainingExportPanel provides export of frame + mask pairs for ML training.
3. **Report Generation**: ReportPanel generates a summary report (PDF if matplotlib available, otherwise structured data).

### A10. Settings Flow

1. **Open**: Click gear icon in Toolbar or header. Opens SettingsModal (600px max-width, 85vh max-height).
2. **Seven Tabs**: Appearance, Player, Segmentation, Tracking, Mask Edit, Calibration, Export.
3. **Appearance Tab**: Theme selection (light/dark/system) with large icon buttons. Show/hide Training tab toggle.
4. **Player Tab**: Default playback speed, loop behavior.
5. **Segmentation Tab**: Preferred engine, auto-QCA toggle.
6. **Tracking Tab**: Confidence threshold (range slider 0-1), ROI mode selection.
7. **Mask Edit Tab**: Default brush/eraser sizes, hardness, smart brush settings (edge sensitivity, intensity tolerance).
8. **Calibration Tab**: Embedded CalibrationPanel. Catheter size grid (5F-8F), manual pixel spacing input.
9. **Export Tab**: Embedded ExportPanel. Output format, path configuration.
10. **Persistence**: All settings stored in `localStorage` under key `coronary-rws-settings` (version 9). Zustand persist middleware handles serialization and migration across 9 schema versions.

---

## B. Interaction Patterns

### B1. Keyboard Shortcuts

All shortcuts registered via `useKeyboardShortcuts` hook using capture-phase event listeners (`window.addEventListener('keydown', handler, true)`). Shortcuts are skipped when focus is in `<input>`, `<textarea>`, or `<select>` elements.

#### Tool Shortcuts
| Key | Action | Context |
|-----|--------|---------|
| B | Select ROI tool / Brush (mask edit) | Global / Mask Edit |
| S | Select Seed tool | Global |
| H | Select Pan tool | Global |
| Escape | Deselect current tool / Exit edit mode | Global |

#### Playback Shortcuts
| Key | Action |
|-----|--------|
| Space | Play/Pause toggle |
| ArrowLeft | Previous frame |
| ArrowRight | Next frame |
| Home | First frame |
| End | Last frame |

#### Mask Edit Shortcuts
| Key | Action |
|-----|--------|
| B | Brush |
| Q | Quick Brush |
| E | Eraser |
| D | Smart Brush |
| G | Smart Eraser |
| L | Lasso |
| P | Polygon |
| W | Magic Wand |
| C | Contour Edit |
| [ | Decrease brush/eraser size by 2 |
| ] | Increase brush/eraser size by 2 |
| Ctrl+Z | Undo |
| Ctrl+Y / Ctrl+Shift+Z | Redo |

### B2. Mouse Interaction Priority Chain (VideoPlayer)

The VideoPlayer uses a priority-based mouse event dispatch. On `mousedown`, interactions are checked in this order:

1. **R-Peak Edit** -- If ECG edit mode active and click is on ECG overlay area.
2. **ECG Scrub** -- If click is on the ECG timeline region.
3. **Pan Tool** -- If pan mode active (`annotationMode === 'pan'`).
4. **Mask Edit** -- If mask edit mode active (brush/eraser/smart tools).
5. **QCA Marker Drag** -- If click is near a proximal/distal diameter marker on the centerline.
6. **Seed Point Drag** -- If click is near an existing seed point (hit test radius).
7. **Fixed ROI** -- If in fixed-ROI mode, places/moves a fixed-size ROI centered on click.
8. **Seed Mode** -- If seed tool active, adds a new seed point at click position.
9. **ROI Mode** -- If ROI tool active, starts drawing a new bounding box.
10. **ROI Drag/Resize** -- If click is on existing ROI border (resize) or interior (drag).

### B3. Canvas Coordinate System

All mouse interactions on the viewer canvas use coordinate transforms:

- **canvasToImage(canvasX, canvasY)**: Converts canvas pixel coordinates to image pixel coordinates, accounting for current zoom level and pan offset.
- **imageToCanvas(imageX, imageY)**: Converts image coordinates back to canvas display coordinates.
- **getTransform()**: Returns the current `{ scale, offsetX, offsetY }` for the viewport.

The zoom system uses `scale * 1.1` per scroll step (zoom in) or `scale / 1.1` per scroll step (zoom out). Zoom range: 0.1x to 10x. Zoom centers on the cursor position (not canvas center).

### B4. Context Menus

Two context menus exist in the application:

**VideoPlayer Context Menu** (right-click on canvas):
- "Set as RWS Start Frame" -- sets `rwsStore` start frame to current frame.
- "Set as RWS End Frame" -- sets `rwsStore` end frame to current frame.
- "Remove Seed Point" -- only shown if right-click is near a seed point.

**ECG Panel Context Menu** (right-click on ECG canvas):
- "Go to frame [N]" -- navigates the viewer to the frame corresponding to the clicked ECG sample position.
- "Remove R-peak" -- only shown if right-click is near an R-peak marker.

### B5. Drag-and-Drop

The root `<div>` in `AnalysisApp` handles `onDrop` and `onDragOver`. Dropping a file sets `pendingLoad` which triggers the Privacy Dialog flow (A1 step 3). No visual drop-zone indicator is currently rendered during `dragover`.

### B6. Right Panel Tab Navigation

Six tabs in the right sidebar panel: `seg-qca`, `rws`, `qfr`, `report`, `training`, `info`. The `training` tab is conditionally shown based on `appearance.showTrainingTab` setting. Tabs are rendered as `<button>` elements in a flex row, with the active tab having a blue-500 bottom border and tertiary background.

### B7. Playback Interaction

- **Transport Controls**: Previous frame, Play/Pause, Next frame buttons.
- **Speed Selector**: 0.25x, 0.5x, 1x (default), 2x.
- **Loop Toggle**: When enabled, playback wraps from last frame to first frame.
- **Timeline Slider**: Horizontal range input for frame scrubbing. Shows current frame / total frames.
- **Track Mode**: When enabled, each frame advance triggers CSRT tracking. Visual indicator: orange controls, "TRACK MODE" pulsing badge.

---

## C. UI States for Every Major Component

### C1. AnalysisApp (Root Layout)

| State | Visual |
|-------|--------|
| No file loaded | Viewer shows empty state, status bar shows "No file loaded", right panel shows placeholder messages |
| Loading | Header shows blue "Loading..." badge, viewer area may show spinner |
| File loaded | Header shows frame count + resolution info, status bar shows "Ready" |
| Error | Header shows red error badge with error message |
| Full screen | Maximize icon changes to Minimize icon, browser enters fullscreen |
| Settings open | SettingsModal overlays the entire app (z-50 modal) |
| Privacy dialog open | PrivacyDialog overlays (z-100, higher than settings) |
| Mobile panel open | Right panel slides in from right with backdrop blur overlay |

### C2. VideoPlayer

| State | Visual |
|-------|--------|
| Empty (no DICOM) | Black background, centered text: "Coronary RWS Analyser" + "Open a DICOM file to begin analysis" |
| Frame displayed | Video frame rendered on canvas, frame info overlay top-left (frame N/M) |
| Zoomed | Zoom indicator top-right shows percentage (e.g., "150%") |
| ROI mode | Cursor shows crosshair, ROI rectangle drawn with dashed border |
| Seed mode | Cursor shows crosshair, seed points rendered as colored circles with index labels |
| Pan mode | Cursor shows grab/grabbing hand |
| Mask edit mode | System cursor hidden, custom circle cursor follows mouse (matching brush/eraser size), "MASK EDIT" indicator shown, floating toolbar visible |
| QCA markers visible | Proximal (cyan) and distal (yellow) markers on centerline, draggable |
| Segmentation overlay | Semi-transparent colored mask over the vessel region |
| Centerline overlay | Thin line along vessel center |
| QFR thumbnail | Small preview image in top-right corner when editing a transferred projection |
| Context menu open | Small dropdown menu at right-click position |
| R-peak edit active | Guide text shown near ECG overlay area |
| Track mode active | "TRACK MODE" indicator with pulse animation |

### C3. PlaybackControls

| State | Visual |
|-------|--------|
| No file loaded | All controls disabled, gray/muted appearance, skeleton placeholders |
| Stopped | Play button shows play icon, frame counter shows "0 / 0" |
| Playing | Play button changes to pause icon, frame counter updates in real-time |
| Paused | Pause icon remains, frame counter frozen at current position |
| Loop enabled | Loop icon highlighted (blue) |
| Track mode active | Transport area tinted orange, "TRACK MODE" badge pulses, confidence percentage shown |
| Tracking in progress | Track button shows spinner, confidence updates per frame |
| Track lost | Confidence drops below threshold, warning indicator |

### C4. SegmentationPanel

| State | Visual |
|-------|--------|
| No file loaded | Engine grid shown but segment button disabled |
| Engine selected | Selected engine has highlighted border (engine-specific color) |
| Segmenting | "Segment" button shows spinner, disabled during processing |
| Segmentation complete | Overlay toggles become active, mask appears on viewer |
| Error | Red error message below segment button |
| Engine info | Expandable info box shows engine description, input requirements |
| Seed points listed | Numbered list of seed point coordinates with individual remove buttons |
| ROI status | Shows "ROI: set" or "ROI: not set" with dimensions if set |

Engine color mapping:
- nnunet: `blue-600`
- nnunet-wide: `indigo-600`
- angiopy: `blue-600`
- roi+angiopy: `cyan-600`
- sam2: `purple-600`
- nnunet-fullframe: `emerald-600`
- segformer: `orange-600`
- hrnet: `rose-600`

### C5. QCAPanel

| State | Visual |
|-------|--------|
| No segmentation | "No segmentation for frame X" message |
| No calibration | Yellow warning banner: "No calibration set. Values in pixels." |
| Calculating | Spinner on calculate button |
| Results displayed | Key metrics: MLD (red text), DS% with severity badge, reference diameters (proximal=cyan, distal=yellow), interpolated RD, lesion length |
| DS% severity badges | `< 50%`: mild (green), `50-69%`: moderate (yellow), `70-89%`: severe (orange), `>= 90%`: critical (red) |

### C6. RWSPanel

| State | Visual |
|-------|--------|
| No data | "Calculate QCA for multiple frames then run RWS analysis" message |
| Frame range set | Start/end frame inputs populated, calculate button enabled (green-600) |
| ECG beats available | Beat buttons (B1, B2, ...) shown, clickable to set frame range |
| Calculating | Spinner, calculate button disabled |
| Single result | Card with MLD RWS (large value), color-coded interpretation, proximal/distal/average values, Dmin/Dmax frame links |
| Multiple results | Scrollable list of result cards + summary statistics section at bottom |
| Outlier method selected | Dropdown shows current method name |
| Vessel selected | Vessel label shown on result cards |

RWS clinical color coding:
- Normal (< 8%): `green-500` (#22c55e)
- Intermediate (8-12%): `amber-500` (#f59e0b)
- Elevated (> 12%): `orange-500` (#f97316)
- High Risk: `red-500` (#ef4444)

### C7. QFRPanel

| State | Visual |
|-------|--------|
| QFR mode off (single view) | Mode toggle shows "Single View" selected, basic projection status |
| QFR mode on (dual view) | Main viewer replaced by QFRModeViewer, side-by-side projections |
| No projections loaded | Step-by-step instruction list shown |
| P1 loaded only | P1 status card shows metadata, P2 shows "Not loaded" |
| Both projections loaded | Both status cards green, angle separation shown |
| Angle < 25 degrees | Yellow warning about insufficient angle separation |
| Both calibrated | Calibration status shows green checkmark for each |
| Reconstructing | 3D Reconstruct button shows spinner, "Reconstructing..." text |
| QFR result available | Large QFR value (color-coded: green >= 0.80, yellow 0.75-0.79, red < 0.75), mode selector (fQFR/cQFR/aQFR), vessel metrics |
| 3D view mode | Three.js mesh replaces dual-projection view, orbit controls for rotation |
| Projection transfer active | One projection loaded in main viewer for editing, thumbnail of other in corner |

### C8. ECGPanel

| State | Visual |
|-------|--------|
| No ECG | "No ECG data" message, panel collapsed |
| ECG loaded | 300x80px canvas with green waveform, red R-peak markers, white frame cursor |
| Edit mode off | Standard display, click scrubs to frame position |
| Edit mode on | Blue highlight/border, guide popup visible: "Click to add, right-click to remove, drag to move" |
| Heart rate available | "HR: X bpm" overlay on the ECG canvas |
| Peak/beat count | "N peaks, M beats" overlay text |
| Motion signal visible | Second waveform overlaid (different color) when motion signal toggle is on |
| Loading motion | Spinner indicator while motion signal is being calculated |

### C9. MaskEditToolbar

| State | Visual |
|-------|--------|
| Visible (edit mode) | Centered floating bar, glass-morphism (`bg-gray-900/95 backdrop-blur-sm`), rounded, shadow |
| Tool selected | Active tool button highlighted with accent color |
| Undo available | Undo button enabled (normal opacity) |
| Undo unavailable | Undo button disabled (`opacity-50`, no pointer events) |
| Redo available/unavailable | Same pattern as undo |
| Morphology dropdown open | Dropdown appears below the toolbar with 5 operation options |
| Settings expanded | Settings panel floats below the toolbar with size sliders, hardness controls |
| Processing | Spinner overlay or disabled state during server-side operations |

### C10. SeriesPicker (Modal)

| State | Visual |
|-------|--------|
| Closed | Not rendered |
| Open, no folder selected | "Select Folder" button prominent, empty grid area |
| Parsing DICOM headers | Progress indicator while scanning folder files |
| Series grid populated | 3-column grid, each cell: thumbnail image, angle labels (RAO/LAO degrees + CRA/CAU degrees), frame count badge |
| Single selection mode | Click to select, selected series highlighted with blue border |
| Dual selection mode | Click to assign P1 (blue outline) or P2 (teal outline), P1/P2 labels on selected |
| Series selected | Highlighted cell, metadata shown (dimensions, modality, angles) |

### C11. Toolbar (Left Sidebar)

| State | Visual |
|-------|--------|
| Default | 14px wide column, icon buttons vertically stacked |
| Tool active | Active tool button has `blue-600` background |
| Tool inactive | Button has `surface-tertiary` background |
| Tool disabled | `opacity-50`, no hover effects, no click handler |
| Hover | Background lightens (`surface-tertiary/80`), tooltip appears |
| QFR mode on | 3D QFR toggle shows active state |

### C12. SettingsModal

| State | Visual |
|-------|--------|
| Closed | Not rendered |
| Open | Modal overlay (z-50), 600px max-width, 85vh max-height, scrollable content |
| Tab active | Selected tab has underline and bold text |
| Theme selected | Large icon buttons, selected theme has primary-color border |
| Toggle on/off | Switch component: on = blue/primary background, off = gray |
| Range slider | Standard range input with value label |

### C13. PrivacyDialog

| State | Visual |
|-------|--------|
| Closed | Not rendered |
| Open | Fixed overlay z-100, centered modal 500px width, lock icon, yellow warning text, 3 buttons: Cancel (gray), Keep Original (gray), Anonymize (blue-600 with checkmark icon) |

---

## D. Responsive Behavior

### D1. Layout Breakpoints

The application uses Tailwind's default breakpoint system:

| Breakpoint | Width | Behavior |
|------------|-------|----------|
| < 768px (mobile) | `md:` prefix | Right panel hidden by default, accessible via hamburger toggle. Panel slides in from right as fixed overlay. |
| >= 768px (tablet) | `md:` | Right panel statically positioned in the flex layout. Hamburger hidden. |
| >= 1024px (desktop) | `lg:` | Header shows file metadata info (hidden on smaller). Full layout with all panels visible. |

### D2. Responsive Components

**Header**:
- `sm:inline` -- Version text hidden below `sm` breakpoint.
- `lg:inline` -- File metadata (frames, resolution) hidden below `lg` breakpoint.
- Mobile hamburger button: `md:hidden`.

**Right Panel**:
- Desktop: `md:static`, part of the flex layout, `w-80` fixed width.
- Mobile: `fixed right-0 top-12 bottom-6 z-40`, slides in/out with `translate-x-full` / `translate-x-0`.
- Mobile backdrop: `absolute inset-0 bg-black/50 z-30 md:hidden backdrop-blur-sm`.
- Transition: `transition-transform duration-300 ease-in-out`.

**Toolbar**:
- Fixed `w-14` on all breakpoints (does not collapse on mobile).

**SeriesPicker**:
- 900px modal, but content is scrollable if viewport is narrower.
- Grid columns may reduce on smaller screens.

**Viewer**:
- `flex-1 min-w-0` -- takes all available horizontal space between toolbar and right panel.
- Canvas resizes to fill container, maintaining aspect ratio of the DICOM image.

### D3. Tailwind Container Configuration

```js
container: {
  center: true,
  padding: '2rem',
  screens: { '2xl': '1400px' }
}
```

Max container width is 1400px, centered with 2rem padding.

### D4. Mobile Panel Interaction

1. User taps hamburger icon in header.
2. `isMobilePanelOpen` toggled to `true`.
3. Backdrop div appears with `bg-black/50 backdrop-blur-sm` covering the viewer.
4. Right panel slides in from right (`translate-x-0` from `translate-x-full`).
5. Tapping the backdrop closes the panel.
6. Panel shows same tabbed content as desktop.

---

## E. Animation & Transition Specs

### E1. CSS Animations (Defined in tailwind.config.js)

```
accordion-down: height 0 -> var(--radix-accordion-content-height), 0.2s ease-out
accordion-up:   height var(--radix-accordion-content-height) -> 0, 0.2s ease-out
```

These are used by Radix UI accordion primitives for expanding/collapsing content sections.

### E2. Tailwind Utility Animations

| Class | Usage |
|-------|-------|
| `animate-spin` | Loading spinners on buttons during async operations |
| `animate-pulse` | Track mode badge pulsing, loading indicators |
| `transition-colors` | Buttons, tab switches, hover states (default 150ms ease) |
| `transition-transform` | Mobile panel slide in/out |
| `transition-opacity` | Fade in/out for overlays |

### E3. Component-Specific Transitions

**Mobile Panel Slide**:
```css
transition-transform duration-300 ease-in-out
transform: translateX(100%) -> translateX(0)
```

**Button Hover**:
```css
transition-colors (150ms default)
bg-surface-tertiary -> bg-surface-tertiary/80
text-content-secondary -> text-content-primary
```

**Tab Switch**:
```css
transition-colors (150ms)
Active: bg-surface-tertiary text-content-primary border-b-2 border-blue-500
Inactive: text-content-muted -> hover:text-content-primary hover:bg-surface-tertiary/50
```

**Context Menu Appearance**: No animation -- appears instantly at cursor position.

**Modal Appearance**: No explicit enter/exit animation on PrivacyDialog, SettingsModal, or SeriesPicker. They render/unmount based on boolean state.

**Segmentation Overlay Render**: No fade animation. Mask appears/disappears immediately when toggled.

### E4. Canvas Rendering

The canvas render loop runs at the display's refresh rate via `requestAnimationFrame`. LayerManager composites 4 layers on each render tick:
- Only layers marked as "dirty" are re-rendered.
- Performance target tracked via `fps` and `renderTime` metrics.
- No CSS animations on canvas content -- all visual changes are immediate per frame.

### E5. Missing/Recommended Animations

The current implementation has minimal animation. The following transitions could be added for polish:
- Modal entrance/exit (fade + scale).
- Toast/notification slide-in for error messages.
- Overlay opacity transition when toggling segmentation mask visibility.
- Smooth zoom transitions instead of immediate scale changes.
- Loading skeleton shimmer for panel content during calculations.

---

## F. Design Token System

### F1. CSS Custom Properties (src/index.css)

The application uses a **dual-tier token system**:

#### Tier 1: Semantic Surface/Content Tokens (Direct Hex Values)

**Light Theme** (`:root`):
```
--surface-primary:   #ffffff   (white -- main background)
--surface-secondary: #f1f5f9   (slate-100 -- panels, header, footer)
--surface-tertiary:  #e2e8f0   (slate-200 -- buttons, inputs, cards)
--content-primary:   #0f172a   (slate-900 -- headings, primary text)
--content-secondary: #475569   (slate-600 -- secondary text, labels)
--content-muted:     #64748b   (slate-500 -- disabled, placeholder)
--brand:             #2563eb   (blue-600 -- primary actions)
--border:            #e2e8f0   (slate-200 -- borders, dividers)
```

**Dark Theme** (`.dark`):
```
--surface-primary:   #0f172a   (slate-900)
--surface-secondary: #1e293b   (slate-800)
--surface-tertiary:  #334155   (slate-700)
--content-primary:   #f1f5f9   (slate-100)
--content-secondary: #cbd5e1   (slate-300)
--content-muted:     #94a3b8   (slate-400)
--brand:             #3b82f6   (blue-500)
--border:            #334155   (slate-700)
```

#### Tier 2: HSL Tokens (Radix UI Compatible)

These use HSL values without the `hsl()` wrapper (e.g., `210 40% 98%`):
```
--background, --foreground
--card, --card-foreground
--popover, --popover-foreground
--primary, --primary-foreground
--secondary, --secondary-foreground
--muted, --muted-foreground
--accent, --accent-foreground
--destructive, --destructive-foreground
--border, --input, --ring
--radius (0.5rem default)
```

### F2. Tailwind Extended Colors (tailwind.config.js)

```js
colors: {
  // Token-mapped
  border:      'var(--border)',
  input:       'hsl(var(--input))',
  ring:        'hsl(var(--ring))',
  background:  'hsl(var(--background))',
  foreground:  'hsl(var(--foreground))',

  // Primary with shade scale
  primary: {
    DEFAULT:    'hsl(var(--primary))',
    foreground: 'hsl(var(--primary-foreground))',
    50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe',
    300: '#93c5fd', 400: '#60a5fa', 500: '#3b82f6',
    600: '#2563eb', 700: '#1d4ed8', 800: '#1e40af', 900: '#1e3a8a',
  },

  // Semantic
  secondary:   { DEFAULT, foreground },
  destructive: { DEFAULT, foreground },
  muted:       { DEFAULT, foreground },
  accent:      { DEFAULT, foreground },
  popover:     { DEFAULT, foreground },
  card:        { DEFAULT, foreground },

  // Surface system
  surface: {
    primary:   'var(--surface-primary)',
    secondary: 'var(--surface-secondary)',
    tertiary:  'var(--surface-tertiary)',
  },
  content: {
    primary:   'var(--content-primary)',
    secondary: 'var(--content-secondary)',
    muted:     'var(--content-muted)',
  },
  brand: 'var(--brand)',

  // Vessel colors (domain-specific)
  vessel: {
    lad:      '#3b82f6',  // blue-500
    lcx:      '#8b5cf6',  // violet-500
    rca:      '#ef4444',  // red-500
    lm:       '#f59e0b',  // amber-500
    other:    '#6b7280',  // gray-500
    stenosis: '#dc2626',  // red-600
  },
}
```

### F3. Border Radius Tokens

```js
borderRadius: {
  lg: 'var(--radius)',          // 0.5rem = 8px
  md: 'calc(var(--radius) - 2px)',  // 6px
  sm: 'calc(var(--radius) - 4px)',  // 4px
}
```

### F4. Spacing Constants (Implicit in Components)

| Element | Size |
|---------|------|
| Header height | `h-12` (48px) |
| Status bar height | `h-6` (24px) |
| Toolbar width | `w-14` (56px) |
| Right panel width | `w-80` (320px) |
| Playback controls height | `h-24` (96px) |
| Panel padding | `p-3` (12px) |
| Tab buttons padding | `px-3 py-2` (12px horizontal, 8px vertical) |
| Button padding | `p-2` (8px) typically |
| Icon size | 18px (lucide-react `size={18}`) |

### F5. Typography

| Element | Classes |
|---------|---------|
| App title | `text-lg font-semibold` |
| Version badge | `text-xs text-content-muted` |
| Tab labels | `text-xs font-medium` |
| Panel headings | `text-sm font-semibold` or `text-base font-medium` |
| Body text | `text-sm` (14px default) |
| Metric values | `text-2xl font-bold` or `text-xl font-semibold` |
| Status bar | `text-xs text-content-muted` |
| Keyboard hints | `<kbd>` with `bg-surface-tertiary px-1 rounded` |
| Badge text | `text-xs` |

### F6. Shadow Tokens

| Context | Shadow |
|---------|--------|
| Right panel (mobile) | `shadow-2xl` |
| Right panel (desktop) | `shadow-none` |
| MaskEditToolbar | Inherits from glass-morphism styling |
| Modals | Standard shadow (not explicitly specified, varies) |

### F7. Custom Scrollbar Styling

Defined in `src/index.css`:
```css
::-webkit-scrollbar       { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.5); }
```

### F8. Theme Implementation

Theme is managed by `useTheme` hook:
1. Reads `appearance.theme` from `settingsStore` (values: `'light'`, `'dark'`, `'system'`).
2. For `'system'`, listens to `window.matchMedia('(prefers-color-scheme: dark)')`.
3. Applies or removes the `'dark'` class on `document.documentElement`.
4. Tailwind's `darkMode: 'class'` configuration enables `.dark:` variant prefixes.
5. Default theme is `'dark'`.

---

## G. Accessibility Audit

### G1. Current Accessibility Features

**Keyboard Navigation**:
- All toolbar buttons are focusable `<button>` elements.
- Keyboard shortcuts for all major tools (B, S, H, Space, arrows, etc.).
- Escape key to deselect/exit modes.
- Tab navigation works through standard DOM order.

**ARIA Labels**:
- Fullscreen button: `aria-label="Toggle Full Screen"`.
- Mobile panel toggle: `aria-label="Toggle Panel"`.
- Toolbar buttons have `title` attributes with tool name + shortcut.
- Some buttons have `aria-label` but coverage is inconsistent.

**Color Contrast**:
- Dark theme: `#f1f5f9` text on `#0f172a` background = ~15.3:1 ratio (excellent).
- Light theme: `#0f172a` text on `#ffffff` background = ~16.8:1 ratio (excellent).
- Muted text (dark): `#94a3b8` on `#0f172a` = ~6.8:1 (passes AA, passes AAA for large text).
- Muted text (light): `#64748b` on `#ffffff` = ~5.5:1 (passes AA).

**Focus Indicators**: Tailwind's default `ring` utility is configured but not consistently applied to all interactive elements.

### G2. Accessibility Gaps

**Critical**:
1. **Canvas-based viewer has no keyboard alternative**. All viewer interactions (ROI drawing, seed placement, zoom, pan) require mouse. No keyboard-only path exists.
2. **ECG canvas has no screen reader access**. The ECG waveform and R-peak editing are purely visual canvas operations.
3. **No ARIA roles for custom widgets**. The tab bar, toolbar, and modal dialogs don't use proper ARIA roles (`tablist`/`tab`/`tabpanel`, `toolbar`, `dialog`).
4. **Context menus are custom implementations** without ARIA menu roles (`menu`/`menuitem`).
5. **No skip-to-content links** for keyboard users to bypass the header/toolbar.

**Major**:
6. **Color-only differentiation**: QFR result color coding (green/yellow/red), RWS severity colors, and engine colors rely solely on color. No accompanying icons, patterns, or text labels for colorblind users in the viewer overlays.
7. **No announce mechanism**: Async operations (segmentation complete, RWS calculated, tracking lost) have no `aria-live` regions to announce results to screen readers.
8. **Mask edit cursor**: Custom circle cursor replaces system cursor. Screen readers and magnification software cannot detect it.
9. **Settings range sliders**: May not announce current value to screen readers.

**Minor**:
10. **Inconsistent `title`/`aria-label`**: Some buttons have both, some have only one, some have neither.
11. **Focus trapping in modals**: Settings modal, Privacy dialog, and Series picker do not trap focus within the modal (Tab key can escape to background elements).
12. **No reduced motion support**: Animations (pulse, spin, slide) run regardless of `prefers-reduced-motion` setting.
13. **Drag-and-drop file loading**: No keyboard alternative for drag-and-drop (though the file input button provides an alternative path).

### G3. Recommendations for Rewrite

1. Implement ARIA roles: `dialog` for modals, `tablist`/`tab`/`tabpanel` for panel tabs, `toolbar` for the toolbar, `menu`/`menuitem` for context menus.
2. Add `aria-live="polite"` regions for announcing calculation results, errors, and mode changes.
3. Add visible focus rings (`ring-2 ring-blue-500 ring-offset-2`) to all interactive elements.
4. Implement focus trapping in all modal dialogs.
5. Add keyboard alternatives for canvas interactions where feasible (e.g., seed point coordinates via number input).
6. Add `@media (prefers-reduced-motion: reduce)` to disable or simplify all animations.
7. Use semantic icons/labels alongside color coding for clinical severity levels.
8. Add skip-to-content navigation link.

---

## H. Medical UX Considerations

### H1. HIPAA Compliance: Privacy Dialog

Every file load triggers a privacy confirmation dialog before any data is sent to the backend. This is a hard gate -- there is no way to skip it.

- **Anonymize option**: Strips patient name, ID, date of birth, and institution from DICOM headers server-side before any processing.
- **Keep original option**: Preserves all DICOM metadata for clinical use.
- **Visual emphasis**: Yellow warning styling draws attention to the privacy implications. Lock icon reinforces security context.
- **Cancel path**: User can abort the load entirely.

### H2. Clinical Thresholds and Color Coding

**RWS Interpretation**:
| Range | Interpretation | Color | Clinical Action |
|-------|---------------|-------|----------------|
| < 8% | Normal | Green (#22c55e) | No intervention needed |
| 8-12% | Intermediate | Amber (#f59e0b) | Consider further assessment |
| > 12% | Elevated Risk | Orange (#f97316) | Clinical review recommended |
| (high_risk) | High Risk | Red (#ef4444) | Urgent evaluation |

**QFR Interpretation**:
| Range | Interpretation | Color |
|-------|---------------|-------|
| >= 0.80 | Not significant | Green |
| 0.75 - 0.79 | Gray zone | Yellow |
| < 0.75 | Hemodynamically significant | Red |

**QCA Diameter Stenosis Severity**:
| Range | Severity | Badge Color |
|-------|----------|-------------|
| < 50% | Mild | Green |
| 50-69% | Moderate | Yellow |
| 70-89% | Severe | Orange |
| >= 90% | Critical | Red |

### H3. Vessel Color Coding (Domain Standard)

Distinct colors for coronary vessel territories:
- LAD (Left Anterior Descending): Blue (#3b82f6)
- LCX (Left Circumflex): Violet (#8b5cf6)
- RCA (Right Coronary Artery): Red (#ef4444)
- LM (Left Main): Amber (#f59e0b)
- Other: Gray (#6b7280)
- Stenosis: Red-600 (#dc2626)

### H4. Clinical Data Integrity

1. **Frame-level precision**: All measurements are frame-indexed, not time-indexed. This avoids floating-point drift in temporal calculations.
2. **ECG-synchronized analysis**: RWS is calculated per cardiac beat (R-peak to R-peak), not arbitrary frame ranges. ECG beat boundaries are derived from R-peak detection and converted to frame indices.
3. **R-peak editing**: Users can correct automatic R-peak detection before RWS calculation. Changes immediately recalculate beat boundaries.
4. **Calibration chain**: All distance measurements (MLD, lesion length, vessel diameter) depend on spatial calibration. Yellow warnings appear when calibration is missing, preventing misinterpretation of pixel-unit values as millimeter values.
5. **QFR angle validation**: System warns when the angular separation between two projections is < 25 degrees, as this produces unreliable stereo reconstruction.
6. **Outlier filtering**: RWS calculation supports 5 outlier methods to handle noisy diameter measurements. Default (Hampel) is clinically validated.

### H5. Non-Destructive Editing

1. **Mask edit undo/redo**: Up to 20 undo levels ensure the user can revert mask edits without losing work.
2. **Original mask preservation**: When entering mask edit mode, the original mask is saved. Cancel restores it completely.
3. **Auto-save on frame change**: During mask editing, navigating to a different frame auto-saves the current mask, preventing accidental loss.
4. **Store reset isolation**: Loading a new file resets all analysis stores, preventing stale data from one study contaminating another.

### H6. Workflow Guardrails

1. **Engine-specific prerequisites**: The UI communicates required inputs per segmentation engine (e.g., "AngioPy requires 2+ seed points") and disables the segment button until prerequisites are met.
2. **Step-by-step QFR guidance**: When projections are not fully configured, the QFRPanel shows a numbered instruction list rather than disabled buttons.
3. **Segmentation auto-chain**: After segmentation, centerline extraction and QCA are automatically triggered, reducing manual steps in the analysis pipeline.
4. **Track mode auto-chain**: When track mode + auto-seg is enabled, each frame advance triggers tracking -> segmentation -> QCA automatically, enabling hands-free multi-frame analysis.

### H7. Reference Citations

The RWSPanel includes a citation reference for the RWS methodology, supporting clinical credibility and allowing users to verify the scientific basis of the calculations.

### H8. Resolution Warning

A `ResolutionWarningModal` component (rendered in AnalysisApp) warns users when their screen resolution may be insufficient for accurate medical image analysis. This helps ensure that clinical decisions are not made on inadequate display hardware.

### H9. Data Export for Clinical Review

- CSV export contains frame-indexed QCA and RWS data for external analysis.
- Report generation provides structured output suitable for clinical documentation.
- Training export supports ML model improvement with annotated frame+mask pairs.

### H10. Catheter Calibration Standard

Calibration uses standard catheter French sizes (5F through 8F) as reference objects, which is the clinical standard for coronary angiography pixel-to-millimeter conversion. Default is 6F (the most commonly used diagnostic catheter).

---

## Appendix: Store Architecture Summary

### State Management Pattern

All 17 stores use Zustand with `subscribeWithSelector` middleware. Pattern:

```typescript
export const useXxxStore = create<XxxState>()(
  subscribeWithSelector((set, get) => ({
    // State fields
    // Action methods (async for API calls)
    // Getter methods
    // Reset method
  }))
);
```

Only `settingsStore` uses `persist` middleware (localStorage, versioned migrations).

### Store Dependency Graph

```
dicomStore ──> playerStore (sets frame info on load)
             ──> resetAllAnalysis() on new file load

ecgStore ──> dicomApi (loads ECG from DICOM)
          ──> playerStore (frame/sample conversion)

segmentationStore ──> segmentationApi
                   ──> settingsStore (engine preference)

annotationStore (independent -- no API calls)

qcaStore ──> qcaApi ──> segmentationStore (needs mask/centerline)

rwsStore ──> rwsApi ──> ecgStore (beat boundaries)

trackingStore ──> trackingApi
               ──> settingsStore (confidence threshold)

qfrModeStore (independent -- self-contained dual-projection state)
              ──> transfers to/from main viewer stores

maskEditStore ──> settingsStore (persisted tool sizes)

motionStore ──> motionApi
overlayStore (independent -- visibility toggles)
calibrationStore ──> dicomStore (reads pixel spacing)
exportStore ──> exportApi
reportStore ──> reportApi
```

### Canvas Layer Stack

```
z-index 0: VideoLayer      (DICOM frame, zoom/pan)
z-index 1: SegmentationLayer (mask overlay, centerline, diameter markers)
z-index 2: AnnotationLayer   (seed points, ROI, cross-sections)
z-index 3: OverlayLayer      (ECG trace, UI elements, frame info)
```

Additional virtual layer: `MASK_EDIT` (rendered as SVG overlay on top of canvas, not composited by LayerManager).

LayerManager renders in z-index order, skipping hidden layers. Each layer has independent opacity (default: 0.3 for LOW quality, 0.7 for HIGH quality) and blend mode (default: source-over alpha compositing).

---

## Appendix: File Inventory

### Components (34 files)
- `src/components/Panels/`: SegmentationPanel, QCAPanel, QFRPanel, RWSPanel, ECGPanel, MetadataDisplay, ReportPanel, TrainingExportPanel, SeriesPicker, CalibrationPanel, ExportPanel
- `src/components/Viewer/`: VideoPlayer (1869 lines), VideoPlayerWithTabs, QFRModeViewer, QFRThumbnailPreview
- `src/components/Controls/`: PlaybackControls, Toolbar
- `src/components/Charts/`: RWSChart, DiameterChart
- `src/components/common/`: Badge, KeyboardShortcuts, SettingsModal, ResolutionWarningModal
- `src/components/MaskEdit/`: MaskEditToolbar
- `src/components/Dialogs/`: PrivacyDialog

### Stores (17 files)
`src/stores/`: playerStore, dicomStore, segmentationStore, annotationStore, ecgStore, qcaStore, rwsStore, trackingStore, qfrModeStore, maskEditStore, motionStore, overlayStore, calibrationStore, exportStore, reportStore, settingsStore (persisted), index (barrel)

### Hooks (4 files)
`src/hooks/`: useCanvasLayers, useKeyboardShortcuts, useTheme, index

### Canvas System (9 files)
`src/lib/canvas/`: LayerManager, VideoLayer, SegmentationLayer, AnnotationLayer, OverlayLayer, MaskEditLayer, types, index, utils

### Configuration
- `tailwind.config.js` -- Design system, colors, animations, plugin
- `src/index.css` -- CSS custom properties, scrollbar, theme tokens
- `vite.config.ts` -- `__API_BASE__` define, path aliases
- `tsconfig.json` -- `@/*` path mapping
