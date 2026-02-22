import { useState, useRef, useEffect, useCallback } from 'react';
import { useQFRStore } from '@/stores/qfrStore';
import { fetchProjectionFrame } from '@/lib/api/qfr';
import { QFRMesh3DViewer } from '@/components/viewer/QFRMesh3DViewer';

const FRENCH_SIZES = [5, 6, 7, 8] as const;

/* ---------- Projection Canvas (one side of the dual viewer) ---------- */

interface ProjectionCanvasProps {
  projectionId: 1 | 2;
  width: number;
  height: number;
}

function ProjectionCanvas({ projectionId, width, height }: ProjectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bitmapRef = useRef<ImageBitmap | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const proj = useQFRStore((s) => (projectionId === 1 ? s.projection1 : s.projection2));

  const [frameIndex, setFrameIndex] = useState(proj.segmentedFrameIndex ?? 0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isDraggingMld, setIsDraggingMld] = useState(false);
  const [dragMldIdx, setDragMldIdx] = useState<number | null>(null);
  const [isHoveringMld, setIsHoveringMld] = useState(false);
  const isUploading = useQFRStore((s) => s.isUploading);
  const isSegmenting = useQFRStore((s) => s.isSegmenting);
  const activeTool = useQFRStore((s) => s.activeTool);
  const upload = useQFRStore((s) => s.uploadProjection);
  const segment = useQFRStore((s) => s.segmentProjection);
  const calibrate = useQFRStore((s) => s.calibrateProjection);
  const setTimiFrame = useQFRStore((s) => s.setTimiFrame);
  const addSeedPoint = useQFRStore((s) => s.addSeedPoint);
  const clearSeedPoints = useQFRStore((s) => s.clearSeedPoints);
  const setOverlayToggle = useQFRStore((s) => s.setOverlayToggle);
  const setActiveTool = useQFRStore((s) => s.setActiveTool);
  const segEngine = useQFRStore((s) => s.segEngine);
  const setSegEngine = useQFRStore((s) => s.setSegEngine);
  const qfrResult = useQFRStore((s) => s.qfrResult);
  const mldOverrideIndex = useQFRStore((s) => s.mldOverrideIndex);
  const recalculateQfr = useQFRStore((s) => s.recalculateQfr);
  const setMldOverride = useQFRStore((s) => s.setMldOverride);
  const isRecalculating = useQFRStore((s) => s.isRecalculating);

  const [catheterFr, setCatheterFr] = useState(6);
  const [showCalibrate, setShowCalibrate] = useState(false);

  // Compute fit transform for canvas coordinates → image coordinates
  const fitTransformRef = useRef({ scale: 1, dx: 0, dy: 0 });

  /* --- MLD drag helpers --- */
  const canvasToImageCoords = useCallback(
    (clientX: number, clientY: number): { imgX: number; imgY: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const cssX = clientX - rect.left;
      const cssY = clientY - rect.top;
      const cssToCanvasX = canvas.width / rect.width;
      const cssToCanvasY = canvas.height / rect.height;
      const canvasX = cssX * cssToCanvasX;
      const canvasY = cssY * cssToCanvasY;
      const { scale, dx, dy } = fitTransformRef.current;
      return {
        imgX: (canvasX - dx) / scale,
        imgY: (canvasY - dy) / scale,
      };
    },
    [],
  );

  const findClosestCenterlineIndex = useCallback(
    (imgX: number, imgY: number): number | null => {
      const cl = proj.centerline;
      if (cl.length < 2) return null;
      const { scale } = fitTransformRef.current;
      const threshold = 10 / scale; // 10 CSS pixels in image coords
      let bestDist = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < cl.length; i++) {
        const dx = cl[i].x - imgX;
        const dy = cl[i].y - imgY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }
      return bestDist <= threshold ? bestIdx : null;
    },
    [proj.centerline],
  );

  const getMldCenterlineIndex = useCallback((): number => {
    // If there's an override, map QFR profile index back to centerline index
    if (mldOverrideIndex !== null && qfrResult) {
      const qfrN = qfrResult.diameters_mm.length;
      const clN = proj.centerline.length;
      if (qfrN > 1 && clN > 1) {
        const fraction = mldOverrideIndex / (qfrN - 1);
        return Math.round(fraction * (clN - 1));
      }
    }
    // Auto-detected: find min diameter in pixel diameters
    const diams = proj.diametersPx;
    let mldIdx = 0;
    let mldVal = Infinity;
    for (let i = 0; i < diams.length; i++) {
      if (diams[i] > 0 && diams[i] < mldVal) {
        mldVal = diams[i];
        mldIdx = i;
      }
    }
    return mldIdx;
  }, [mldOverrideIndex, qfrResult, proj.centerline.length, proj.diametersPx]);

  const centerlineIdxToQfrIdx = useCallback(
    (clIdx: number): number => {
      if (!qfrResult) return 0;
      const fraction = clIdx / Math.max(proj.centerline.length - 1, 1);
      const qfrN = qfrResult.diameters_mm.length;
      return Math.round(fraction * (qfrN - 1));
    },
    [qfrResult, proj.centerline.length],
  );

  /* --- Render a bitmap + overlays onto the canvas --- */
  const renderBitmap = useCallback(
    (bitmap: ImageBitmap) => {
      const canvas = canvasRef.current;
      if (!canvas || width === 0 || height === 0) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);

      const fitScale = Math.min(width / bitmap.width, height / bitmap.height);
      const dx = (width - bitmap.width * fitScale) / 2;
      const dy = (height - bitmap.height * fitScale) / 2;
      fitTransformRef.current = { scale: fitScale, dx, dy };

      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(bitmap, dx, dy, bitmap.width * fitScale, bitmap.height * fitScale);
      bitmapRef.current = bitmap;

      // Draw overlays
      ctx.save();
      ctx.translate(dx, dy);
      ctx.scale(fitScale, fitScale);

      // Segmentation overlays only visible on the frame that was segmented
      const onSegFrame = proj.segmentedFrameIndex === frameIndex;

      // Mask overlay (semi-transparent purple)
      // Mask is grayscale PNG (no alpha) — convert luminance to alpha so
      // only non-zero (vessel) pixels are tinted.
      if (onSegFrame && proj.showMask && proj.maskBitmap) {
        const mw = proj.maskBitmap.width;
        const mh = proj.maskBitmap.height;
        const offscreen = new OffscreenCanvas(mw, mh);
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          offCtx.drawImage(proj.maskBitmap, 0, 0);
          const imgData = offCtx.getImageData(0, 0, mw, mh);
          const d = imgData.data;
          // Convert grayscale to purple with alpha from luminance
          for (let i = 0; i < d.length; i += 4) {
            const lum = d[i]; // grayscale value
            d[i] = 147;     // R (purple-600 #9333ea)
            d[i + 1] = 51;  // G
            d[i + 2] = 234; // B
            d[i + 3] = lum;  // alpha from mask luminance
          }
          offCtx.putImageData(imgData, 0, 0);
          ctx.globalAlpha = 0.5;
          ctx.drawImage(offscreen, 0, 0);
          ctx.globalAlpha = 1.0;
        }
      }

      // Centerline overlay (yellow line)
      if (onSegFrame && proj.showCenterline && proj.centerline.length >= 2) {
        ctx.strokeStyle = '#eab308'; // yellow-500
        ctx.lineWidth = 2 / fitScale;
        ctx.beginPath();
        ctx.moveTo(proj.centerline[0].x, proj.centerline[0].y);
        for (let i = 1; i < proj.centerline.length; i++) {
          ctx.lineTo(proj.centerline[i].x, proj.centerline[i].y);
        }
        ctx.stroke();
      }

      // Diameter cross-section markers
      if (onSegFrame && proj.showDiameters && proj.centerline.length >= 2 && proj.diametersPx.length > 0) {
        const cl = proj.centerline;
        const diams = proj.diametersPx;

        // Find auto-detected MLD index (minimum diameter)
        let autoMldIdx = 0;
        let autoMldVal = Infinity;
        for (let i = 0; i < cl.length && i < diams.length; i++) {
          if (diams[i] > 0 && diams[i] < autoMldVal) {
            autoMldVal = diams[i];
            autoMldIdx = i;
          }
        }

        // Determine the active MLD index (override or auto)
        const activeMldIdx = getMldCenterlineIndex();
        const hasOverride = mldOverrideIndex !== null;

        // Helper: draw perpendicular diameter line at index i
        const drawCrossSec = (i: number, color: string, lw: number, dashed = false) => {
          const d = diams[Math.min(i, diams.length - 1)];
          if (!d || d <= 0) return;
          const prev = cl[Math.max(0, i - 1)];
          const next = cl[Math.min(cl.length - 1, i + 1)];
          const tx = next.x - prev.x;
          const ty = next.y - prev.y;
          const tLen = Math.sqrt(tx * tx + ty * ty);
          if (tLen === 0) return;
          const nx = -ty / tLen;
          const ny = tx / tLen;
          const halfD = d / 2;
          ctx.strokeStyle = color;
          ctx.lineWidth = lw / fitScale;
          if (dashed) ctx.setLineDash([4 / fitScale, 3 / fitScale]);
          ctx.beginPath();
          ctx.moveTo(cl[i].x - nx * halfD, cl[i].y - ny * halfD);
          ctx.lineTo(cl[i].x + nx * halfD, cl[i].y + ny * halfD);
          ctx.stroke();
          if (dashed) ctx.setLineDash([]);
        };

        // Draw sampled regular markers (~20 lines)
        const step = Math.max(1, Math.floor(cl.length / 20));
        for (let i = 0; i < cl.length && i < diams.length; i += step) {
          if (i === activeMldIdx || i === autoMldIdx) continue;
          drawCrossSec(i, '#f59e0b', 1); // amber
        }

        // When override is active, show auto MLD dimmed
        if (hasOverride && autoMldIdx < cl.length) {
          drawCrossSec(autoMldIdx, 'rgba(239,68,68,0.35)', 1.5);
        }

        // Draw active MLD marker
        const mldColor = hasOverride ? '#06b6d4' : '#ef4444'; // cyan if override, red if auto
        drawCrossSec(activeMldIdx, mldColor, 2);

        // MLD label
        if (activeMldIdx < cl.length) {
          const mldDiamMm = qfrResult?.mld_mm ?? proj.diametersMm[Math.min(activeMldIdx, proj.diametersMm.length - 1)];
          const prev = cl[Math.max(0, activeMldIdx - 1)];
          const next = cl[Math.min(cl.length - 1, activeMldIdx + 1)];
          const tx = next.x - prev.x;
          const ty = next.y - prev.y;
          const tLen = Math.sqrt(tx * tx + ty * ty);
          if (tLen > 0 && mldDiamMm != null) {
            const nx = -ty / tLen;
            const ny = tx / tLen;
            const halfD = (diams[Math.min(activeMldIdx, diams.length - 1)] ?? 0) / 2;
            const ex = cl[activeMldIdx].x + nx * halfD;
            const ey = cl[activeMldIdx].y + ny * halfD;
            ctx.font = `${11 / fitScale}px sans-serif`;
            ctx.fillStyle = mldColor;
            ctx.textAlign = 'left';
            const label = hasOverride ? 'MLD (manual)' : 'MLD';
            ctx.fillText(`${label} ${mldDiamMm.toFixed(2)}mm`, ex + 4 / fitScale, ey - 2 / fitScale);
          }
        }

        // Drag preview marker (cyan dashed)
        if (isDraggingMld && dragMldIdx !== null && dragMldIdx !== activeMldIdx && dragMldIdx < cl.length) {
          drawCrossSec(dragMldIdx, '#06b6d4', 2, true);
        }
      }

      // Seed point overlay (red dots)
      if (proj.showSeedPoints && proj.seedPoints.length > 0) {
        ctx.fillStyle = '#ef4444'; // red-500
        const radius = 4 / fitScale;
        for (const pt of proj.seedPoints) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.restore();
    },
    [width, height, frameIndex, proj.segmentedFrameIndex,
     proj.showMask, proj.maskBitmap, proj.showCenterline, proj.centerline,
     proj.showSeedPoints, proj.seedPoints, proj.showDiameters, proj.diametersPx, proj.diametersMm,
     qfrResult, fitTransformRef, mldOverrideIndex, getMldCenterlineIndex, isDraggingMld, dragMldIdx],
  );

  /* --- Fetch + render when frame changes --- */
  useEffect(() => {
    if (!proj.loaded) return;
    let cancelled = false;
    fetchProjectionFrame(projectionId, frameIndex)
      .then((bitmap) => {
        if (!cancelled) {
          bitmapRef.current = bitmap;
          renderBitmap(bitmap);
        }
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [proj.loaded, projectionId, frameIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  /* --- Re-render on overlay changes (without refetching frame) --- */
  useEffect(() => {
    if (bitmapRef.current) renderBitmap(bitmapRef.current);
  }, [renderBitmap]);

  /* --- MLD drag: is mouse near the MLD marker? --- */
  const isNearMld = useCallback(
    (imgX: number, imgY: number): boolean => {
      if (!qfrResult || !proj.showDiameters || proj.centerline.length < 2) return false;
      const mldClIdx = getMldCenterlineIndex();
      if (mldClIdx >= proj.centerline.length) return false;
      const pt = proj.centerline[mldClIdx];
      const { scale } = fitTransformRef.current;
      const threshold = 12 / scale;
      const dx = pt.x - imgX;
      const dy = pt.y - imgY;
      return Math.sqrt(dx * dx + dy * dy) <= threshold;
    },
    [qfrResult, proj.showDiameters, proj.centerline, getMldCenterlineIndex],
  );

  /* --- Canvas mouse handlers (seed points + MLD drag) --- */
  const handleCanvasMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      // Check for MLD drag initiation first
      if (qfrResult && proj.showDiameters && proj.centerline.length >= 2) {
        const coords = canvasToImageCoords(e.clientX, e.clientY);
        if (coords && isNearMld(coords.imgX, coords.imgY)) {
          setIsDraggingMld(true);
          setDragMldIdx(getMldCenterlineIndex());
          e.preventDefault();
          return;
        }
      }
    },
    [qfrResult, proj.showDiameters, proj.centerline.length, canvasToImageCoords, isNearMld, getMldCenterlineIndex],
  );

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isDraggingMld) {
        const coords = canvasToImageCoords(e.clientX, e.clientY);
        if (coords) {
          const clIdx = findClosestCenterlineIndex(coords.imgX, coords.imgY);
          if (clIdx !== null) {
            setDragMldIdx(clIdx);
          }
        }
        return;
      }
      // Hover detection for cursor change
      if (qfrResult && proj.showDiameters && proj.centerline.length >= 2) {
        const coords = canvasToImageCoords(e.clientX, e.clientY);
        if (coords) {
          setIsHoveringMld(isNearMld(coords.imgX, coords.imgY));
        }
      }
    },
    [isDraggingMld, canvasToImageCoords, findClosestCenterlineIndex, qfrResult, proj.showDiameters, proj.centerline.length, isNearMld],
  );

  const handleCanvasMouseUp = useCallback(
    (_e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isDraggingMld && dragMldIdx !== null) {
        const qfrIdx = centerlineIdxToQfrIdx(dragMldIdx);
        recalculateQfr(qfrIdx);
        setIsDraggingMld(false);
        setDragMldIdx(null);
        setIsHoveringMld(false);
        return;
      }
      setIsDraggingMld(false);
      setDragMldIdx(null);
    },
    [isDraggingMld, dragMldIdx, centerlineIdxToQfrIdx, recalculateQfr],
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      // Don't add seeds if we just finished an MLD drag
      if (activeTool !== 'seed') return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const coords = canvasToImageCoords(e.clientX, e.clientY);
      if (!coords) return;
      const imgX = Math.round(coords.imgX);
      const imgY = Math.round(coords.imgY);

      if (imgX >= 0 && imgY >= 0 && imgX < proj.imageWidth && imgY < proj.imageHeight) {
        addSeedPoint(projectionId, { x: imgX, y: imgY });
      }
    },
    [activeTool, proj.imageWidth, proj.imageHeight, addSeedPoint, projectionId, canvasToImageCoords],
  );

  /* --- File handling --- */
  const handleFile = useCallback(
    (file: File) => {
      upload(projectionId, file);
      setFrameIndex(0);
    },
    [upload, projectionId],
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  /* --- TIMI frame count --- */
  const timiFrameCount =
    proj.timiStart !== null && proj.timiEnd !== null ? proj.timiEnd - proj.timiStart : null;

  /* --- Segmentation status badge --- */
  const statusBadge = proj.segmented
    ? 'bg-green-600/80 text-white'
    : proj.loaded
      ? 'bg-yellow-600/80 text-white'
      : 'bg-gray-600/80 text-gray-300';

  const statusLabel = proj.segmented
    ? `Segmented (${proj.numCenterlinePoints} pts)`
    : proj.loaded
      ? 'Not segmented'
      : 'Empty';

  // Trimmed mean of QCA pixel diameters (exclude first/last 5) for calibration
  const trimmedMeanPx = (() => {
    const d = proj.diametersPx;
    if (d.length <= 10) return null;
    const trimmed = d.slice(5, -5);
    return trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  })();

  const isUncalibrated = proj.pixelSpacing === 0.3;

  return (
    <div
      className="flex-1 min-h-0 flex flex-col bg-black relative border border-border rounded overflow-hidden"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {/* Status badges (top-left) */}
      <div className="absolute top-2 left-2 z-10 flex items-center gap-1">
        <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${statusBadge}`}>
          Proj {projectionId}: {statusLabel}
        </span>
        {isUncalibrated && proj.loaded && (
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-orange-600/80 text-white">
            Uncalibrated
          </span>
        )}
        {timiFrameCount !== null && (
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-600/80 text-white">
            TIMI: {timiFrameCount} frames
          </span>
        )}
      </div>

      {/* Overlay toggles (top-right) */}
      {proj.loaded && proj.segmented && (
        <div className="absolute top-2 right-2 z-10 flex items-center gap-1">
          <label className="flex items-center gap-0.5 text-[10px] text-white/70">
            <input
              type="checkbox"
              checked={proj.showMask}
              onChange={(e) => setOverlayToggle(projectionId, 'showMask', e.target.checked)}
              className="w-3 h-3"
            />
            Mask
          </label>
          <label className="flex items-center gap-0.5 text-[10px] text-white/70">
            <input
              type="checkbox"
              checked={proj.showCenterline}
              onChange={(e) => setOverlayToggle(projectionId, 'showCenterline', e.target.checked)}
              className="w-3 h-3"
            />
            CL
          </label>
          <label className="flex items-center gap-0.5 text-[10px] text-white/70">
            <input
              type="checkbox"
              checked={proj.showDiameters}
              onChange={(e) => setOverlayToggle(projectionId, 'showDiameters', e.target.checked)}
              className="w-3 h-3"
            />
            Diam
          </label>
          <label className="flex items-center gap-0.5 text-[10px] text-white/70">
            <input
              type="checkbox"
              checked={proj.showSeedPoints}
              onChange={(e) => setOverlayToggle(projectionId, 'showSeedPoints', e.target.checked)}
              className="w-3 h-3"
            />
            Seeds
          </label>
          {mldOverrideIndex !== null && (
            <button
              onClick={() => setMldOverride(null)}
              className="px-1.5 py-0.5 text-[10px] rounded bg-cyan-600/80 text-white hover:bg-cyan-500/80 transition-colors"
            >
              Reset MLD
            </button>
          )}
          {isRecalculating && (
            <span className="px-1.5 py-0.5 text-[10px] rounded bg-cyan-600/80 text-white animate-pulse">
              Recalculating...
            </span>
          )}
        </div>
      )}

      {/* Canvas / Upload area */}
      {proj.loaded ? (
        <canvas
          ref={canvasRef}
          className={`flex-1 min-h-0 w-full ${
            isDraggingMld ? 'cursor-grabbing' : isHoveringMld ? 'cursor-grab' : activeTool === 'seed' ? 'cursor-crosshair' : ''
          }`}
          onClick={handleCanvasClick}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={() => { setIsDraggingMld(false); setDragMldIdx(null); setIsHoveringMld(false); }}
        />
      ) : (
        <div
          className={`flex-1 min-h-0 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors ${
            isDragOver ? 'bg-brand/10' : 'bg-surface-primary'
          }`}
          onClick={() => fileRef.current?.click()}
        >
          <div className="text-4xl text-content-muted">+</div>
          <p className="text-sm text-content-muted">
            {isUploading ? 'Uploading...' : 'Drop DICOM here or click to upload'}
          </p>
          <p className="text-xs text-content-muted">Projection {projectionId}</p>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        className="hidden"
        onChange={handleFileInput}
      />

      {/* Angle display + controls at bottom */}
      <div className="shrink-0 bg-surface-secondary border-t border-border px-3 py-2 space-y-2">
        {/* Angle label */}
        {proj.loaded && (
          <div className="text-xs text-content-secondary text-center font-mono">
            {proj.angleDeg.toFixed(1)}&deg;/{proj.secondaryAngleDeg.toFixed(1)}&deg; &middot; {proj.numFrames} frames &middot;{' '}
            {proj.imageWidth}x{proj.imageHeight} &middot; {proj.pixelSpacing.toFixed(3)} mm/px
          </div>
        )}

        {/* Action buttons row 1: Upload + Segment + Calibrate */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={isUploading}
            className="flex-1 py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors disabled:opacity-50"
          >
            {isUploading ? 'Uploading...' : proj.loaded ? 'Replace' : 'Upload'}
          </button>

          {proj.loaded && (
            <>
              {/* Segment / Re-segment */}
              <button
                onClick={() => segment(projectionId, frameIndex)}
                disabled={isSegmenting || proj.seedPoints.length < (segEngine === 'angiopy' ? 2 : 1)}
                title={
                  proj.seedPoints.length < (segEngine === 'angiopy' ? 2 : 1)
                    ? `Place at least ${segEngine === 'angiopy' ? 2 : 1} seed point${segEngine === 'angiopy' ? 's' : ''} (have ${proj.seedPoints.length})`
                    : `Segment with ${segEngine === 'seedmodel' ? 'SeedModel' : 'AngioPy'}`
                }
                className="flex-1 py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors disabled:opacity-50"
              >
                {isSegmenting
                  ? 'Segmenting...'
                  : proj.seedPoints.length < (segEngine === 'angiopy' ? 2 : 1)
                    ? `Segment (need ${(segEngine === 'angiopy' ? 2 : 1) - proj.seedPoints.length} more seed${proj.seedPoints.length === 0 ? 's' : ''})`
                    : proj.segmented
                      ? 'Re-segment'
                      : 'Segment'}
              </button>

              <button
                onClick={() => setShowCalibrate(!showCalibrate)}
                className="flex-1 py-1.5 text-xs rounded border border-border text-content-secondary hover:text-content-primary hover:border-content-muted transition-colors"
              >
                Calibrate
              </button>
            </>
          )}
        </div>

        {/* Engine selection row */}
        {proj.loaded && (
          <div className="flex items-center gap-1">
            <label className="text-[10px] text-content-muted mr-1">Engine:</label>
            {(['seedmodel', 'angiopy'] as const).map((eng) => (
              <button
                key={eng}
                onClick={() => setSegEngine(eng)}
                className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${
                  segEngine === eng
                    ? 'border-brand bg-brand/10 text-brand'
                    : 'border-border text-content-secondary hover:border-content-muted'
                }`}
              >
                {eng === 'seedmodel' ? 'SeedModel' : 'AngioPy'}
              </button>
            ))}
          </div>
        )}

        {/* Action buttons row 2: Seed tool + TIMI buttons */}
        {proj.loaded && (
          <div className="flex items-center gap-2">
            {/* Seed tool toggle */}
            <button
              onClick={() => setActiveTool(activeTool === 'seed' ? 'select' : 'seed')}
              className={`py-1.5 px-2 text-xs rounded border transition-colors ${
                activeTool === 'seed'
                  ? 'border-red-500 bg-red-500/10 text-red-400'
                  : 'border-border text-content-secondary hover:text-content-primary hover:border-content-muted'
              }`}
            >
              {activeTool === 'seed' ? 'Seed ON' : 'Seed'}
              {proj.seedPoints.length > 0 && (
                <span className="ml-1 text-[10px] opacity-80">({proj.seedPoints.length})</span>
              )}
            </button>
            {proj.seedPoints.length > 0 && (
              <button
                onClick={() => clearSeedPoints(projectionId)}
                className="py-1.5 px-2 text-xs rounded border border-border text-content-secondary hover:text-red-400 hover:border-red-400 transition-colors"
              >
                Clear Seeds
              </button>
            )}

            <div className="flex-1" />

            {/* TIMI T0 / T1 buttons */}
            <button
              onClick={() => setTimiFrame(projectionId, 'start', frameIndex)}
              className={`py-1.5 px-2 text-xs rounded border transition-colors ${
                proj.timiStart !== null
                  ? 'border-green-500 bg-green-500/10 text-green-400'
                  : 'border-border text-content-secondary hover:text-content-primary hover:border-content-muted'
              }`}
              title={proj.timiStart !== null ? `T₀ = frame ${proj.timiStart}` : 'Set current frame as T₀'}
            >
              Set T₀{proj.timiStart !== null && <span className="ml-1 text-[10px]">({proj.timiStart})</span>}
            </button>
            <button
              onClick={() => setTimiFrame(projectionId, 'end', frameIndex)}
              className={`py-1.5 px-2 text-xs rounded border transition-colors ${
                proj.timiEnd !== null
                  ? 'border-green-500 bg-green-500/10 text-green-400'
                  : 'border-border text-content-secondary hover:text-content-primary hover:border-content-muted'
              }`}
              title={proj.timiEnd !== null ? `T₁ = frame ${proj.timiEnd}` : 'Set current frame as T₁'}
            >
              Set T₁{proj.timiEnd !== null && <span className="ml-1 text-[10px]">({proj.timiEnd})</span>}
            </button>
          </div>
        )}

        {/* Calibration panel (toggle) */}
        {showCalibrate && proj.loaded && (
          <div className="space-y-2 pt-1 border-t border-border">
            {/* Catheter calibration — pixel value from QCA trimmed mean */}
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-content-muted">Fr:</label>
              <div className="flex gap-1">
                {FRENCH_SIZES.map((fr) => (
                  <button
                    key={fr}
                    onClick={() => setCatheterFr(fr)}
                    className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                      catheterFr === fr
                        ? 'border-brand bg-brand/10 text-brand'
                        : 'border-border text-content-secondary hover:border-content-muted'
                    }`}
                  >
                    {fr}F
                  </button>
                ))}
              </div>
              <span className="text-[10px] text-content-muted mx-1">|</span>
              <label className="text-[10px] text-content-muted">Px:</label>
              {trimmedMeanPx != null ? (
                <span className="text-[10px] text-content-secondary font-mono">
                  {trimmedMeanPx.toFixed(1)}
                </span>
              ) : (
                <span className="text-[10px] text-content-muted italic">segment first</span>
              )}
              <button
                onClick={() => {
                  if (trimmedMeanPx != null) {
                    calibrate(projectionId, catheterFr, trimmedMeanPx);
                    setShowCalibrate(false);
                  }
                }}
                disabled={trimmedMeanPx == null}
                className="px-2 py-0.5 text-xs rounded bg-brand text-white hover:bg-blue-600 transition-colors disabled:opacity-40"
              >
                Apply
              </button>
            </div>
          </div>
        )}

        {/* Per-projection frame scrubber */}
        {proj.loaded && proj.numFrames > 1 && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => setFrameIndex((prev) => Math.max(0, prev - 1))}
              className="text-xs text-content-secondary hover:text-content-primary"
            >
              &lsaquo;
            </button>
            <div className="flex-1 relative">
              <input
                type="range"
                min={0}
                max={proj.numFrames - 1}
                value={frameIndex}
                onChange={(e) => setFrameIndex(Number(e.target.value))}
                className="w-full h-1 accent-brand cursor-pointer"
              />
              {/* TIMI marker indicators on the scrubber */}
              {proj.timiStart !== null && (
                <div
                  className="absolute top-0 w-0.5 h-3 bg-green-400 pointer-events-none"
                  style={{ left: `${(proj.timiStart / Math.max(proj.numFrames - 1, 1)) * 100}%` }}
                  title={`T₀: frame ${proj.timiStart}`}
                />
              )}
              {proj.timiEnd !== null && (
                <div
                  className="absolute top-0 w-0.5 h-3 bg-red-400 pointer-events-none"
                  style={{ left: `${(proj.timiEnd / Math.max(proj.numFrames - 1, 1)) * 100}%` }}
                  title={`T₁: frame ${proj.timiEnd}`}
                />
              )}
            </div>
            <span className="text-[10px] text-content-secondary font-mono min-w-[40px] text-center">
              {frameIndex + 1}/{proj.numFrames}
            </span>
            <button
              onClick={() => setFrameIndex((prev) => Math.min(proj.numFrames - 1, prev + 1))}
              className="text-xs text-content-secondary hover:text-content-primary"
            >
              &rsaquo;
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Shared Controls Bar ---------- */

function SharedControls() {
  const projection1 = useQFRStore((s) => s.projection1);
  const projection2 = useQFRStore((s) => s.projection2);
  const mode = useQFRStore((s) => s.mode);
  const isReconstructing = useQFRStore((s) => s.isReconstructing);
  const qfrResult = useQFRStore((s) => s.qfrResult);
  const mesh3D = useQFRStore((s) => s.mesh3D);
  const error = useQFRStore((s) => s.error);
  const reconstruct = useQFRStore((s) => s.reconstruct);
  const setViewMode = useQFRStore((s) => s.setViewMode);
  const mldOverrideIndex = useQFRStore((s) => s.mldOverrideIndex);

  const bothLoaded = projection1.loaded && projection2.loaded;
  const bothSegmented = projection1.segmented && projection2.segmented;
  const bothCalibrated = projection1.pixelSpacing !== 0.3 || projection2.pixelSpacing !== 0.3;

  const angularSep = bothLoaded ? Math.abs(projection1.angleDeg - projection2.angleDeg) : null;
  const angularWarning = angularSep !== null && (angularSep < 25 || angularSep > 40);

  // TIMI check for cQFR
  const timiP1Set = projection1.timiStart !== null && projection1.timiEnd !== null;
  const timiP2Set = projection2.timiStart !== null && projection2.timiEnd !== null;
  const timiRequired = mode === 'cQFR' || mode === 'aQFR';
  const timiMissing = timiRequired && (!timiP1Set || !timiP2Set);

  // Validation issues
  const issues: string[] = [];
  if (!projection1.loaded) issues.push('Projection 1 not loaded');
  if (!projection2.loaded) issues.push('Projection 2 not loaded');
  if (projection1.loaded && !projection1.segmented) issues.push('Projection 1 not segmented');
  if (projection2.loaded && !projection2.segmented) issues.push('Projection 2 not segmented');
  if (!bothCalibrated) issues.push('Projections using default calibration (0.3 mm/px)');
  if (timiMissing) issues.push(`TIMI frames required for ${mode}`);
  if (angularSep !== null && angularSep < 25) issues.push(`Angular separation too low (${angularSep.toFixed(1)}°)`);

  const canReconstruct = bothSegmented && !isReconstructing;

  const qfrColor = (val: number) => {
    if (val >= 0.9) return 'text-green-400';
    if (val >= 0.8) return 'text-yellow-400';
    if (val >= 0.75) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-surface-secondary border-t border-border px-4 py-2 shrink-0">
      <div className="flex items-center gap-4">
        {/* Angular separation */}
        {bothLoaded && angularSep !== null && (
          <div className="text-xs text-content-secondary">
            Angular sep:{' '}
            <span className={angularWarning ? 'text-yellow-400 font-medium' : ''}>
              {angularSep.toFixed(1)}&deg;
            </span>
            {angularWarning && (
              <span className="text-yellow-400 ml-1">(recommended: 25-40&deg;)</span>
            )}
          </div>
        )}

        {/* Reconstruct button with tooltip */}
        <div className="relative group">
          <button
            onClick={reconstruct}
            disabled={!canReconstruct}
            className="px-4 py-1.5 text-sm font-medium rounded bg-brand text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isReconstructing ? 'Reconstructing...' : 'Reconstruct & Calculate QFR'}
          </button>
          {/* Validation tooltip */}
          {issues.length > 0 && !canReconstruct && (
            <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-20">
              <div className="bg-gray-900 text-white text-xs rounded px-3 py-2 shadow-lg max-w-[280px]">
                <div className="font-medium mb-1">Requirements:</div>
                <ul className="space-y-0.5">
                  {issues.map((issue, i) => (
                    <li key={i} className="text-yellow-300">- {issue}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {/* Warnings shown even when can reconstruct */}
          {canReconstruct && issues.length > 0 && (
            <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-20">
              <div className="bg-gray-900 text-white text-xs rounded px-3 py-2 shadow-lg max-w-[280px]">
                <div className="font-medium mb-1">Warnings:</div>
                <ul className="space-y-0.5">
                  {issues.map((issue, i) => (
                    <li key={i} className="text-yellow-300">- {issue}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* QFR result inline */}
        {qfrResult && (
          <div className="flex items-center gap-3">
            <div className="text-xs text-content-secondary">
              QFR ({qfrResult.mode}):
            </div>
            <span className={`text-lg font-bold ${qfrColor(qfrResult.qfr)}`}>
              {qfrResult.qfr.toFixed(3)}
            </span>
            {qfrResult.qfr < 0.8 && (
              <span className="text-xs text-red-400">Significant (&lt;0.80)</span>
            )}
            <span className="text-xs text-content-muted">
              Ref: {qfrResult.reference_diameter_mm.toFixed(2)}mm &middot;{' '}
              {mldOverrideIndex !== null ? (
                <span className="text-cyan-400">MLD (manual): {qfrResult.mld_mm.toFixed(2)}mm</span>
              ) : (
                <>MLD: {qfrResult.mld_mm.toFixed(2)}mm</>
              )}
            </span>
          </div>
        )}

        {/* View 3D button */}
        {mesh3D && (
          <button
            onClick={() => setViewMode('3d')}
            className="px-3 py-1.5 text-sm font-medium rounded border border-brand text-brand hover:bg-brand hover:text-white transition-colors"
          >
            View 3D
          </button>
        )}

        {/* Error */}
        {error && <span className="text-xs text-red-400 ml-auto">{error}</span>}
      </div>
    </div>
  );
}

/* ---------- QFR Dual Viewer (exported) ---------- */

export function QFRDualViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const viewMode = useQFRStore((s) => s.viewMode);
  const mesh3D = useQFRStore((s) => s.mesh3D);

  // Re-attach ResizeObserver whenever the container div is in the DOM
  // (it disappears during 3D mode and reappears when switching back)
  const isProjectionView = !(viewMode === '3d' && mesh3D);

  useEffect(() => {
    if (!isProjectionView) return;
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [isProjectionView]);

  // 3D view mode
  if (!isProjectionView) {
    return <QFRMesh3DViewer />;
  }

  const halfWidth = Math.floor(size.width / 2) - 4; // account for gap
  const canvasHeight = size.height;

  return (
    <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
      {/* Dual viewer area */}
      <div ref={containerRef} className="flex-1 min-h-0 flex gap-1 p-1 bg-black overflow-hidden">
        <ProjectionCanvas projectionId={1} width={halfWidth} height={canvasHeight} />
        <ProjectionCanvas projectionId={2} width={halfWidth} height={canvasHeight} />
      </div>

      {/* Shared controls bar */}
      <SharedControls />
    </div>
  );
}
