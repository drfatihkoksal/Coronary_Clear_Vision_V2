import { useRef, useEffect, useCallback, useState } from 'react';
import { useTimingStore } from '@/stores/timingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useStudyStore } from '@/stores/studyStore';

const CHART_HEIGHT = 100;
const BG_COLOR = '#0f172a';
const GRID_COLOR = 'rgba(100,116,139,0.15)';
const ECG_COLOR = '#22c55e';
const MOTION_COLOR = '#06b6d4';
const RPEAK_COLOR = '#ef4444';
const MOTION_PEAK_COLOR = '#f97316';
const CURSOR_COLOR = '#facc15';
const BEAT_BOUNDARY_COLOR = 'rgba(59,130,246,0.2)';
const DRAG_PEAK_COLOR = '#f97316';

export function ECGChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const widthRef = useRef(0);

  // ECG state
  const ecgSignal = useTimingStore((s) => s.ecgSignal);
  const rPeaks = useTimingStore((s) => s.rPeaks);
  const beatBoundaries = useTimingStore((s) => s.beatBoundaries);
  const heartRate = useTimingStore((s) => s.heartRate);
  const isLoadingEcg = useTimingStore((s) => s.isLoadingEcg);
  const ecgError = useTimingStore((s) => s.ecgError);
  const isEditMode = useTimingStore((s) => s.isEcgEditMode);
  const ecgVisible = useTimingStore((s) => s.ecgVisible);
  const motionVisible = useTimingStore((s) => s.motionVisible);

  // Motion state
  const motionSignal = useTimingStore((s) => s.motionSignal);
  const motionPeaks = useTimingStore((s) => s.motionPeaks);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);
  const metadata = useStudyStore((s) => s.metadata);

  // Drag state
  const [dragPeakIndex, setDragPeakIndex] = useState<number | null>(null);
  const [dragX, setDragX] = useState<number | null>(null);
  const isDragging = dragPeakIndex !== null;

  const hasEcg = ecgSignal !== null && ecgSignal.length > 0;
  const hasMotion = motionSignal !== null && motionSignal.length > 0;

  const xToSampleIndex = useCallback(
    (clientX: number): number | null => {
      const canvas = canvasRef.current;
      if (!canvas || !ecgSignal) return null;
      const rect = canvas.getBoundingClientRect();
      const fraction = (clientX - rect.left) / rect.width;
      return Math.round(fraction * (ecgSignal.length - 1));
    },
    [ecgSignal],
  );

  const findNearestPeak = useCallback(
    (sampleIndex: number, tolerancePx: number = 10): number | null => {
      const canvas = canvasRef.current;
      if (!canvas || !ecgSignal || rPeaks.length === 0) return null;
      const toleranceSamples = (tolerancePx / canvas.getBoundingClientRect().width) * ecgSignal.length;
      let closestIdx = -1;
      let closestDist = Infinity;
      for (let i = 0; i < rPeaks.length; i++) {
        const dist = Math.abs(rPeaks[i] - sampleIndex);
        if (dist < closestDist) { closestDist = dist; closestIdx = i; }
      }
      return closestDist <= toleranceSamples ? closestIdx : null;
    },
    [ecgSignal, rPeaks],
  );

  // --- Drawing ---
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    const logicalW = w / dpr;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = dpr;
    for (let i = 1; i < 3; i++) {
      const gy = (i / 3) * h;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    if (!hasEcg && !hasMotion) return;

    const sigLen = ecgSignal ? ecgSignal.length : (motionSignal ? motionSignal.length : 0);
    const numFrames = totalFrames > 0 ? totalFrames : sigLen;

    // Mapping helpers
    const sigToX = (si: number) => (si / Math.max(sigLen - 1, 1)) * w;
    const frameToX = (fi: number) => sigToX((fi / Math.max(numFrames - 1, 1)) * (sigLen - 1));

    // Beat boundaries
    if (ecgVisible && beatBoundaries.length > 0) {
      ctx.fillStyle = BEAT_BOUNDARY_COLOR;
      for (let i = 0; i < beatBoundaries.length; i++) {
        if (i % 2 === 0) {
          const x1 = frameToX(beatBoundaries[i].startFrame);
          const x2 = frameToX(beatBoundaries[i].endFrame);
          ctx.fillRect(x1, 0, x2 - x1, h);
        }
      }
    }

    // Draw signal helper — each signal spans the full canvas width
    const drawSignal = (sig: number[], color: string, lineW: number) => {
      let minV = Infinity, maxV = -Infinity;
      for (const v of sig) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
      const range = maxV - minV || 1;
      const pad = 0.1;
      const sMin = minV - range * pad;
      const sRange = (maxV + range * pad) - sMin;
      const valToY = (v: number) => h - ((v - sMin) / sRange) * h;
      const indexToX = (i: number) => (i / Math.max(sig.length - 1, 1)) * w;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineW * dpr;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const step = Math.max(1, Math.floor(sig.length / (logicalW * 2)));
      for (let i = 0; i < sig.length; i += step) {
        const x = indexToX(i);
        const y = valToY(sig[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      if ((sig.length - 1) % step !== 0) {
        ctx.lineTo(indexToX(sig.length - 1), valToY(sig[sig.length - 1]));
      }
      ctx.stroke();

      return valToY;
    };

    // ECG signal
    if (hasEcg && ecgVisible) {
      drawSignal(ecgSignal!, ECG_COLOR, 1.5);
    }

    // Motion signal overlay
    if (hasMotion && motionVisible) {
      const motionValToY = drawSignal(motionSignal!, MOTION_COLOR, 1.2);

      // Motion peaks — map using motion signal length (frame-based)
      if (motionPeaks.length > 0) {
        const motionLen = motionSignal!.length;
        ctx.fillStyle = MOTION_PEAK_COLOR;
        for (const peak of motionPeaks) {
          const x = (peak / Math.max(motionLen - 1, 1)) * w;
          ctx.beginPath();
          ctx.moveTo(x, h - 4 * dpr);
          ctx.lineTo(x - 3 * dpr, h);
          ctx.lineTo(x + 3 * dpr, h);
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // R-peak markers
    if (hasEcg && ecgVisible && rPeaks.length > 0) {
      for (let i = 0; i < rPeaks.length; i++) {
        if (isDragging && dragPeakIndex === i) continue;
        const x = sigToX(rPeaks[i]);
        ctx.strokeStyle = RPEAK_COLOR;
        ctx.lineWidth = 1 * dpr;
        ctx.setLineDash([2 * dpr, 2 * dpr]);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();

        // Triangle marker in edit mode
        if (isEditMode) {
          ctx.setLineDash([]);
          ctx.fillStyle = RPEAK_COLOR;
          ctx.beginPath();
          ctx.moveTo(x - 3 * dpr, 0);
          ctx.lineTo(x + 3 * dpr, 0);
          ctx.lineTo(x, 6 * dpr);
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.setLineDash([]);
    }

    // Dragged peak
    if (isDragging && dragX !== null) {
      const rect = canvasRef.current!.getBoundingClientRect();
      const cx = (dragX / rect.width) * w;
      ctx.strokeStyle = DRAG_PEAK_COLOR;
      ctx.lineWidth = 2 * dpr;
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Frame cursor
    if (totalFrames > 0) {
      const cx = frameToX(currentFrame);
      ctx.strokeStyle = CURSOR_COLOR;
      ctx.lineWidth = 1.5 * dpr;
      ctx.shadowColor = CURSOR_COLOR;
      ctx.shadowBlur = 3 * dpr;
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Info text top-left
    ctx.fillStyle = 'rgba(241,245,249,0.6)';
    ctx.font = `${9 * dpr}px -apple-system, BlinkMacSystemFont, sans-serif`;
    if (heartRate != null) ctx.fillText(`HR: ${heartRate} bpm`, 4 * dpr, 11 * dpr);

    // Edit mode hint bottom-right
    if (isEditMode) {
      ctx.fillStyle = 'rgba(250,204,21,0.7)';
      ctx.font = `${8 * dpr}px -apple-system, BlinkMacSystemFont, sans-serif`;
      const hint = 'Click: add | Right-click: remove | Drag: move';
      const tw = ctx.measureText(hint).width;
      ctx.fillText(hint, w - tw - 4 * dpr, h - 4 * dpr);
    }
  }, [ecgSignal, rPeaks, beatBoundaries, heartRate, currentFrame, totalFrames,
    ecgVisible, motionVisible, motionSignal, motionPeaks, isEditMode,
    isDragging, dragPeakIndex, dragX, hasEcg, hasMotion]);

  // Resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      if (Math.abs(width - widthRef.current) < 1) return;
      widthRef.current = width;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = CHART_HEIGHT * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${CHART_HEIGHT}px`;
      draw();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  // --- Interactions ---
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) return;
    const canvas = canvasRef.current;
    if (!canvas || !ecgSignal || totalFrames === 0) return;
    if (isEditMode) {
      const si = xToSampleIndex(e.clientX);
      if (si !== null) useTimingStore.getState().addPeak(si);
    } else {
      const rect = canvas.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      usePlayerStore.getState().goToFrame(Math.round(fraction * (totalFrames - 1)));
    }
  }, [ecgSignal, totalFrames, isEditMode, xToSampleIndex, isDragging]);

  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isEditMode) return;
    e.preventDefault();
    const si = xToSampleIndex(e.clientX);
    if (si !== null) useTimingStore.getState().removePeak(si);
  }, [isEditMode, xToSampleIndex]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isEditMode || e.button !== 0) return;
    const si = xToSampleIndex(e.clientX);
    if (si === null) return;
    const idx = findNearestPeak(si);
    if (idx !== null) {
      e.preventDefault();
      setDragPeakIndex(idx);
      const rect = canvasRef.current!.getBoundingClientRect();
      setDragX(e.clientX - rect.left);
    }
  }, [isEditMode, xToSampleIndex, findNearestPeak]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) setDragX(e.clientX - rect.left);
  }, [isDragging]);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || dragPeakIndex === null) return;
    const from = rPeaks[dragPeakIndex];
    const to = xToSampleIndex(e.clientX);
    if (to !== null && from !== to) useTimingStore.getState().movePeak(from, to);
    setDragPeakIndex(null);
    setDragX(null);
  }, [isDragging, dragPeakIndex, rPeaks, xToSampleIndex]);

  const handleMouseLeave = useCallback(() => {
    if (isDragging) { setDragPeakIndex(null); setDragX(null); }
  }, [isDragging]);

  if (!metadata) return null;

  // No signals at all
  if (!hasEcg && !hasMotion && !isLoadingEcg && !ecgError) return null;

  return (
    <div className="relative" style={{ height: CHART_HEIGHT }}>
      {/* Toggle buttons - top-left overlay */}
      <div className="absolute top-1 left-1 flex gap-1 z-10">
        {hasEcg && (
          <button
            onClick={() => useTimingStore.getState().toggleEcgVisible()}
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${ecgVisible ? 'bg-green-600 text-white' : 'bg-gray-700/80 text-gray-500 hover:bg-gray-600'
              }`}
            title={ecgVisible ? 'Hide ECG' : 'Show ECG'}
          >
            <span className="text-green-300">&#9679;</span> ECG
          </button>
        )}
        {hasMotion && (
          <button
            onClick={() => useTimingStore.getState().toggleMotionVisible()}
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${motionVisible ? 'bg-cyan-600 text-white' : 'bg-gray-700/80 text-gray-500 hover:bg-gray-600'
              }`}
            title={motionVisible ? 'Hide Motion' : 'Show Motion'}
          >
            <span className="text-cyan-300">&#9650;</span> Motion
          </button>
        )}
        {hasEcg && (
          <button
            onClick={() => useTimingStore.getState().toggleEcgEditMode()}
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${isEditMode ? 'bg-yellow-600 text-white' : 'bg-gray-700/80 text-gray-300 hover:bg-gray-600'
              }`}
            title={isEditMode ? 'Exit R-peak edit' : 'Edit R-peaks'}
          >
            {isEditMode ? '\u2713 R-peaks' : '\u270F R-peaks'}
          </button>
        )}
      </div>

      {/* Info badge - top-right */}
      {hasEcg && (
        <div className="absolute top-1 right-1 flex gap-2 z-10 text-[9px] text-gray-400">
          {heartRate != null && <span>HR: {heartRate} bpm</span>}
          {rPeaks.length > 0 && <span>R: {rPeaks.length}</span>}
          {hasMotion && motionPeaks.length > 0 && <span>M: {motionPeaks.length}</span>}
        </div>
      )}

      {/* Loading overlay */}
      {isLoadingEcg && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/60 z-20 text-xs text-gray-400">
          Loading ECG...
        </div>
      )}

      {/* Error overlay */}
      {ecgError && !isLoadingEcg && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/60 z-20 text-xs text-red-400">
          {ecgError}
        </div>
      )}

      {/* Canvas */}
      <div ref={containerRef} className="w-full h-full">
        <canvas
          ref={canvasRef}
          className={`block w-full h-full ${isEditMode ? 'cursor-crosshair' : 'cursor-pointer'}`}
          onClick={handleClick}
          onContextMenu={handleContextMenu}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />
      </div>
    </div>
  );
}
