import { useRef, useEffect, useCallback } from 'react';
import { useTimingStore } from '@/stores/timingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useToolStore } from '@/stores/toolStore';

const OVERLAY_HEIGHT = 60; // pixels from the bottom of the canvas
const SIGNAL_COLOR = 'rgba(34,197,94,0.6)';
const RPEAK_COLOR = 'rgba(239,68,68,0.5)';
const CURSOR_COLOR = 'rgba(250,204,21,0.8)';
const BG_COLOR = 'rgba(15,23,42,0.4)';

interface Props {
  containerWidth: number;
  containerHeight: number;
}

export function OverlayLayer({ containerWidth, containerHeight }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const ecgSignal = useTimingStore((s) => s.ecgSignal);
  const rPeaks = useTimingStore((s) => s.rPeaks);
  const ecgOverlay = useToolStore((s) => s.overlays.ecgOverlay);

  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const totalFrames = usePlayerStore((s) => s.totalFrames);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = containerWidth * dpr;
    const h = containerHeight * dpr;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    // Only draw if overlay is enabled and ECG data is available
    if (!ecgOverlay || !ecgSignal || ecgSignal.length === 0) return;

    const overlayH = OVERLAY_HEIGHT * dpr;
    const overlayY = h - overlayH;

    // Semi-transparent background strip
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, overlayY, w, overlayH);

    const sigLen = ecgSignal.length;
    const numFrames = totalFrames > 0 ? totalFrames : sigLen;

    // Compute signal bounds
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < sigLen; i++) {
      if (ecgSignal[i] < minVal) minVal = ecgSignal[i];
      if (ecgSignal[i] > maxVal) maxVal = ecgSignal[i];
    }
    const range = maxVal - minVal || 1;
    const padding = 0.1;
    const scaledMin = minVal - range * padding;
    const scaledMax = maxVal + range * padding;
    const scaledRange = scaledMax - scaledMin;

    const sigToX = (si: number) => (si / (sigLen - 1)) * w;
    const valToY = (v: number) => overlayY + overlayH - ((v - scaledMin) / scaledRange) * overlayH;
    const frameToX = (fi: number) => {
      const si = (fi / Math.max(numFrames - 1, 1)) * (sigLen - 1);
      return sigToX(si);
    };

    // Draw ECG signal
    ctx.strokeStyle = SIGNAL_COLOR;
    ctx.lineWidth = 1.2 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();

    const pixelStep = Math.max(1, Math.floor(sigLen / (containerWidth * 2)));
    for (let i = 0; i < sigLen; i += pixelStep) {
      const x = sigToX(i);
      const y = valToY(ecgSignal[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    if ((sigLen - 1) % pixelStep !== 0) {
      ctx.lineTo(sigToX(sigLen - 1), valToY(ecgSignal[sigLen - 1]));
    }
    ctx.stroke();

    // Draw R-peak markers
    if (rPeaks.length > 0) {
      ctx.strokeStyle = RPEAK_COLOR;
      ctx.lineWidth = 1 * dpr;
      for (const peak of rPeaks) {
        const x = frameToX(peak);
        ctx.beginPath();
        ctx.moveTo(x, overlayY);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
    }

    // Draw current frame cursor
    if (totalFrames > 0) {
      const cx = frameToX(currentFrame);
      ctx.strokeStyle = CURSOR_COLOR;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(cx, overlayY);
      ctx.lineTo(cx, h);
      ctx.stroke();
    }
  }, [containerWidth, containerHeight, ecgSignal, rPeaks, ecgOverlay, currentFrame, totalFrames]);

  useEffect(() => {
    draw();
  }, [draw]);

  if (!ecgOverlay) return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ width: containerWidth, height: containerHeight, zIndex: 40 }}
    />
  );
}
