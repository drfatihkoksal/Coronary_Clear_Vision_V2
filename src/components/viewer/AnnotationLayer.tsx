import { useRef, useEffect, useCallback } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useToolStore } from '@/stores/toolStore';

interface Props {
  containerWidth: number;
  containerHeight: number;
  imageWidth: number;
  imageHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
  onCanvasClick?: (imageX: number, imageY: number) => void;
}

export function AnnotationLayer({ containerWidth, containerHeight, imageWidth, imageHeight, scale, offsetX, offsetY, onCanvasClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const seedPoints = useAnalysisStore((s) => s.seedPoints);
  const rois = useAnalysisStore((s) => s.rois);
  const overlays = useToolStore((s) => s.overlays);

  const frameData = useAnalysisStore((s) => s.frameData);
  const qcaResults = useAnalysisStore((s) => s.qcaResults);
  const seeds = seedPoints.get(currentFrame) || [];
  const roi = rois.get(currentFrame);
  const segData = frameData.get(currentFrame);
  const qca = qcaResults.get(currentFrame);

  const fitScale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
  const totalScale = fitScale * scale;
  const imgX = (containerWidth - imageWidth * totalScale) / 2 + offsetX;
  const imgY = (containerHeight - imageHeight * totalScale) / 2 + offsetY;

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = containerWidth;
    canvas.height = containerHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw seed points
    if (overlays.seedPoints && seeds.length > 0) {
      ctx.fillStyle = '#ef4444';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      for (const p of seeds) {
        const cx = imgX + p.x * totalScale;
        const cy = imgY + p.y * totalScale;
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Draw ROI
    if (overlays.roi && roi) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        imgX + roi.x * totalScale,
        imgY + roi.y * totalScale,
        roi.width * totalScale,
        roi.height * totalScale,
      );
      ctx.setLineDash([]);
    }

    // Draw centerline
    if (overlays.centerline && segData && segData.centerline.length > 1) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const p0 = segData.centerline[0];
      ctx.moveTo(imgX + p0.x * totalScale, imgY + p0.y * totalScale);
      for (let i = 1; i < segData.centerline.length; i++) {
        const p = segData.centerline[i];
        ctx.lineTo(imgX + p.x * totalScale, imgY + p.y * totalScale);
      }
      ctx.stroke();
    }

    // Draw diameter markers — use QCA centerline when available (correct indices + auto spacing)
    if (overlays.diameterMarkers) {
      // Pick the best data source: QCA has calibrated resolution, segData is fallback
      const hasQCA = qca && qca.centerline.length > 1 && qca.diameterProfilePx.length > 0;
      const hasSeg = segData && segData.centerline.length > 1 && segData.diameters_px.length > 0;

      if (hasQCA) {
        const cl = qca.centerline;
        const diams = qca.diameterProfilePx;
        const specialIndices = new Set([qca.proximalRefIndex, qca.mldIndex, qca.distalRefIndex]);

        // Helper: draw a perpendicular diameter line at centerline index i
        const drawCrossSec = (i: number, color: string, lineWidth: number) => {
          const d = diams[i];
          if (d <= 0) return;
          const prev = cl[Math.max(0, i - 1)];
          const next = cl[Math.min(cl.length - 1, i + 1)];
          const tx = next.x - prev.x;
          const ty = next.y - prev.y;
          const tLen = Math.sqrt(tx * tx + ty * ty);
          if (tLen === 0) return;
          const nx = -ty / tLen;
          const ny = tx / tLen;
          const halfD = d / 2;
          const cx = imgX + cl[i].x * totalScale;
          const cy = imgY + cl[i].y * totalScale;
          const dx = nx * halfD * totalScale;
          const dy = ny * halfD * totalScale;
          ctx.strokeStyle = color;
          ctx.lineWidth = lineWidth;
          ctx.beginPath();
          ctx.moveTo(cx - dx, cy - dy);
          ctx.lineTo(cx + dx, cy + dy);
          ctx.stroke();
        };

        // Draw all regular markers (skip special ones, they get drawn on top)
        for (let i = 0; i < cl.length && i < diams.length; i++) {
          if (specialIndices.has(i)) continue;
          drawCrossSec(i, '#f59e0b', 1);
        }

        // Draw special QCA markers on top (proximal, MLD, distal)
        const specials: { idx: number | undefined; label: string; color: string; valueMm: number | undefined }[] = [
          { idx: qca.proximalRefIndex, label: 'Prox', color: '#3b82f6', valueMm: qca.proximalRefMm },
          { idx: qca.mldIndex, label: 'MLD', color: '#ef4444', valueMm: qca.mldMm },
          { idx: qca.distalRefIndex, label: 'Dist', color: '#8b5cf6', valueMm: qca.distalRefMm },
        ];
        for (const m of specials) {
          if (m.idx == null || m.idx < 0 || m.idx >= cl.length) continue;
          drawCrossSec(m.idx, m.color, 2);
          // Label
          const d = diams[m.idx] ?? 0;
          if (d <= 0) continue;
          const prev = cl[Math.max(0, m.idx - 1)];
          const next = cl[Math.min(cl.length - 1, m.idx + 1)];
          const tx = next.x - prev.x;
          const ty = next.y - prev.y;
          const tLen = Math.sqrt(tx * tx + ty * ty);
          if (tLen === 0) continue;
          const nx = -ty / tLen;
          const ny = tx / tLen;
          const halfD = d / 2;
          const ex = imgX + cl[m.idx].x * totalScale + nx * halfD * totalScale;
          const ey = imgY + cl[m.idx].y * totalScale + ny * halfD * totalScale;
          const labelText = m.valueMm != null ? `${m.label} ${m.valueMm.toFixed(2)}mm` : m.label;
          ctx.font = '11px sans-serif';
          ctx.fillStyle = m.color;
          ctx.textAlign = 'left';
          ctx.fillText(labelText, ex + 4, ey - 2);
        }
      } else if (hasSeg) {
        // Fallback: segmentation data only, sample ~20 markers
        const cl = segData.centerline;
        const diams = segData.diameters_px;
        const step = Math.max(1, Math.floor(cl.length / 20));
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1;
        for (let i = 0; i < cl.length && i < diams.length; i += step) {
          const d = diams[i];
          if (d <= 0) continue;
          const prev = cl[Math.max(0, i - 1)];
          const next = cl[Math.min(cl.length - 1, i + 1)];
          const tx = next.x - prev.x;
          const ty = next.y - prev.y;
          const tLen = Math.sqrt(tx * tx + ty * ty);
          if (tLen === 0) continue;
          const nx = -ty / tLen;
          const ny = tx / tLen;
          const halfD = d / 2;
          const cx = imgX + cl[i].x * totalScale;
          const cy = imgY + cl[i].y * totalScale;
          const dx = nx * halfD * totalScale;
          const dy = ny * halfD * totalScale;
          ctx.beginPath();
          ctx.moveTo(cx - dx, cy - dy);
          ctx.lineTo(cx + dx, cy + dy);
          ctx.stroke();
        }
      }
    }
  }, [containerWidth, containerHeight, seeds, roi, segData, qca, overlays, imgX, imgY, totalScale]);

  useEffect(() => {
    render();
  }, [render]);

  const handleClick = (e: React.MouseEvent) => {
    if (!onCanvasClick) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Convert to image coordinates
    const ix = (canvasX - imgX) / totalScale;
    const iy = (canvasY - imgY) / totalScale;

    if (ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight) {
      onCanvasClick(ix, iy);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0"
      style={{ width: containerWidth, height: containerHeight }}
      onClick={handleClick}
    />
  );
}
