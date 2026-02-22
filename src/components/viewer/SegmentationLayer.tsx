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
}

export function SegmentationLayer({ containerWidth, containerHeight, imageWidth, imageHeight, scale, offsetX, offsetY }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const frameData = useAnalysisStore((s) => s.frameData);
  const overlays = useToolStore((s) => s.overlays);

  const segData = frameData.get(currentFrame);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = containerWidth;
    canvas.height = containerHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!segData) return;

    const fitScale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
    const totalScale = fitScale * scale;
    const x = (containerWidth - imageWidth * totalScale) / 2 + offsetX;
    const y = (containerHeight - imageHeight * totalScale) / 2 + offsetY;

    // Draw mask overlay
    if (overlays.mask && segData.maskBitmap) {
      ctx.globalAlpha = 0.35;
      ctx.drawImage(segData.maskBitmap, x, y, imageWidth * totalScale, imageHeight * totalScale);
      ctx.globalAlpha = 1.0;
    }

    // Draw centerline
    if (overlays.centerline && segData.centerline.length >= 2) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const cl = segData.centerline;
      ctx.moveTo(x + cl[0].x * totalScale, y + cl[0].y * totalScale);
      for (let i = 1; i < cl.length; i++) {
        ctx.lineTo(x + cl[i].x * totalScale, y + cl[i].y * totalScale);
      }
      ctx.stroke();
    }
  }, [containerWidth, containerHeight, imageWidth, imageHeight, scale, offsetX, offsetY, segData, overlays]);

  useEffect(() => {
    render();
  }, [render]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ width: containerWidth, height: containerHeight }}
    />
  );
}
