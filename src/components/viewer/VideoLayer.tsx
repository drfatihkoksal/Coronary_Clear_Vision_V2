import { useRef, useEffect, useCallback } from 'react';
import { useStudyStore } from '@/stores/studyStore';
import { usePlayerStore } from '@/stores/playerStore';

interface Props {
  containerWidth: number;
  containerHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function VideoLayer({ containerWidth, containerHeight, scale, offsetX, offsetY }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentBitmapRef = useRef<ImageBitmap | null>(null);

  const metadata = useStudyStore((s) => s.metadata);
  const getFrame = useStudyStore((s) => s.getFrame);
  const currentFrame = usePlayerStore((s) => s.currentFrame);

  const renderFrame = useCallback((bitmap: ImageBitmap) => {
    const canvas = canvasRef.current;
    if (!canvas || containerWidth === 0 || containerHeight === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (canvas.width !== containerWidth || canvas.height !== containerHeight) {
      canvas.width = containerWidth;
      canvas.height = containerHeight;
    }

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const fitScale = Math.min(containerWidth / bitmap.width, containerHeight / bitmap.height);
    const totalScale = fitScale * scale;
    const x = (containerWidth - bitmap.width * totalScale) / 2 + offsetX;
    const y = (containerHeight - bitmap.height * totalScale) / 2 + offsetY;

    ctx.imageSmoothingEnabled = scale < 2;
    ctx.drawImage(bitmap, x, y, bitmap.width * totalScale, bitmap.height * totalScale);
    currentBitmapRef.current = bitmap;
  }, [containerWidth, containerHeight, scale, offsetX, offsetY]);

  useEffect(() => {
    if (!metadata) return;
    let cancelled = false;
    getFrame(currentFrame).then((bitmap) => {
      if (!cancelled) renderFrame(bitmap);
    }).catch(console.error);
    return () => { cancelled = true; };
  }, [currentFrame, metadata, getFrame, renderFrame]);

  // Re-render on transform/size change
  useEffect(() => {
    if (currentBitmapRef.current) {
      renderFrame(currentBitmapRef.current);
    }
  }, [renderFrame]);

  return <canvas ref={canvasRef} className="absolute inset-0" />;
}
