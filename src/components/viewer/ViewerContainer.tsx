import { useState, useRef, useEffect, useCallback } from 'react';
import { VideoLayer } from './VideoLayer';
import { SegmentationLayer } from './SegmentationLayer';
import { AnnotationLayer } from './AnnotationLayer';
import { OverlayLayer } from './OverlayLayer';
import { ViewerContextMenu } from './ViewerContextMenu';
import { useStudyStore } from '@/stores/studyStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useToolStore } from '@/stores/toolStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useRWSStore } from '@/stores/rwsStore';

const ROI_SIZE = 160;

export function ViewerContainer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });

  const metadata = useStudyStore((s) => s.metadata);
  const activeTool = useToolStore((s) => s.activeTool);
  const currentFrame = usePlayerStore((s) => s.currentFrame);
  const addSeedPoint = useAnalysisStore((s) => s.addSeedPoint);
  const setRoi = useAnalysisStore((s) => s.setRoi);

  const rwsStartFrame = useRWSStore((s) => s.startFrame);
  const rwsEndFrame = useRWSStore((s) => s.endFrame);
  const rwsSetRange = useRWSStore((s) => s.setRange);

  const [contextMenu, setContextMenu] = useState<{ visible: boolean; x: number; y: number }>({
    visible: false,
    x: 0,
    y: 0,
  });

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setContextMenu({ visible: true, x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu((prev) => ({ ...prev, visible: false }));
  }, []);

  const handleSetStartFrame = useCallback((frame: number) => {
    rwsSetRange(frame, rwsEndFrame ?? frame);
  }, [rwsSetRange, rwsEndFrame]);

  const handleSetEndFrame = useCallback((frame: number) => {
    rwsSetRange(rwsStartFrame ?? frame, frame);
  }, [rwsSetRange, rwsStartFrame]);

  const imageWidth = metadata?.imageWidth ?? 512;
  const imageHeight = metadata?.imageHeight ?? 512;

  // Track container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Convert screen coords to image coords
  const screenToImage = useCallback((clientX: number, clientY: number): { ix: number; iy: number } | null => {
    const container = containerRef.current;
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    const fitScale = Math.min(size.width / imageWidth, size.height / imageHeight);
    const totalScale = fitScale * transform.scale;
    const imgX = (size.width - imageWidth * totalScale) / 2 + transform.offsetX;
    const imgY = (size.height - imageHeight * totalScale) / 2 + transform.offsetY;
    const ix = (clientX - rect.left - imgX) / totalScale;
    const iy = (clientY - rect.top - imgY) / totalScale;
    if (ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight) {
      return { ix, iy };
    }
    return null;
  }, [size.width, size.height, imageWidth, imageHeight, transform]);

  // Place ROI centred at image coords
  const placeRoi = useCallback((imageX: number, imageY: number) => {
    const half = ROI_SIZE / 2;
    const x = Math.max(0, Math.min(Math.round(imageX - half), imageWidth - ROI_SIZE));
    const y = Math.max(0, Math.min(Math.round(imageY - half), imageHeight - ROI_SIZE));
    const w = Math.min(ROI_SIZE, imageWidth - x);
    const h = Math.min(ROI_SIZE, imageHeight - y);
    setRoi(currentFrame, { x, y, width: w, height: h });
  }, [currentFrame, setRoi, imageWidth, imageHeight]);

  // ROI drag state
  const [isDraggingRoi, setIsDraggingRoi] = useState(false);

  // Handle canvas clicks based on active tool
  const handleCanvasClick = useCallback((imageX: number, imageY: number) => {
    if (activeTool === 'seed') {
      addSeedPoint(currentFrame, { x: Math.round(imageX), y: Math.round(imageY) });
    }
    // ROI placement is handled by mouseDown/mouseMove (drag) instead
  }, [activeTool, currentFrame, addSeedPoint]);

  // Mouse wheel zoom — use native listener with { passive: false } so preventDefault works
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setTransform((prev) => ({
        ...prev,
        scale: Math.max(0.1, Math.min(10, prev.scale * delta)),
      }));
    };

    container.addEventListener('wheel', onWheel, { passive: false });
    return () => container.removeEventListener('wheel', onWheel);
  }, []);

  // Pan
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && (activeTool === 'pan' || e.altKey))) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - transform.offsetX, y: e.clientY - transform.offsetY });
      e.preventDefault();
    } else if (e.button === 0 && activeTool === 'roi') {
      const pt = screenToImage(e.clientX, e.clientY);
      if (pt) {
        placeRoi(pt.ix, pt.iy);
        setIsDraggingRoi(true);
      }
      e.preventDefault();
    }
  }, [activeTool, transform.offsetX, transform.offsetY, screenToImage, placeRoi]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setTransform((prev) => ({
        ...prev,
        offsetX: e.clientX - panStart.x,
        offsetY: e.clientY - panStart.y,
      }));
    } else if (isDraggingRoi && activeTool === 'roi') {
      const pt = screenToImage(e.clientX, e.clientY);
      if (pt) {
        placeRoi(pt.ix, pt.iy);
      }
    }
  }, [isPanning, panStart, isDraggingRoi, activeTool, screenToImage, placeRoi]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    setIsDraggingRoi(false);
  }, []);

  // Keyboard shortcuts for tools
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const { setActiveTool } = useToolStore.getState();
      switch (e.key) {
        case 'b': case 'B': setActiveTool('roi'); break;
        case 's': case 'S': setActiveTool('seed'); break;
        case 'h': case 'H': setActiveTool('pan'); break;
        case 'Escape': setActiveTool('select'); break;
        case 'r': case 'R': setTransform({ scale: 1, offsetX: 0, offsetY: 0 }); break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const cursorClass = activeTool === 'seed' ? 'cursor-crosshair' : activeTool === 'roi' ? 'cursor-crosshair' : activeTool === 'pan' ? 'cursor-grab' : 'cursor-default';

  return (
    <div
      ref={containerRef}
      className={`flex-1 bg-black relative overflow-hidden ${cursorClass}`}
      role="application"
      aria-label="Angiogram viewer"
      onMouseDown={(e) => { closeContextMenu(); handleMouseDown(e); }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onContextMenu={handleContextMenu}
    >
      {/* Video layer */}
      <VideoLayer
        containerWidth={size.width}
        containerHeight={size.height}
        scale={transform.scale}
        offsetX={transform.offsetX}
        offsetY={transform.offsetY}
      />

      {/* Segmentation overlay */}
      {metadata && (
        <SegmentationLayer
          containerWidth={size.width}
          containerHeight={size.height}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          scale={transform.scale}
          offsetX={transform.offsetX}
          offsetY={transform.offsetY}
        />
      )}

      {/* Annotation layer (seeds, ROI) */}
      {metadata && (
        <AnnotationLayer
          containerWidth={size.width}
          containerHeight={size.height}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          scale={transform.scale}
          offsetX={transform.offsetX}
          offsetY={transform.offsetY}
          onCanvasClick={handleCanvasClick}
        />
      )}

      {/* ECG overlay (optional, toggled via toolStore.overlays.ecgOverlay) */}
      {metadata && (
        <OverlayLayer
          containerWidth={size.width}
          containerHeight={size.height}
        />
      )}

      {/* Empty state */}
      {!metadata && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-content-muted text-sm">Load a DICOM file to begin</p>
        </div>
      )}

      {/* Zoom indicator */}
      {metadata && transform.scale !== 1 && (
        <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/60 text-xs text-white pointer-events-none">
          {Math.round(transform.scale * 100)}%
        </div>
      )}

      {/* Frame counter */}
      {metadata && (
        <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/60 text-xs text-white pointer-events-none">
          {currentFrame + 1}/{metadata.numFrames}
        </div>
      )}

      {/* Context menu */}
      {contextMenu.visible && (
        <ViewerContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          currentFrame={currentFrame}
          onClose={closeContextMenu}
          onSetStartFrame={handleSetStartFrame}
          onSetEndFrame={handleSetEndFrame}
        />
      )}
    </div>
  );
}
