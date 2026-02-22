import { useEffect, useRef, useState, useCallback } from 'react';
import { usePlayerStore } from '@/stores/playerStore';
import { useStudyStore } from '@/stores/studyStore';
import { useTrackingStore } from '@/stores/trackingStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useRWSStore } from '@/stores/rwsStore';

const SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0];

export function PlaybackControls() {
  const metadata = useStudyStore((s) => s.metadata);
  const {
    currentFrame, totalFrames, playbackState, playbackSpeed, isLooping,
    rangeStart, rangeEnd,
    togglePlayPause, stepForward, stepBackward,
    goToFrame, setPlaybackSpeed, toggleLoop,
    setPlaybackRange, clearPlaybackRange,
  } = usePlayerStore();

  const animationRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const getFrame = useStudyStore((s) => s.getFrame);
  const frameRate = useStudyStore((s) => s.metadata?.frameRate ?? 15);

  // Tracking state
  const isTrackMode = useTrackingStore((s) => s.isTrackMode);
  const enableTrackMode = useTrackingStore((s) => s.enableTrackMode);
  const disableTrackMode = useTrackingStore((s) => s.disableTrackMode);
  const autoSegQcaEnabled = useTrackingStore((s) => s.autoSegQcaEnabled);
  const setAutoSegQcaEnabled = useTrackingStore((s) => s.setAutoSegQcaEnabled);
  const isTracking = useTrackingStore((s) => s.isTracking);
  const confidence = useTrackingStore((s) => s.confidence);
  const confidenceThreshold = useTrackingStore((s) => s.confidenceThreshold);

  // ROI for current frame
  const rois = useAnalysisStore((s) => s.rois);
  const currentRoi = rois.get(currentFrame) ?? null;

  // RWS frame range → playback range sync
  const rwsStartFrame = useRWSStore((s) => s.startFrame);
  const rwsEndFrame = useRWSStore((s) => s.endFrame);

  useEffect(() => {
    if (rwsStartFrame != null && rwsEndFrame != null && rwsStartFrame < rwsEndFrame) {
      setPlaybackRange(rwsStartFrame, rwsEndFrame);
    } else {
      clearPlaybackRange();
    }
  }, [rwsStartFrame, rwsEndFrame, setPlaybackRange, clearPlaybackRange]);

  // Effective range for UI
  const effectiveStart = rangeStart ?? 0;
  const effectiveEnd = rangeEnd ?? totalFrames - 1;
  const hasRange = rangeStart != null && rangeEnd != null;

  // Tracking playback state
  const [isPlayingWithTrack, setIsPlayingWithTrack] = useState(false);
  const isPlayingWithTrackRef = useRef(false);
  const playbackIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Set total frames when study loads
  useEffect(() => {
    if (metadata) {
      usePlayerStore.getState().setTotalFrames(metadata.numFrames);
      usePlayerStore.getState().setFrameRate(metadata.frameRate);
    }
  }, [metadata]);

  // Normal playback animation loop (only when NOT in track-mode play)
  useEffect(() => {
    if (playbackState !== 'playing') return;

    const interval = 1000 / (frameRate * playbackSpeed);

    const tick = (timestamp: number) => {
      if (timestamp - lastTimeRef.current >= interval) {
        lastTimeRef.current = timestamp;
        usePlayerStore.getState().stepForward();
      }
      animationRef.current = requestAnimationFrame(tick);
    };

    lastTimeRef.current = performance.now();
    animationRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(animationRef.current);
  }, [playbackState, playbackSpeed, frameRate]);

  // Prefetch next frames
  useEffect(() => {
    if (!metadata) return;
    const prefetchCount = 3;
    for (let i = 1; i <= prefetchCount; i++) {
      const nextFrame = (currentFrame + i) % metadata.numFrames;
      getFrame(nextFrame).catch(() => {});
    }
  }, [currentFrame, metadata, getFrame]);

  // Cleanup tracking playback on unmount
  useEffect(() => {
    return () => {
      if (playbackIntervalRef.current) {
        clearTimeout(playbackIntervalRef.current);
      }
    };
  }, []);

  // --- Track Mode: play with tracking ---
  const playWithTracking = useCallback(async () => {
    const analysisStore = useAnalysisStore.getState();
    const frame = usePlayerStore.getState().currentFrame;
    const roi = analysisStore.rois.get(frame);

    if (!roi) return;

    // Initialize tracker on current frame
    const trackingStore = useTrackingStore.getState();
    await trackingStore.initializeTracking(frame, [
      Math.round(roi.x), Math.round(roi.y),
      Math.round(roi.width), Math.round(roi.height),
    ]);

    if (!useTrackingStore.getState().isInitialized) return;

    // Fire-and-forget seg+QCA on reference frame (don't block playback start)
    if (useTrackingStore.getState().autoSegQcaEnabled) {
      analysisStore.segmentAndExtract(frame, roi).catch(() => {});
    }

    setIsPlayingWithTrack(true);
    isPlayingWithTrackRef.current = true;

    const trackNextFrame = async () => {
      if (!isPlayingWithTrackRef.current) {
        setIsPlayingWithTrack(false);
        return;
      }

      const ps = usePlayerStore.getState();
      const { currentFrame: f, isLooping: loop, rangeStart: rStart, rangeEnd: rEnd } = ps;
      const lo = rStart ?? 0;
      const hi = rEnd ?? ps.totalFrames - 1;
      const nextIdx = f + 1;

      if (nextIdx > hi) {
        if (loop) {
          usePlayerStore.getState().goToFrame(lo);
        } else {
          isPlayingWithTrackRef.current = false;
          setIsPlayingWithTrack(false);
          return;
        }
      } else {
        const success = await useTrackingStore.getState().trackSingleFrame(nextIdx);
        if (!success) {
          isPlayingWithTrackRef.current = false;
          setIsPlayingWithTrack(false);
          return;
        }
        usePlayerStore.getState().goToFrame(nextIdx);
      }

      // Schedule next frame — use fresh speed values
      const fr = useStudyStore.getState().metadata?.frameRate ?? 15;
      const spd = usePlayerStore.getState().playbackSpeed;
      const delay = 1000 / (fr * spd);
      playbackIntervalRef.current = setTimeout(trackNextFrame, delay);
    };

    trackNextFrame();
  }, []);

  const stopPlayWithTracking = useCallback(() => {
    isPlayingWithTrackRef.current = false;
    setIsPlayingWithTrack(false);
    if (playbackIntervalRef.current) {
      clearTimeout(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
  }, []);

  // --- Unified handlers ---
  const handlePlayPause = useCallback(() => {
    if (isTrackMode) {
      if (isPlayingWithTrack) {
        stopPlayWithTracking();
      } else {
        if (!currentRoi) return;
        playWithTracking();
      }
    } else {
      togglePlayPause();
    }
  }, [isTrackMode, isPlayingWithTrack, currentRoi, playWithTracking, stopPlayWithTracking, togglePlayPause]);

  const handleStepForward = useCallback(async () => {
    if (isTrackMode) {
      // Use getState() for fresh values (avoids stale closure on rapid presses)
      const ps = usePlayerStore.getState();
      const frame = ps.currentFrame;
      const hi = ps.rangeEnd ?? ps.totalFrames - 1;
      const roi = useAnalysisStore.getState().rois.get(frame) ?? null;
      if (!roi) return;

      const trackingStore = useTrackingStore.getState();

      // Auto-initialize if not yet initialized
      if (!trackingStore.isInitialized) {
        await trackingStore.initializeTracking(frame, [
          Math.round(roi.x), Math.round(roi.y),
          Math.round(roi.width), Math.round(roi.height),
        ]);
        if (!useTrackingStore.getState().isInitialized) return;
        if (useTrackingStore.getState().autoSegQcaEnabled) {
          useAnalysisStore.getState().segmentAndExtract(frame, roi).catch(() => {});
        }
      }

      const nextIdx = Math.min(frame + 1, hi);
      if (nextIdx !== frame) {
        const success = await useTrackingStore.getState().trackSingleFrame(nextIdx);
        if (success) {
          goToFrame(nextIdx);
        }
      }
    } else {
      stepForward();
    }
  }, [isTrackMode, goToFrame, stepForward]);

  const handleToggleTrackMode = useCallback(() => {
    if (isTrackMode) {
      if (isPlayingWithTrack) stopPlayWithTracking();
      disableTrackMode();
    } else {
      enableTrackMode();
    }
  }, [isTrackMode, isPlayingWithTrack, stopPlayWithTracking, enableTrackMode, disableTrackMode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          handlePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          stepBackward();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleStepForward();
          break;
        case 'ArrowUp':
          e.preventDefault();
          {
            const currentIdx = SPEED_OPTIONS.indexOf(playbackSpeed);
            if (currentIdx < SPEED_OPTIONS.length - 1) {
              setPlaybackSpeed(SPEED_OPTIONS[currentIdx + 1]);
            }
          }
          break;
        case 'ArrowDown':
          e.preventDefault();
          {
            const currentIdx = SPEED_OPTIONS.indexOf(playbackSpeed);
            if (currentIdx > 0) {
              setPlaybackSpeed(SPEED_OPTIONS[currentIdx - 1]);
            }
          }
          break;
        case 'Home':
          e.preventDefault();
          goToFrame(usePlayerStore.getState().rangeStart ?? 0);
          break;
        case 'End':
          e.preventDefault();
          goToFrame(usePlayerStore.getState().rangeEnd ?? totalFrames - 1);
          break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handlePlayPause, handleStepForward, stepBackward, playbackSpeed, setPlaybackSpeed, goToFrame, totalFrames]);

  if (!metadata) {
    return (
      <div className="h-playback bg-surface-secondary border-t border-border flex items-center justify-center shrink-0">
        <span className="text-sm text-content-muted">No study loaded</span>
      </div>
    );
  }

  const isPlaying = isTrackMode ? isPlayingWithTrack : playbackState === 'playing';

  return (
    <div className="h-playback bg-surface-secondary border-t border-border flex flex-col shrink-0">
      {/* Track mode indicator */}
      {isTrackMode && (
        <div className="flex items-center justify-center gap-2 py-0.5 border-b border-border">
          <span className="bg-orange-600 text-white text-[10px] px-1.5 py-0.5 rounded font-medium animate-pulse">
            TRACK
          </span>
          {autoSegQcaEnabled && (
            <span className="bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">
              +SEG/QCA
            </span>
          )}
          {confidence > 0 && (
            <span className={`text-[10px] font-mono ${confidence >= confidenceThreshold ? 'text-green-400' : 'text-red-400'}`}>
              {(confidence * 100).toFixed(0)}%
            </span>
          )}
          {isTracking && (
            <span className="w-1.5 h-1.5 bg-orange-400 rounded-full animate-pulse" />
          )}
        </div>
      )}

      {/* Controls row */}
      <div className="flex-1 flex items-center px-4 gap-3">
        {/* Transport buttons */}
        <div className="flex items-center gap-1">
          <CtrlButton onClick={() => goToFrame(effectiveStart)} title="First frame (Home)">{'\u23EE'}</CtrlButton>
          <CtrlButton
            onClick={stepBackward}
            title="Previous frame (\u2190)"
            trackMode={isTrackMode}
          >
            {'\u23EA'}
          </CtrlButton>
          <CtrlButton
            onClick={handlePlayPause}
            title={isTrackMode ? 'Play with tracking (Space)' : 'Play/Pause (Space)'}
            primary
            trackMode={isTrackMode}
          >
            {isPlaying ? '\u23F8' : '\u25B6'}
          </CtrlButton>
          <CtrlButton
            onClick={handleStepForward}
            title="Next frame (\u2192)"
            trackMode={isTrackMode}
          >
            {'\u23E9'}
          </CtrlButton>
          <CtrlButton onClick={() => goToFrame(effectiveEnd)} title="Last frame (End)">{'\u23ED'}</CtrlButton>
        </div>

        {/* Frame slider */}
        <input
          type="range"
          min={effectiveStart}
          max={effectiveEnd}
          value={currentFrame}
          onChange={(e) => {
            if (isPlayingWithTrack) stopPlayWithTracking();
            goToFrame(parseInt(e.target.value));
          }}
          className="flex-1 h-1.5 accent-brand cursor-pointer"
          title={`Frame ${currentFrame + 1}/${totalFrames}${hasRange ? ` [${effectiveStart + 1}–${effectiveEnd + 1}]` : ''}`}
        />

        {/* Frame counter */}
        <span className="text-xs text-content-secondary font-mono min-w-[80px] text-center">
          {currentFrame + 1}/{totalFrames}
          {hasRange && (
            <span className="text-brand text-[10px] ml-0.5">
              [{effectiveStart + 1}–{effectiveEnd + 1}]
            </span>
          )}
        </span>

        {/* Speed selector */}
        <select
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
          className="text-xs bg-surface-tertiary text-content-primary border border-border rounded px-1 py-0.5"
        >
          {SPEED_OPTIONS.map((s) => (
            <option key={s} value={s}>{s}x</option>
          ))}
        </select>

        {/* Loop toggle */}
        <button
          onClick={toggleLoop}
          className={`text-xs px-2 py-0.5 rounded ${isLooping ? 'bg-brand text-white' : 'text-content-muted hover:text-content-primary'}`}
          title="Toggle loop"
        >
          {'\u{1F501}'}
        </button>

        {/* Separator */}
        <div className="w-px h-5 bg-border" />

        {/* Track Mode toggle */}
        <button
          onClick={handleToggleTrackMode}
          disabled={!currentRoi && !isTrackMode}
          className={`text-xs px-2 py-1 rounded font-medium transition-colors ${
            isTrackMode
              ? 'bg-orange-600 text-white hover:bg-orange-500'
              : 'text-content-muted hover:text-content-primary hover:bg-surface-tertiary disabled:opacity-40 disabled:cursor-not-allowed'
          }`}
          title={currentRoi ? 'Toggle Track Mode' : 'Draw ROI first (B key)'}
        >
          Track
        </button>

        {/* Auto Seg/QCA toggle (shown only when track mode is on) */}
        {isTrackMode && (
          <button
            onClick={() => setAutoSegQcaEnabled(!autoSegQcaEnabled)}
            className={`text-xs px-2 py-1 rounded font-medium transition-colors ${
              autoSegQcaEnabled
                ? 'bg-green-600 text-white hover:bg-green-500'
                : 'text-content-muted hover:text-content-primary hover:bg-surface-tertiary'
            }`}
            title="Auto-run segmentation + QCA on each tracked frame"
          >
            +Seg/QCA
          </button>
        )}
      </div>
    </div>
  );
}

function CtrlButton({ children, onClick, title, primary = false, trackMode = false }: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  primary?: boolean;
  trackMode?: boolean;
}) {
  const trackClass = trackMode ? (primary ? 'bg-orange-600 hover:bg-orange-500 text-white' : 'text-orange-400 hover:bg-surface-tertiary') : '';
  return (
    <button
      onClick={onClick}
      title={title}
      className={`w-8 h-8 flex items-center justify-center rounded text-sm transition-colors
        ${trackClass || (primary ? 'bg-brand text-white hover:bg-blue-600' : 'text-content-secondary hover:bg-surface-tertiary hover:text-content-primary')}`}
    >
      {children}
    </button>
  );
}
