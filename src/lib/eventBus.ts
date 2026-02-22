import type { StudyMetadata } from '@/types';

type EventMap = {
  'study:loaded': { sessionId: string; metadata: StudyMetadata };
  'study:cleared': void;
  'frame:changed': { frameIndex: number };
  'segmentation:completed': { frameIndex: number };
  'calibration:changed': { pixelSpacing: number };
  'beat:selected': { startFrame: number; endFrame: number };
  'mask-edit:saved': { frameIndex: number };
  'tracking:updated': { frameIndex: number };
};

type EventHandler<T> = T extends void ? () => void : (data: T) => void;

class EventBus {
  private listeners = new Map<string, Set<Function>>();

  on<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.off(event, handler);
  }

  off<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): void {
    this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof EventMap>(
    event: K,
    ...args: EventMap[K] extends void ? [] : [EventMap[K]]
  ): void {
    this.listeners.get(event)?.forEach((handler) => {
      try {
        (handler as Function)(...args);
      } catch (error) {
        console.error(`EventBus error in handler for "${String(event)}":`, error);
      }
    });
  }
}

export const eventBus = new EventBus();
export type { EventMap };
