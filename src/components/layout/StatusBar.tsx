import { useSessionStore } from '@/stores/sessionStore';

export function StatusBar() {
  const isConnected = useSessionStore((s) => s.isConnected);
  const backendVersion = useSessionStore((s) => s.backendVersion);

  return (
    <footer className="h-statusbar flex items-center px-4 bg-surface-secondary border-t border-border text-xs shrink-0">
      <span className="text-amber-500 font-medium">
        For Research Use Only &mdash; Not for Clinical Decision Making
      </span>
      <span className="ml-auto text-content-muted" aria-live="polite">
        Backend: {isConnected ? (
          <span className="text-green-500">Connected{backendVersion ? ` (v${backendVersion})` : ''}</span>
        ) : (
          <span className="text-red-500">Disconnected</span>
        )}
      </span>
    </footer>
  );
}
