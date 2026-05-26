import { forwardRef } from 'react';
import { useAppState } from '../state/appState';

interface Props {
  ready: boolean;
  error: string | null;
}

export const CameraView = forwardRef<HTMLVideoElement, Props>(function CameraView(
  { ready, error },
  ref,
) {
  const mirror = useAppState((s) => s.mirror);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <video
        ref={ref}
        playsInline
        muted
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          transform: mirror ? 'scaleX(-1)' : 'none',
          background: '#000',
        }}
      />
      {!ready && !error && (
        <Overlay text="Requesting camera permission…" />
      )}
      {error && <Overlay text={`Camera error: ${error}`} color="#f66" />}
    </div>
  );
});

function Overlay({ text, color = '#bbb' }: { text: string; color?: string }) {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'grid',
        placeItems: 'center',
        color,
        background: 'rgba(0,0,0,0.6)',
        fontFamily: 'system-ui',
      }}
    >
      {text}
    </div>
  );
}
