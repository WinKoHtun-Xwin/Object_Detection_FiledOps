import { useEffect, useRef } from 'react';
import { useAppState } from '../state/appState';
import { drawBoxes } from './drawBoxes';

interface Props {
  readonly videoRef: React.RefObject<HTMLVideoElement | null>;
}

// Renders detection results on top of the <video>.
// Sized to match the video element's intrinsic resolution so coordinates
// stay correct under CSS `object-fit: contain`.
export function OverlayCanvas({ videoRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastResult = useAppState((s) => s.lastResult);
  const mirror = useAppState((s) => s.mirror);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas internal resolution to video's; CSS handles display fit.
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    if (canvas.width !== vw) canvas.width = vw;
    if (canvas.height !== vh) canvas.height = vh;

    if (!lastResult) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    if (lastResult.type === 'detect') {
      drawBoxes(ctx, lastResult.boxes, false); // video itself is mirrored via CSS
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [lastResult, videoRef, mirror]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        pointerEvents: 'none',
        transform: mirror ? 'scaleX(-1)' : 'none',
      }}
    />
  );
}
