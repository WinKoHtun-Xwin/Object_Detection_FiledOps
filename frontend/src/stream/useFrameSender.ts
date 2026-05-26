import { useEffect, useRef } from 'react';
import { useAppState } from '../state/appState';
import { buildPacket } from './protocol';

interface UseFrameSenderArgs {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  ready: boolean;
  send: (data: ArrayBuffer) => boolean;
  maxSide?: number;       // longest side after downscale
  jpegQuality?: number;   // 0..1
}

// Captures frames via canvas, encodes JPEG, calls send().
// Drops the frame if send() returns false (WS not open OR in-flight).
// Tracks FPS of *sent* (accepted) frames and writes to Zustand.
export function useFrameSender({
  videoRef,
  ready,
  send,
  maxSide = 640,
  jpegQuality = 0.7,
}: UseFrameSenderArgs) {
  const setFps = useAppState((s) => s.setFps);
  const paused = useAppState((s) => s.paused);
  const mode = useAppState((s) => s.mode);
  const yoloSize = useAppState((s) => s.yoloSize);
  const sam3Prompt = useAppState((s) => s.sam3Prompt);
  const sam3Text = useAppState((s) => s.sam3Text);
  const tracking = useAppState((s) => s.tracking);
  const yoloConf = useAppState((s) => s.yoloConf);
  const cameraId = useAppState((s) => s.cameraId);
  const recognizeFaces = useAppState((s) => s.recognizeFaces);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  if (canvasRef.current === null && typeof document !== 'undefined') {
    canvasRef.current = document.createElement('canvas');
  }

  // Keep latest mode/prompt in a ref so the RAF loop reads fresh values
  // without restarting on every state change.
  const stateRef = useRef({ mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId, recognizeFaces });
  stateRef.current = { mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId, recognizeFaces };

  useEffect(() => {
    if (!ready) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let sentCount = 0;
    let lastReport = performance.now();

    function variantIdFor(s: typeof stateRef.current): number {
      if (s.mode.startsWith('yolo_')) {
        return { n: 0, s: 1, m: 2, l: 3, x: 4 }[s.yoloSize];
      }
      return { text: 0, point: 1, box: 2 }[s.sam3Prompt];
    }

    function buildHeader(s: typeof stateRef.current): Record<string, unknown> {
      const base: Record<string, unknown> = { camera_id: s.cameraId, recognize: s.recognizeFaces };
      if (s.mode.startsWith('yolo_')) {
        return { ...base, conf: s.yoloConf };
      }
      if (s.mode === 'sam3_image' || s.mode === 'sam3_video') {
        return {
          ...base,
          prompt: s.sam3Prompt,
          text: s.sam3Text,
          tracking: s.tracking,
        };
      }
      return base;
    }

    async function tick() {
      const s = stateRef.current;
      // video/canvas/ctx are non-null: guarded before useEffect body runs.
      const v = video!;
      const c = canvas!;
      const x = ctx!;
      if (!s.paused && v.videoWidth > 0) {
        const vw = v.videoWidth;
        const vh = v.videoHeight;
        const scale = Math.min(1, maxSide / Math.max(vw, vh));
        const w = Math.round(vw * scale);
        const h = Math.round(vh * scale);
        if (c.width !== w) c.width = w;
        if (c.height !== h) c.height = h;
        x.drawImage(v, 0, 0, w, h);
        const blob: Blob | null = await new Promise((res) =>
          c.toBlob(res, 'image/jpeg', jpegQuality),
        );
        if (blob) {
          const pkt = await buildPacket({
            mode: s.mode,
            variantId: variantIdFor(s),
            header: buildHeader(s),
            jpeg: blob,
          });
          const accepted = send(pkt);
          if (accepted) sentCount++;
        }
      }
      const now = performance.now();
      if (now - lastReport >= 500) {
        const fps = (sentCount * 1000) / (now - lastReport);
        setFps(Math.round(fps * 10) / 10);
        sentCount = 0;
        lastReport = now;
      }
      raf = requestAnimationFrame(() => void tick());
    }
    raf = requestAnimationFrame(() => void tick());
    return () => cancelAnimationFrame(raf);
  }, [ready, videoRef, send, maxSide, jpegQuality, setFps]);
}
