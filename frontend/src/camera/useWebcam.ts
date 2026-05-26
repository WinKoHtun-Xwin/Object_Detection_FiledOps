import { useEffect, useRef, useState } from 'react';

interface UseWebcamOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  deviceId?: string | null;
}

interface UseWebcamResult {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  ready: boolean;
  error: string | null;
}

export function useWebcam(opts: UseWebcamOptions = {}): UseWebcamResult {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    setReady(false);

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: opts.width ?? 1280 },
            height: { ideal: opts.height ?? 720 },
            facingMode: opts.facingMode ?? 'user',
            ...(opts.deviceId ? { deviceId: { exact: opts.deviceId } } : {}),
          },
          audio: false,
        });
        if (cancelled || !videoRef.current) {
          stream?.getTracks().forEach((t) => t.stop());
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [opts.width, opts.height, opts.facingMode, opts.deviceId]);

  return { videoRef, ready, error };
}
