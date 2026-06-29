import { useEffect, useState } from 'react';
import { useAppState } from '../state/appState';
import { useWebcam } from '../camera/useWebcam';
import { CameraView } from '../camera/CameraView';
import { CameraPicker } from '../camera/CameraPicker';
import { ClipsPanel } from '../clips/ClipsPanel';
import { useInferenceWS } from '../stream/useInferenceWS';
import { useFrameSender } from '../stream/useFrameSender';
import { OverlayCanvas } from '../overlay/OverlayCanvas';
import type { YoloSize } from '../types';
import '../App.css';

const WS_URL = 'ws://localhost:8000/ws';

const SIZES: YoloSize[] = ['n', 's', 'm', 'l', 'x'];

export function LivePage() {
  const yoloSize = useAppState((s) => s.yoloSize);
  const setYoloSize = useAppState((s) => s.setYoloSize);
  const mirror = useAppState((s) => s.mirror);
  const setMirror = useAppState((s) => s.setMirror);
  const paused = useAppState((s) => s.paused);
  const setPaused = useAppState((s) => s.setPaused);
  const recognizeFaces = useAppState((s) => s.recognizeFaces);
  const setRecognizeFaces = useAppState((s) => s.setRecognizeFaces);
  const yoloConf = useAppState((s) => s.yoloConf);
  const setYoloConf = useAppState((s) => s.setYoloConf);
  const fps = useAppState((s) => s.fps);
  const inferenceMs = useAppState((s) => s.inferenceMs);
  const lastResult = useAppState((s) => s.lastResult);
  const pushResult = useAppState((s) => s.pushResult);

  const selectedDeviceId = useAppState((s) => s.selectedDeviceId);
  const { videoRef, ready, error } = useWebcam({ deviceId: selectedDeviceId });
  const { status, send, lastMessage } = useInferenceWS(WS_URL);
  useFrameSender({ videoRef, ready, send });

  const [wsError, setWsError] = useState<string | null>(null);
  useEffect(() => {
    if (!lastMessage) return;
    if (lastMessage.type === 'error') setWsError(lastMessage.message);
    else {
      setWsError(null);
      pushResult(lastMessage);
    }
  }, [lastMessage, pushResult]);

  const objectCount = lastResult?.type === 'detect' ? lastResult.boxes.length : 0;

  return (
    <div style={{ display: 'flex', height: '100%', fontFamily: 'system-ui', color: '#eee', background: '#0c0c0c' }}>
      <aside style={{ width: 280, padding: 16, borderRight: '1px solid #222', background: '#141414', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0 }}>Object Detection</h2>
        <CameraPicker />

        <h4>Model size</h4>
        <select
          value={yoloSize}
          onChange={(e) => setYoloSize(e.target.value as YoloSize)}
          style={{
            width: '100%',
            padding: 6,
            background: '#0c0c0c',
            color: '#eee',
            border: '1px solid #333',
            borderRadius: 4,
          }}
        >
          {SIZES.map((s) => (
            <option key={s} value={s}>
              {s} ({sizeLabel(s)})
            </option>
          ))}
        </select>

        <h4>Confidence</h4>
        <label style={{ display: 'block', fontSize: 12 }}>
          {yoloConf.toFixed(2)}
          <input
            type="range"
            min={0.05}
            max={0.9}
            step={0.05}
            value={yoloConf}
            onChange={(e) => setYoloConf(Number(e.target.value))}
            style={{ display: 'block', width: '100%' }}
          />
        </label>

        <h4>Controls</h4>
        <label style={{ display: 'block', margin: '4px 0' }}>
          <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} /> Mirror
        </label>
        <label style={{ display: 'block', margin: '4px 0' }}>
          <input type="checkbox" checked={paused} onChange={(e) => setPaused(e.target.checked)} /> Pause
        </label>
        <label style={{ display: 'block', margin: '4px 0' }}>
          <input type="checkbox" checked={recognizeFaces}
            onChange={(e) => setRecognizeFaces(e.target.checked)} /> Recognize faces
        </label>

        <ClipsPanel />

        <p style={{ color: '#888', fontSize: 12, marginTop: 24 }}>
          YOLO26 object detection × n/s/m/l/x.
        </p>
      </aside>

      <main style={{ flex: 1, position: 'relative', display: 'grid', placeItems: 'center' }}>
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          <CameraView ref={videoRef} ready={ready} error={error} />
          <OverlayCanvas videoRef={videoRef} />
        </div>

        <div
          style={{
            position: 'absolute',
            top: 12,
            right: 12,
            background: 'rgba(0,0,0,0.6)',
            padding: '8px 12px',
            borderRadius: 6,
            fontSize: 12,
            lineHeight: 1.6,
            fontFamily: 'ui-monospace, monospace',
          }}
        >
          <div>WS: <b style={{ color: status === 'open' ? '#4f4' : '#f55' }}>{status}</b></div>
          <div>mode: <b>detect · {yoloSize}</b></div>
          <div>FPS: <b>{fps.toFixed(1)}</b></div>
          <div>inference: <b>{inferenceMs.toFixed(1)} ms</b></div>
          <div>objects: <b>{objectCount}</b></div>
          {wsError && (
            <div style={{ color: '#f87171', marginTop: 4, maxWidth: 240, whiteSpace: 'pre-wrap' }}>
              err: {wsError}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function sizeLabel(s: YoloSize): string {
  return { n: 'nano', s: 'small', m: 'medium', l: 'large', x: 'extra' }[s];
}
