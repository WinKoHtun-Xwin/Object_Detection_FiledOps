import { useEffect, useState } from 'react';
import { useAppState } from '../state/appState';

interface DeviceOption {
  deviceId: string;
  label: string;
}

export function CameraPicker() {
  const selectedDeviceId = useAppState((s) => s.selectedDeviceId);
  const setSelectedDeviceId = useAppState((s) => s.setSelectedDeviceId);
  const cameraId = useAppState((s) => s.cameraId);
  const setCameraId = useAppState((s) => s.setCameraId);

  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        // Request a temporary stream so device labels are populated.
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
        tmp.getTracks().forEach((t) => t.stop());

        const all = await navigator.mediaDevices.enumerateDevices();
        const cams = all
          .filter((d) => d.kind === 'videoinput')
          .map((d, i) => ({
            deviceId: d.deviceId,
            label: d.label || `Camera ${i + 1}`,
          }));
        setDevices(cams);
        if (cams.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(cams[0].deviceId);
          setCameraId(`cam-${cams[0].deviceId.slice(0, 6) || 'default'}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    // selectedDeviceId/setters omitted intentionally — run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <h4>Camera</h4>
      <select
        value={selectedDeviceId ?? ''}
        onChange={(e) => {
          setSelectedDeviceId(e.target.value);
          const dev = devices.find((d) => d.deviceId === e.target.value);
          if (dev) setCameraId(`cam-${dev.deviceId.slice(0, 6)}`);
        }}
        style={{
          width: '100%', padding: 6, background: '#0c0c0c',
          color: '#eee', border: '1px solid #333', borderRadius: 4,
        }}
      >
        {devices.length === 0 && <option value="">(no cameras)</option>}
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
        ))}
      </select>

      <label style={{ display: 'block', marginTop: 8, fontSize: 12, color: '#aaa' }}>
        Camera id (used for clip folder name)
        <input
          type="text"
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
          style={{
            display: 'block', width: '100%', marginTop: 4, padding: 6,
            background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4,
          }}
        />
      </label>

      {error && (
        <p style={{ color: '#f87171', fontSize: 12, marginTop: 6 }}>{error}</p>
      )}
    </div>
  );
}
