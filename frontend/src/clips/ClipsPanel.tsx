import { useEffect, useState } from 'react';
import { useAppState } from '../state/appState';

const API_BASE = 'http://localhost:8001';
const POLL_INTERVAL_MS = 10000;

interface ClipEntry {
  name: string;
  date: string;
  mp4_url: string;
  jpg_url: string | null;
  size_bytes: number;
  mtime: number;
}

export function ClipsPanel() {
  const cameraId = useAppState((s) => s.cameraId);
  const [clips, setClips] = useState<ClipEntry[]>([]);
  const [openClip, setOpenClip] = useState<ClipEntry | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!cameraId) return;
    let cancelled = false;

    async function load() {
      try {
        const r = await fetch(
          `${API_BASE}/api/clips?camera_id=${encodeURIComponent(cameraId)}&limit=20`,
        );
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!cancelled) {
          setClips(data.clips ?? []);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    }

    void load();
    const t = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [cameraId]);

  return (
    <>
      <h4 style={{ marginTop: 24 }}>Recent clips ({clips.length})</h4>
      {error && (
        <p style={{ color: '#f87171', fontSize: 11 }}>err: {error}</p>
      )}
      {clips.length === 0 && !error && (
        <p style={{ color: '#777', fontSize: 11 }}>No clips yet. Move in front of the camera.</p>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {clips.map((c) => (
          <button
            key={c.mp4_url}
            onClick={() => setOpenClip(c)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: 4,
              background: '#0c0c0c',
              border: '1px solid #222',
              borderRadius: 4,
              cursor: 'pointer',
              textAlign: 'left',
              color: '#eee',
            }}
            title={`${c.date} ${c.name}`}
          >
            {c.jpg_url ? (
              <img
                src={`${API_BASE}${c.jpg_url}`}
                alt={c.name}
                style={{ width: 64, height: 36, objectFit: 'cover', borderRadius: 2 }}
              />
            ) : (
              <div style={{ width: 64, height: 36, background: '#222', borderRadius: 2 }} />
            )}
            <div style={{ fontSize: 11, lineHeight: 1.3 }}>
              <div style={{ fontFamily: 'ui-monospace, monospace' }}>{c.name}</div>
              <div style={{ color: '#888' }}>{c.date} · {(c.size_bytes / 1024).toFixed(0)} KB</div>
            </div>
          </button>
        ))}
      </div>

      {openClip && (
        <ClipModal clip={openClip} onClose={() => setOpenClip(null)} />
      )}
    </>
  );
}

function ClipModal({ clip, onClose }: { clip: ClipEntry; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.85)',
        display: 'grid',
        placeItems: 'center',
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#141414',
          padding: 16,
          borderRadius: 8,
          maxWidth: '90vw',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>
            {clip.date} · {clip.name}
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              color: '#eee',
              border: '1px solid #333',
              borderRadius: 4,
              padding: '4px 10px',
              cursor: 'pointer',
            }}
          >
            close (esc)
          </button>
        </div>
        <video
          controls
          autoPlay
          src={`${API_BASE}${clip.mp4_url}`}
          style={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 4 }}
        />
      </div>
    </div>
  );
}
