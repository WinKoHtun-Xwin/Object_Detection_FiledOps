import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  absUrl, deletePerson, getPerson, listPersonClips,
  type PersonClip, type PersonDetail,
} from '../api/recognition';

export function PersonDetailPage() {
  const { id } = useParams<{ id: string }>();
  const nav = useNavigate();
  const pid = id ? Number(id) : NaN;
  const [person, setPerson] = useState<PersonDetail | null>(null);
  const [clips, setClips] = useState<PersonClip[]>([]);
  const [openClip, setOpenClip] = useState<PersonClip | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(pid)) return;
    (async () => {
      try {
        const [p, c] = await Promise.all([getPerson(pid), listPersonClips(pid)]);
        setPerson(p);
        setClips(c);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [pid]);

  async function onDelete() {
    if (!person) return;
    if (!confirm(`Delete ${person.name} and all their faces?`)) return;
    try {
      await deletePerson(person.id);
      nav('/people');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  if (err) return <div style={{ padding: 16, color: '#f87171' }}>err: {err}</div>;
  if (!person) return <div style={{ padding: 16, color: '#aaa' }}>loading...</div>;

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>{person.name}</h2>
        <button onClick={() => void onDelete()}
          style={{ padding: '6px 12px', background: '#c2410c', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}>
          Delete person
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginTop: 16 }}>
        <section>
          <h3>Gallery ({person.faces.length})</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))', gap: 6 }}>
            {person.faces.map((f) => (
              <img key={f.id} src={absUrl(f.crop_url)} alt={`face ${f.id}`}
                style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }}
                title={f.source} />
            ))}
          </div>
        </section>

        <section>
          <h3>Clips ({clips.length})</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {clips.length === 0 && <p style={{ color: '#777', fontSize: 12 }}>No clips yet.</p>}
            {clips.map((c) => (
              <button key={c.clip_path} onClick={() => setOpenClip(c)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: 4,
                  background: '#0c0c0c', border: '1px solid #222', borderRadius: 4,
                  cursor: 'pointer', textAlign: 'left', color: '#eee',
                }}>
                <img src={absUrl(c.jpg_url)} alt={c.clip_path}
                  style={{ width: 80, height: 45, objectFit: 'cover', borderRadius: 2 }} />
                <div style={{ fontSize: 11, lineHeight: 1.3 }}>
                  <div style={{ fontFamily: 'ui-monospace, monospace' }}>{c.clip_path}</div>
                  <div style={{ color: '#888' }}>conf {c.confidence.toFixed(2)} · t={c.frame_ts.toFixed(1)}s</div>
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>

      {openClip && <ClipModal clip={openClip} onClose={() => setOpenClip(null)} />}
    </div>
  );
}

function ClipModal({ clip, onClose }: { clip: PersonClip; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)',
      display: 'grid', placeItems: 'center', zIndex: 1000,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: '#141414', padding: 16, borderRadius: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{clip.clip_path}</div>
          <button onClick={onClose}
            style={{ background: 'transparent', color: '#eee', border: '1px solid #333', borderRadius: 4, padding: '4px 10px', cursor: 'pointer' }}>
            close
          </button>
        </div>
        <video controls autoPlay src={absUrl(clip.mp4_url)}
          style={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 4 }} />
      </div>
    </div>
  );
}
