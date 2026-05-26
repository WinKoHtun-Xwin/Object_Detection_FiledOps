import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  absUrl, createPerson, listPeople, type PersonSummary,
} from '../api/recognition';

export function PeoplePage() {
  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  async function refresh() {
    try { setPeople(await listPeople()); setError(null); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }

  useEffect(() => { void refresh(); }, []);

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>People ({people.length})</h2>
        <button
          onClick={() => setShowAdd(true)}
          style={{ padding: '6px 12px', background: '#2e6cdf', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}
        >
          + Add person
        </button>
      </div>

      {error && <p style={{ color: '#f87171' }}>err: {error}</p>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginTop: 16 }}>
        {people.map((p) => (
          <Link key={p.id} to={`/people/${p.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
            <div style={{ background: '#141414', border: '1px solid #222', borderRadius: 6, padding: 12 }}>
              {p.latest_face_url ? (
                <img src={absUrl(p.latest_face_url)} alt={p.name}
                  style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }} />
              ) : (
                <div style={{ width: '100%', aspectRatio: '1', background: '#222', borderRadius: 4 }} />
              )}
              <div style={{ marginTop: 8, fontWeight: 600 }}>{p.name}</div>
              <div style={{ fontSize: 11, color: '#888' }}>{p.face_count} face(s)</div>
            </div>
          </Link>
        ))}
      </div>

      {showAdd && (
        <AddPersonModal onClose={() => setShowAdd(false)} onCreated={() => { void refresh(); setShowAdd(false); }} />
      )}
    </div>
  );
}

function AddPersonModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!name.trim() || files.length === 0) {
      setErr('name and at least one image required');
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await createPerson(name.trim(), files);
      onCreated();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)',
      display: 'grid', placeItems: 'center', zIndex: 1000,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: '#141414', padding: 24, borderRadius: 8, minWidth: 320,
        display: 'flex', flexDirection: 'column', gap: 12,
      }}>
        <h3 style={{ margin: 0 }}>Add person</h3>
        <input
          type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)}
          style={{ padding: 8, background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4 }}
        />
        <input
          type="file" multiple accept="image/*"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          style={{ color: '#aaa' }}
        />
        <div style={{ fontSize: 11, color: '#888' }}>{files.length} file(s) selected</div>
        {err && <p style={{ color: '#f87171', fontSize: 12 }}>{err}</p>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose} disabled={busy}
            style={{ padding: '6px 12px', background: 'transparent', color: '#aaa', border: '1px solid #333', borderRadius: 4, cursor: 'pointer' }}>
            Cancel
          </button>
          <button onClick={() => void submit()} disabled={busy}
            style={{ padding: '6px 12px', background: '#2e6cdf', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}>
            {busy ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
