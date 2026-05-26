import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import {
  absUrl, dismissReview, labelReview, listPeople, listReview,
  type PersonSummary, type ReviewItem,
} from '../api/recognition';

const POLL_MS = 10000;

export function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    try {
      const [it, ppl] = await Promise.all([listReview(), listPeople()]);
      setItems(it);
      setPeople(ppl);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(t);
  }, []);

  async function onConfirm(item: ReviewItem) {
    if (!item.suggested) return;
    try {
      await labelReview(item.id, { person_id: item.suggested.id });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onLabelNew(item: ReviewItem) {
    const name = prompt('Name for new person?');
    if (!name?.trim()) return;
    try {
      await labelReview(item.id, { new_name: name.trim() });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onLabelExisting(item: ReviewItem, person_id: number) {
    try {
      await labelReview(item.id, { person_id });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onDismiss(item: ReviewItem) {
    try {
      await dismissReview(item.id);
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <h2 style={{ margin: 0 }}>Review queue ({items.length})</h2>
      {err && <p style={{ color: '#f87171' }}>err: {err}</p>}
      {items.length === 0 && <p style={{ color: '#777' }}>No pending faces. Triggers fill this queue automatically.</p>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12, marginTop: 16 }}>
        {items.map((it) => (
          <div key={it.id} style={{ background: '#141414', border: '1px solid #222', borderRadius: 6, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <img src={absUrl(it.crop_url)} alt={`q${it.id}`}
              style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }} />
            {it.suggested ? (
              <div style={{ fontSize: 12 }}>
                Maybe <b>{it.suggested.name}</b> ({it.suggested.score.toFixed(2)})
              </div>
            ) : (
              <div style={{ fontSize: 12, color: '#888' }}>Unknown face</div>
            )}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {it.suggested && (
                <button onClick={() => void onConfirm(it)} style={btn('#2e6cdf')}>Confirm</button>
              )}
              <button onClick={() => void onLabelNew(it)} style={btn('#16a34a')}>New...</button>
              <select
                onChange={(e) => { if (e.target.value) void onLabelExisting(it, Number(e.target.value)); e.target.value=''; }}
                defaultValue=""
                style={{ padding: '4px 6px', background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4, fontSize: 11 }}>
                <option value="">Label existing...</option>
                {people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <button onClick={() => void onDismiss(it)} style={btn('#444')}>Dismiss</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function btn(bg: string): CSSProperties {
  return {
    padding: '4px 8px', background: bg, color: '#fff', border: 0,
    borderRadius: 4, cursor: 'pointer', fontSize: 11,
  };
}
