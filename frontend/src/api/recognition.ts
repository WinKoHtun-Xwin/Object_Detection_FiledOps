const API_BASE = 'http://localhost:8001';

export interface PersonSummary {
  id: number;
  name: string;
  face_count: number;
  latest_face_url: string | null;
}

export interface PersonFace {
  id: number;
  crop_url: string;
  source: 'upload' | 'queue';
  created_at: number;
}

export interface PersonDetail {
  id: number;
  name: string;
  faces: PersonFace[];
}

export interface PersonClip {
  clip_path: string;
  mp4_url: string;
  jpg_url: string;
  confidence: number;
  frame_ts: number;
  created_at: number;
}

export interface ReviewItem {
  id: number;
  crop_url: string;
  suggested: { id: number; name: string; score: number } | null;
  source_clip: string | null;
  created_at: number;
}

export function absUrl(path: string): string {
  return path.startsWith('http') ? path : `${API_BASE}${path}`;
}

export async function listPeople(): Promise<PersonSummary[]> {
  const r = await fetch(`${API_BASE}/api/people`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function getPerson(id: number): Promise<PersonDetail> {
  const r = await fetch(`${API_BASE}/api/people/${id}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function listPersonClips(id: number): Promise<PersonClip[]> {
  const r = await fetch(`${API_BASE}/api/people/${id}/clips`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function createPerson(name: string, files: File[]): Promise<PersonSummary> {
  const fd = new FormData();
  fd.set('name', name);
  for (const f of files) fd.append('files', f);
  const r = await fetch(`${API_BASE}/api/people`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function deletePerson(id: number): Promise<void> {
  const r = await fetch(`${API_BASE}/api/people/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
}

export async function listReview(limit = 50): Promise<ReviewItem[]> {
  const r = await fetch(`${API_BASE}/api/review?limit=${limit}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function labelReview(id: number, body: { person_id?: number; new_name?: string }): Promise<{ person_id: number }> {
  const r = await fetch(`${API_BASE}/api/review/${id}/label`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function dismissReview(id: number): Promise<void> {
  const r = await fetch(`${API_BASE}/api/review/${id}/dismiss`, { method: 'POST' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
}
