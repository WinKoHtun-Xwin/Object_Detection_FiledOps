// Shared types — match backend protocol/result schema.

export type ModeId = 'yolo_detect';

export type YoloSize = 'n' | 's' | 'm' | 'l' | 'x';

export interface Box {
  x: number; y: number; w: number; h: number;
  label?: string;
  conf?: number;
}

export interface FaceMatch {
  bbox: [number, number, number, number];   // normalized x, y, w, h
  name: string;
  score: number;
  kind: 'high' | 'mid' | 'unknown';
}

export type InferenceResult =
  | { type: 'detect'; frame_id: number; ms: number; boxes: Box[]; faces?: FaceMatch[] };
