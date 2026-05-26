// Shared types — match backend protocol/result schema.

export type ModeId =
  | 'yolo_detect'
  | 'yolo_seg'
  | 'yolo_pose'
  | 'yolo_obb'
  | 'yolo_cls'
  | 'sam3_image'
  | 'sam3_video';

export type YoloSize = 'n' | 's' | 'm' | 'l' | 'x';

export type Sam3PromptKind = 'text' | 'point' | 'box';

export interface Box {
  x: number; y: number; w: number; h: number;
  label?: string;
  conf?: number;
}

export interface Keypoint { x: number; y: number; vis: number; }

export interface Person { box: Box; keypoints: Keypoint[]; }

export interface MaskResult {
  rle: string;
  score: number;
  label?: string;
  box?: Box;
}

export interface OBBox {
  points: [number, number][]; // 4 normalized [x,y] corners in 0..1
  label: string;
  conf: number;
}

export interface ClsTop {
  label: string;
  conf: number;
}

export interface FaceMatch {
  bbox: [number, number, number, number];   // normalized x, y, w, h
  name: string;
  score: number;
  kind: 'high' | 'mid' | 'unknown';
}

export type InferenceResult =
  | { type: 'detect'; frame_id: number; ms: number; boxes: Box[]; faces?: FaceMatch[] }
  | { type: 'pose';   frame_id: number; ms: number; people: Person[]; faces?: FaceMatch[] }
  | { type: 'sam3';   frame_id: number; ms: number; masks: MaskResult[] }
  | { type: 'obb';    frame_id: number; ms: number; obboxes: OBBox[] }
  | { type: 'cls';    frame_id: number; ms: number; topk: ClsTop[] };
