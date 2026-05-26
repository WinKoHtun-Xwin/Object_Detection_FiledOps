import { create } from 'zustand';
import type { InferenceResult, ModeId, Sam3PromptKind, YoloSize } from '../types';

interface AppState {
  mode: ModeId;
  yoloSize: YoloSize;
  yoloConf: number;
  sam3Prompt: Sam3PromptKind;
  sam3Text: string;
  tracking: boolean;
  mirror: boolean;
  paused: boolean;
  recognizeFaces: boolean;

  selectedDeviceId: string | null;
  cameraId: string;

  fps: number;
  inferenceMs: number;
  lastResult: InferenceResult | null;

  setMode: (m: ModeId) => void;
  setYoloSize: (s: YoloSize) => void;
  setYoloConf: (c: number) => void;
  setSam3Prompt: (p: Sam3PromptKind) => void;
  setSam3Text: (t: string) => void;
  setTracking: (b: boolean) => void;
  setMirror: (b: boolean) => void;
  setPaused: (b: boolean) => void;
  setRecognizeFaces: (b: boolean) => void;

  setSelectedDeviceId: (id: string | null) => void;
  setCameraId: (id: string) => void;

  pushResult: (r: InferenceResult) => void;
  setFps: (n: number) => void;
}

export const useAppState = create<AppState>((set) => ({
  mode: 'yolo_detect',
  yoloSize: 'n',
  yoloConf: 0.25,
  sam3Prompt: 'text',
  sam3Text: 'person',
  tracking: false,
  mirror: true,
  paused: false,
  recognizeFaces: false,

  selectedDeviceId: null,
  cameraId: 'cam-default',

  fps: 0,
  inferenceMs: 0,
  lastResult: null,

  setMode: (mode) => set({ mode }),
  setYoloSize: (yoloSize) => set({ yoloSize }),
  setYoloConf: (yoloConf) => set({ yoloConf }),
  setSam3Prompt: (sam3Prompt) => set({ sam3Prompt }),
  setSam3Text: (sam3Text) => set({ sam3Text }),
  setTracking: (tracking) => set({ tracking }),
  setMirror: (mirror) => set({ mirror }),
  setPaused: (paused) => set({ paused }),
  setRecognizeFaces: (recognizeFaces) => set({ recognizeFaces }),

  setSelectedDeviceId: (selectedDeviceId) => set({ selectedDeviceId }),
  setCameraId: (cameraId) => set({ cameraId }),

  pushResult: (r) => set({ lastResult: r, inferenceMs: r.ms }),
  setFps: (fps) => set({ fps }),
}));
