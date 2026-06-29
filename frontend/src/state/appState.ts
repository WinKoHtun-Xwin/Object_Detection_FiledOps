import { create } from 'zustand';
import type { InferenceResult, YoloSize } from '../types';

interface AppState {
  yoloSize: YoloSize;
  yoloConf: number;
  tracking: boolean;
  mirror: boolean;
  paused: boolean;
  recognizeFaces: boolean;

  selectedDeviceId: string | null;
  cameraId: string;

  fps: number;
  inferenceMs: number;
  lastResult: InferenceResult | null;

  setYoloSize: (s: YoloSize) => void;
  setYoloConf: (c: number) => void;
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
  yoloSize: 'n',
  yoloConf: 0.25,
  tracking: false,
  mirror: false,
  paused: false,
  recognizeFaces: false,

  selectedDeviceId: null,
  cameraId: 'cam-default',

  fps: 0,
  inferenceMs: 0,
  lastResult: null,

  setYoloSize: (yoloSize) => set({ yoloSize }),
  setYoloConf: (yoloConf) => set({ yoloConf }),
  setTracking: (tracking) => set({ tracking }),
  setMirror: (mirror) => set({ mirror }),
  setPaused: (paused) => set({ paused }),
  setRecognizeFaces: (recognizeFaces) => set({ recognizeFaces }),

  setSelectedDeviceId: (selectedDeviceId) => set({ selectedDeviceId }),
  setCameraId: (cameraId) => set({ cameraId }),

  pushResult: (r) => set({ lastResult: r, inferenceMs: r.ms }),
  setFps: (fps) => set({ fps }),
}));
