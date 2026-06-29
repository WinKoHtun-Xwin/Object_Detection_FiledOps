// Binary packet builder — mirrors backend/app/protocol/frame.py.
//
// Layout: [u8 mode_id][u8 variant_id][u32 LE header_len][header_json][jpeg_bytes]

import type { ModeId, Box, FaceMatch } from '../types';

const MODE_TO_ID: Record<ModeId, number> = {
  yolo_detect: 0,
};

export interface PacketParts {
  mode: ModeId;
  variantId: number;
  header?: Record<string, unknown>;
  jpeg: Blob;
}

export async function buildPacket(parts: PacketParts): Promise<ArrayBuffer> {
  const headerStr = parts.header ? JSON.stringify(parts.header) : '';
  const headerBytes = new TextEncoder().encode(headerStr);
  const jpegBuf = await parts.jpeg.arrayBuffer();

  const out = new ArrayBuffer(6 + headerBytes.byteLength + jpegBuf.byteLength);
  const view = new DataView(out);
  view.setUint8(0, MODE_TO_ID[parts.mode]);
  view.setUint8(1, parts.variantId);
  view.setUint32(2, headerBytes.byteLength, true);

  const u8 = new Uint8Array(out);
  u8.set(headerBytes, 6);
  u8.set(new Uint8Array(jpegBuf), 6 + headerBytes.byteLength);
  return out;
}

export type ServerMessage =
  | { type: 'error'; frame_id: number; message: string }
  | { type: 'detect'; frame_id: number; ms: number; boxes: Box[]; faces?: FaceMatch[] };
