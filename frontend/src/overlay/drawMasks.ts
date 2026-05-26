import type { MaskResult } from '../types';

// Cache decoded mask bitmaps by their data URL.
const bitmapCache = new Map<string, ImageBitmap>();
// In-flight decodes so we don't kick off duplicates.
const pending = new Map<string, Promise<ImageBitmap>>();

async function ensureBitmap(src: string): Promise<ImageBitmap> {
  const cached = bitmapCache.get(src);
  if (cached) return cached;
  const inflight = pending.get(src);
  if (inflight !== undefined) return inflight;

  const p = (async () => {
    const res = await fetch(src);
    const blob = await res.blob();
    const bmp = await createImageBitmap(blob);
    bitmapCache.set(src, bmp);
    pending.delete(src);
    return bmp;
  })();
  pending.set(src, p);
  return p;
}

function pruneCache(): void {
  if (bitmapCache.size <= 64) return;
  const drop = bitmapCache.size - 32;
  let i = 0;
  for (const [k, bmp] of bitmapCache) {
    if (i++ >= drop) break;
    bmp.close?.();
    bitmapCache.delete(k);
  }
}

interface DrawOptions {
  outlineWidth?: number;
  outlineColor?: string;
  glow?: boolean;
}

// Ensures bitmaps for the given masks are decoded. Returns true when ALL bitmaps
// are ready (caller can draw synchronously). When false is returned, the caller
// has typically already drawn whatever was ready and should redraw once the
// returned promise resolves.
export async function preloadMasks(masks: MaskResult[]): Promise<void> {
  await Promise.all(masks.map((m) => ensureBitmap(m.rle)));
}

// Synchronous draw. Skips any masks whose bitmaps aren't decoded yet.
export function drawMasks(
  ctx: CanvasRenderingContext2D,
  masks: MaskResult[],
  opts: DrawOptions = {},
): void {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  if (masks.length === 0) return;

  pruneCache();

  const outlineWidth = opts.outlineWidth ?? 3;
  const outlineColor = opts.outlineColor ?? '#ffffff';
  const glow = opts.glow ?? true;

  for (const m of masks) {
    const bmp = bitmapCache.get(m.rle);
    if (!bmp) {
      // Kick off decode so the next call finds it ready.
      void ensureBitmap(m.rle);
      continue;
    }

    const mw = bmp.width;
    const mh = bmp.height;
    const ring = document.createElement('canvas');
    ring.width = mw;
    ring.height = mh;
    const rctx = ring.getContext('2d');
    if (!rctx) continue;

    const n = outlineWidth;
    const offsets: [number, number][] = [
      [-n, 0], [n, 0], [0, -n], [0, n],
      [-n, -n], [n, -n], [-n, n], [n, n],
    ];
    for (const [dx, dy] of offsets) {
      rctx.drawImage(bmp, dx, dy);
    }
    rctx.globalCompositeOperation = 'destination-out';
    rctx.drawImage(bmp, 0, 0);

    rctx.globalCompositeOperation = 'source-in';
    rctx.fillStyle = outlineColor;
    rctx.fillRect(0, 0, mw, mh);

    if (glow) {
      ctx.save();
      ctx.shadowColor = outlineColor;
      ctx.shadowBlur = 12;
      ctx.drawImage(ring, 0, 0, W, H);
      ctx.restore();
    } else {
      ctx.drawImage(ring, 0, 0, W, H);
    }
  }
}
