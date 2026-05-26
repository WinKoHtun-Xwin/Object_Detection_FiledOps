import type { Box } from '../types';

const COLORS = [
  '#22d3ee', '#f472b6', '#a3e635', '#fb923c', '#facc15',
  '#60a5fa', '#f87171', '#34d399', '#c084fc', '#fde68a',
];

function colorFor(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
}

export function drawBoxes(ctx: CanvasRenderingContext2D, boxes: Box[], mirror: boolean): void {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  if (mirror) {
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
  }
  ctx.lineWidth = 2;
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';

  for (const b of boxes) {
    const x = b.x * W;
    const y = b.y * H;
    const w = b.w * W;
    const h = b.h * H;
    const color = colorFor(b.label ?? '');
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const tag = `${b.label} ${(b.conf ?? 0).toFixed(2)}`;
    const padX = 4;
    const tw = ctx.measureText(tag).width + padX * 2;
    const th = 18;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = '#000';
    // Un-mirror the label so the text reads correctly even when mirrored.
    if (mirror) {
      ctx.save();
      ctx.translate(x + tw, Math.max(0, y - th));
      ctx.scale(-1, 1);
      ctx.fillText(tag, padX, 2);
      ctx.restore();
    } else {
      ctx.fillText(tag, x + padX, Math.max(0, y - th) + 2);
    }
  }
  ctx.restore();
}
