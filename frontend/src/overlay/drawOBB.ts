import type { OBBox } from '../types';

const COLORS = ['#22d3ee', '#f472b6', '#a3e635', '#fb923c', '#facc15', '#60a5fa', '#f87171', '#34d399'];

function colorFor(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
}

export function drawOBB(ctx: CanvasRenderingContext2D, obboxes: OBBox[]): void {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 2;
  ctx.font = '13px ui-monospace, monospace';
  ctx.textBaseline = 'top';

  for (const o of obboxes) {
    if (o.points.length < 4) continue;
    const color = colorFor(o.label);
    ctx.strokeStyle = color;
    ctx.beginPath();
    o.points.forEach((p, i) => {
      const x = p[0] * W;
      const y = p[1] * H;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    const [px, py] = o.points[0];
    const tag = `${o.label} ${o.conf.toFixed(2)}`;
    const padX = 4;
    const tw = ctx.measureText(tag).width + padX * 2;
    const th = 17;
    const x = px * W;
    const y = Math.max(0, py * H - th);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, tw, th);
    ctx.fillStyle = '#000';
    ctx.fillText(tag, x + padX, y + 2);
  }
}
