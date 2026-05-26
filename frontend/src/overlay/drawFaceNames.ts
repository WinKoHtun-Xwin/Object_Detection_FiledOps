import type { FaceMatch } from '../types';

const COLORS: Record<FaceMatch['kind'], string> = {
  high: '#22c55e',
  mid: '#eab308',
  unknown: '#6b7280',
};

export function drawFaceNames(ctx: CanvasRenderingContext2D, faces: FaceMatch[]): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.save();
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  for (const f of faces) {
    const [nx, ny, nw, nh] = f.bbox;
    const x = nx * w;
    const y = ny * h;
    const bw = nw * w;
    const bh = nh * h;

    ctx.strokeStyle = COLORS[f.kind];
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);

    const label = f.kind === 'unknown' ? 'Unknown' : `${f.name} ${f.score.toFixed(2)}`;
    const padding = 4;
    const textW = ctx.measureText(label).width + padding * 2;
    const textH = 18;
    const lx = x;
    const ly = Math.max(0, y - textH);
    ctx.fillStyle = COLORS[f.kind];
    ctx.fillRect(lx, ly, textW, textH);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, lx + padding, ly + 2);
  }
  ctx.restore();
}
