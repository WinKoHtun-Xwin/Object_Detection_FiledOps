import type { ClsTop } from '../types';

// Classification has no spatial boxes — render a top-K leaderboard
// in the upper-left of the canvas.
export function drawClassification(ctx: CanvasRenderingContext2D, topk: ClsTop[]): void {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  if (topk.length === 0) return;

  const lineH = Math.round(H * 0.04);
  const padX = 14;
  const padY = 12;
  ctx.font = `${Math.round(lineH * 0.7)}px ui-monospace, monospace`;
  ctx.textBaseline = 'top';

  const maxLabelW = Math.max(...topk.map((t) => ctx.measureText(t.label).width));
  const boxW = Math.min(W * 0.45, maxLabelW + 160);
  const boxH = topk.length * lineH + padY * 2;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(padX, padY, boxW, boxH);

  topk.forEach((t, i) => {
    const y = padY + i * lineH + padY / 2;
    const labelX = padX + 12;
    const barX = labelX + maxLabelW + 16;
    const barMax = boxW - (barX - padX) - 12;
    const barW = Math.max(2, t.conf * barMax);

    ctx.fillStyle = i === 0 ? '#facc15' : '#cbd5f5';
    ctx.fillText(t.label, labelX, y);

    ctx.fillStyle = i === 0 ? 'rgba(250,204,21,0.85)' : 'rgba(96,165,250,0.7)';
    ctx.fillRect(barX, y + 2, barW, lineH * 0.55);

    ctx.fillStyle = '#fff';
    ctx.fillText(`${(t.conf * 100).toFixed(1)}%`, barX + barMax + 4, y);
  });
}
