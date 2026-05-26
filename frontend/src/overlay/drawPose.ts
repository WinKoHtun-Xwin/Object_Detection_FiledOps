import type { Person } from '../types';

// COCO 17-keypoint skeleton edges (pairs of keypoint indices).
const EDGES: ReadonlyArray<[number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],            // head
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],   // arms
  [5, 11], [6, 12], [11, 12],                // torso
  [11, 13], [13, 15], [12, 14], [14, 16],    // legs
];

const KEYPOINT_RADIUS = 3;
const VIS_THRESHOLD = 0.3;

export function drawPose(ctx: CanvasRenderingContext2D, people: Person[]): void {
  const { width: W, height: H } = ctx.canvas;
  ctx.clearRect(0, 0, W, H);

  for (const person of people) {
    const kps = person.keypoints;

    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    for (const [a, b] of EDGES) {
      const ka = kps[a];
      const kb = kps[b];
      if (!ka || !kb || ka.vis < VIS_THRESHOLD || kb.vis < VIS_THRESHOLD) continue;
      ctx.beginPath();
      ctx.moveTo(ka.x * W, ka.y * H);
      ctx.lineTo(kb.x * W, kb.y * H);
      ctx.stroke();
    }

    ctx.fillStyle = '#facc15';
    for (const k of kps) {
      if (k.vis < VIS_THRESHOLD) continue;
      ctx.beginPath();
      ctx.arc(k.x * W, k.y * H, KEYPOINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
