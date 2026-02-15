/** Draw one synthetic frame: night cityscape (gate, lit buildings, dark sky) to match HUD reference. */
export function drawMockFrame(ctx: CanvasRenderingContext2D, width: number, height: number, t: number): void {
  const w = width;
  const h = height;

  // Dark blue sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#0a0e1a");
  sky.addColorStop(0.6, "#0d1525");
  sky.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Ground / wet street (darker, slight reflection band)
  const groundY = h * 0.72;
  ctx.fillStyle = "#0c0f14";
  ctx.fillRect(0, groundY, w, h - groundY);
  ctx.fillStyle = "rgba(40, 35, 25, 0.5)";
  ctx.fillRect(0, groundY, w, (h - groundY) * 0.4);

  // Central arch/gate shape (simplified Pailou - lit in gold/amber)
  const cx = w / 2;
  const gateW = Math.min(w, h) * 0.35;
  const gateH = h * 0.5;
  const gateTop = h * 0.2;
  ctx.fillStyle = "rgba(80, 60, 30, 0.9)";
  ctx.fillRect(cx - gateW / 2, gateTop, gateW, gateH);
  ctx.strokeStyle = "rgba(220, 180, 100, 0.9)";
  ctx.lineWidth = 3;
  ctx.strokeRect(cx - gateW / 2, gateTop, gateW, gateH);
  ctx.fillStyle = "rgba(200, 160, 80, 0.6)";
  ctx.fillRect(cx - gateW / 2 + 8, gateTop + 8, gateW - 16, 24);
  ctx.fillRect(cx - gateW / 2 + 8, gateTop + gateH - 32, gateW - 16, 24);

  // Buildings left (warm lit windows)
  for (let i = 0; i < 6; i++) {
    const bx = w * (0.05 + 0.12 * i + 0.02 * Math.sin(t + i));
    const by = groundY - 80 - 120 * (i % 2);
    const bw = 60 + 20 * Math.sin(t * 0.3 + i);
    const bh = 100 + 60 * (i % 2);
    ctx.fillStyle = "rgba(30, 25, 20, 0.95)";
    ctx.fillRect(bx, by, bw, bh);
    for (let wy = 0; wy < 4; wy++) {
      for (let wx = 0; wx < 2; wx++) {
        ctx.fillStyle = `rgba(255, 220, 180, ${0.6 + 0.2 * Math.sin(t + wx + wy)})`;
        ctx.fillRect(bx + 8 + wx * (bw / 2 - 6), by + 12 + wy * 22, 14, 16);
      }
    }
  }

  // Buildings right
  for (let i = 0; i < 6; i++) {
    const bx = w * (0.55 + 0.12 * i + 0.02 * Math.cos(t + i));
    const by = groundY - 80 - 120 * (i % 2);
    const bw = 60 + 20 * Math.cos(t * 0.3 + i);
    const bh = 100 + 60 * (i % 2);
    ctx.fillStyle = "rgba(30, 25, 20, 0.95)";
    ctx.fillRect(bx, by, bw, bh);
    for (let wy = 0; wy < 4; wy++) {
      for (let wx = 0; wx < 2; wx++) {
        ctx.fillStyle = `rgba(255, 200, 150, ${0.5 + 0.25 * Math.sin(t * 0.7 + wx + wy)})`;
        ctx.fillRect(bx + 8 + wx * (bw / 2 - 6), by + 12 + wy * 22, 14, 16);
      }
    }
  }

  // Subtle motion blur / figures (small dark ellipses)
  for (let i = 0; i < 4; i++) {
    const px = (w * (0.3 + 0.4 * (i / 4 + 0.05 * Math.sin(t * 2 + i)))) % w;
    const py = groundY - 20 - 30 * Math.sin(t + i);
    ctx.fillStyle = "rgba(20, 18, 15, 0.7)";
    ctx.beginPath();
    ctx.ellipse(px, py, 12, 24, 0, 0, Math.PI * 2);
    ctx.fill();
  }
}
