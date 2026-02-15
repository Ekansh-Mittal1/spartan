import { useEffect, useRef } from "react";
import { drawMockFrame } from "../mock/drawFrame";

interface MockVideoCanvasProps {
  width: number;
  height: number;
  className?: string;
}

export function MockVideoCanvas({ width, height, className = "" }: MockVideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  const startRef = useRef(performance.now() / 1000);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const ctx2: CanvasRenderingContext2D = ctx;

    let cancelled = false;
    function loop() {
      if (cancelled) return;
      frameRef.current = requestAnimationFrame(loop);
      const t = performance.now() / 1000 - startRef.current;
      drawMockFrame(ctx2, width, height, t);
    }
    loop();
    return () => {
      cancelled = true;
      cancelAnimationFrame(frameRef.current);
    };
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={{ display: "block", width: "100%", height: "100%", objectFit: "contain" }}
    />
  );
}
