import { type RefObject, useEffect, useRef } from "react";
import type { WarpMesh } from "../lib/warpMesh";

const VS = `
  attribute vec2 a_position;
  attribute vec2 a_uv;
  uniform vec4 u_scaleBias;
  varying vec2 v_uv;
  void main() {
    v_uv = a_uv;
    float x = a_position.x * u_scaleBias.x + u_scaleBias.z;
    float y = a_position.y * u_scaleBias.y + u_scaleBias.w;
    gl_Position = vec4(x, y, 0.0, 1.0);
  }
`;

const FS = `
  precision mediump float;
  varying vec2 v_uv;
  uniform sampler2D u_tex;
  uniform float u_zoom;
  uniform float u_uvOffsetX;
  uniform vec4 u_texRect;
  void main() {
    vec2 uv = 0.5 + (v_uv - 0.5) / u_zoom;
    uv.x += u_uvOffsetX;
    uv = u_texRect.xy + uv * u_texRect.zw;
    gl_FragColor = texture2D(u_tex, uv);
  }
`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vsSource: string, fsSource: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

export interface WarpLayerProps {
  width: number;
  height: number;
  mesh: WarpMesh;
  /** Ref to video or image element (used each frame so no re-render when ref populates). */
  sourceRef: RefObject<HTMLVideoElement | HTMLImageElement | null>;
  /** Digital zoom: 1 = no zoom, >1 = zoom in (crop around center). */
  zoom?: number;
  /** Horizontal UV offset for stereo separation (left eye negative, right eye positive). */
  uvOffsetX?: number;
  className?: string;
}

/**
 * Renders the source (video/image) through a precomputed warp mesh using WebGL.
 * No per-pixel math in the loop—only mesh draw and texture sample.
 */
export function WarpLayer({
  width,
  height,
  mesh,
  sourceRef,
  zoom = 1,
  uvOffsetX = 0,
  className = "",
}: WarpLayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufRef = useRef<{ pos: WebGLBuffer; uv: WebGLBuffer; idx: WebGLBuffer } | null>(null);
  const texRef = useRef<WebGLTexture | null>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;

    const gl = canvas.getContext("webgl", { alpha: false, premultipliedAlpha: false });
    if (!gl) return;
    glRef.current = gl;

    const program = createProgram(gl, VS, FS);
    if (!program) return;
    programRef.current = program;

    const posBuf = gl.createBuffer();
    const uvBuf = gl.createBuffer();
    const idxBuf = gl.createBuffer();
    if (!posBuf || !uvBuf || !idxBuf) return;
    bufRef.current = { pos: posBuf, uv: uvBuf, idx: idxBuf };

    const tex = gl.createTexture();
    if (!tex) return;
    texRef.current = tex;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return () => {
      cancelAnimationFrame(rafRef.current);
      gl.deleteProgram(program);
      gl.deleteBuffer(posBuf);
      gl.deleteBuffer(uvBuf);
      gl.deleteBuffer(idxBuf);
      gl.deleteTexture(tex);
      glRef.current = null;
      programRef.current = null;
      bufRef.current = null;
      texRef.current = null;
    };
  }, [width, height]);

  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const buffers = bufRef.current;
    if (!gl || !program || !buffers || mesh.positions.length === 0) return;

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.uv);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
  }, [mesh]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    const program = programRef.current;
    const buffers = bufRef.current;
    const tex = texRef.current;
    if (!canvas || !gl || !program || !buffers || !tex || width <= 0 || height <= 0) return;

    let cancelled = false;

    function draw() {
      if (cancelled) return;
      const source = sourceRef.current;
      if (!source) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      const g = gl;
      const prog = program;
      const bufs = buffers;
      const texture = tex;
      if (!g || !prog || !bufs || !texture) return;

      g.viewport(0, 0, width, height);
      g.clearColor(0, 0, 0.02, 1);
      g.clear(g.COLOR_BUFFER_BIT);

      g.useProgram(prog);

      const posLoc = g.getAttribLocation(prog, "a_position");
      const uvLoc = g.getAttribLocation(prog, "a_uv");
      const texLoc = g.getUniformLocation(prog, "u_tex");

      g.activeTexture(g.TEXTURE0);
      g.bindTexture(g.TEXTURE_2D, texture);
      g.texImage2D(g.TEXTURE_2D, 0, g.RGBA, g.RGBA, g.UNSIGNED_BYTE, source);

      if (texLoc) g.uniform1i(texLoc, 0);

      const tw = (source as HTMLVideoElement).videoWidth ?? (source as HTMLImageElement).naturalWidth ?? width;
      const th = (source as HTMLVideoElement).videoHeight ?? (source as HTMLImageElement).naturalHeight ?? height;
      const viewAspect = width / height;
      const texAspect = tw / th;
      let uMin: number, vMin: number, uSize: number, vSize: number;
      if (viewAspect >= texAspect) {
        uMin = 0;
        uSize = 1;
        vSize = texAspect / viewAspect;
        vMin = (1 - vSize) / 2;
      } else {
        vMin = 0;
        vSize = 1;
        uSize = viewAspect / texAspect;
        uMin = (1 - uSize) / 2;
      }
      const texRectLoc = g.getUniformLocation(prog, "u_texRect");
      if (texRectLoc) g.uniform4f(texRectLoc, uMin, vMin, uSize, vSize);

      g.bindBuffer(g.ARRAY_BUFFER, bufs.pos);
      g.enableVertexAttribArray(posLoc);
      g.vertexAttribPointer(posLoc, 2, g.FLOAT, false, 0, 0);

      g.bindBuffer(g.ARRAY_BUFFER, bufs.uv);
      g.enableVertexAttribArray(uvLoc);
      g.vertexAttribPointer(uvLoc, 2, g.FLOAT, false, 0, 0);

      g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, bufs.idx);

      const scaleBiasLoc = g.getUniformLocation(prog, "u_scaleBias");
      if (scaleBiasLoc) {
        g.uniform4f(scaleBiasLoc, 2 / width, -2 / height, -1, 1);
      }
      const zoomLoc = g.getUniformLocation(prog, "u_zoom");
      if (zoomLoc) g.uniform1f(zoomLoc, zoom);
      const uvOffsetLoc = g.getUniformLocation(prog, "u_uvOffsetX");
      if (uvOffsetLoc) g.uniform1f(uvOffsetLoc, uvOffsetX);

      g.drawElements(g.TRIANGLES, mesh.indices.length, g.UNSIGNED_SHORT, 0);
    }

    function loop() {
      if (cancelled) return;
      draw();
      rafRef.current = requestAnimationFrame(loop);
    }

    loop();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [sourceRef, width, height, mesh, zoom, uvOffsetX]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={{ display: "block", width: "100%", height: "100%", objectFit: "fill" }}
      aria-hidden
    />
  );
}
