/**
 * Precomputed warp mesh for lens correction (Brown–Conrady radial model).
 * Center of distortion is the viewport center. For each output (screen) pixel we
 * sample from source at (map_x, map_y) so that after the physical lens the image looks rectilinear.
 * Forward model: source = screen * (1 + k1*r² + k2*r⁴); r is normalized distance from viewport center.
 */

export interface WarpMesh {
  /** Vertex positions (x, y) in viewport pixel space, interleaved. Length = (cols+1)*(rows+1)*2 */
  positions: Float32Array;
  /** Texture coords (u, v) in [0,1] for sampling source. Length = (cols+1)*(rows+1)*2 */
  uvs: Float32Array;
  /** Triangle indices. Length = cols*rows*6 (two triangles per cell) */
  indices: Uint16Array;
  cols: number;
  rows: number;
}

/**
 * Build a warp mesh for the given viewport size and radial coefficients.
 * Mesh is a grid; each vertex gets (position, uv) from the radial map.
 * Recompute when width, height, k1, or k2 change (no per-pixel math in render loop).
 */
export function buildWarpMesh(
  width: number,
  height: number,
  k1: number,
  k2: number,
  gridCols: number = 64,
  gridRows: number = 64
): WarpMesh {
  const cols = Math.max(2, gridCols);
  const rows = Math.max(2, gridRows);
  const cx = width / 2;
  const cy = height / 2;
  const normX = 2 / width;  // so that at x=0 -> -1, x=width -> 1
  const normY = 2 / height;

  const numVerts = (cols + 1) * (rows + 1);
  const positions = new Float32Array(numVerts * 2);
  const uvs = new Float32Array(numVerts * 2);

  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const idx = (j * (cols + 1) + i) * 2;
      const px = (i / cols) * width;
      const py = (j / rows) * height;
      positions[idx] = px;
      positions[idx + 1] = py;

      const nx = (px - cx) * normX;
      const ny = (py - cy) * normY;
      const r2 = nx * nx + ny * ny;
      const r4 = r2 * r2;
      const factor = 1 + k1 * r2 + k2 * r4;
      const sx = cx + (px - cx) * factor;
      const sy = cy + (py - cy) * factor;
      const u = sx / width;
      const v = sy / height;
      uvs[idx] = u;
      uvs[idx + 1] = v;
    }
  }

  const numCells = cols * rows;
  const indices = new Uint16Array(numCells * 6);
  let triIdx = 0;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * (cols + 1) + i;
      const b = a + 1;
      const c = a + (cols + 1);
      const d = c + 1;
      indices[triIdx++] = a;
      indices[triIdx++] = c;
      indices[triIdx++] = b;
      indices[triIdx++] = b;
      indices[triIdx++] = c;
      indices[triIdx++] = d;
    }
  }

  return { positions, uvs, indices, cols, rows };
}
