import type { DisplayLayout, LayoutRect } from "./types";

/** Left viewport rect; width reduced so center_gap_px remains between left and right. */
export function leftViewport(layout: DisplayLayout): LayoutRect {
  const half = Math.floor(layout.center_gap_px / 2);
  const le = layout.left_eye;
  const dx = layout.eye_offset.dx;
  const dy = layout.eye_offset.dy;
  return {
    x: le.x + dx,
    y: le.y + dy,
    w: Math.max(0, le.w - half),
    h: le.h,
  };
}

/** Right viewport rect; shifted right and narrowed for center gap. */
export function rightViewport(layout: DisplayLayout): LayoutRect {
  const half = Math.floor(layout.center_gap_px / 2);
  const re = layout.right_eye;
  const dx = layout.eye_offset.dx;
  const dy = layout.eye_offset.dy;
  return {
    x: re.x + dx + half,
    y: re.y + dy,
    w: Math.max(0, re.w - half),
    h: re.h,
  };
}
