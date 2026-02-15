/** HUD overlay state (mock or from WebSocket). */
export interface HudState {
  bpm: number;
  temp_c: number;
  quality: number;
  heading_deg: number;
  mic_level: number;
  drawer_open: boolean;
  mic_on: boolean;
  light_on: boolean;
  thermal_on: boolean;
  alert_banner: string | null;
  mode: string;
  reasoning_text: string;
  /** Latest VLM (vision) inference summary from backend. */
  vlm_text: string;
  /** World-context / people-tracker output from backend. */
  world_context: string;
  /** Mock GPS latitude (e.g. 37.7749). */
  gps_lat: number;
  /** Mock GPS longitude (e.g. -122.4194). */
  gps_lon: number;
  /** Fixed anchor bearing (deg from north) for IMU arrow; arrow points toward this. */
  anchor_heading_deg: number;
  /** Currently active camera ID (the one being sent to VLM). Empty if none. */
  active_camera_id: string;
  /** All known camera IDs that have sent at least one frame. */
  camera_ids: string[];
}

/** Per-eye rect from display config. */
export interface LayoutRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Display layout; embedded at build time. */
export interface DisplayLayout {
  panel_width: number;
  panel_height: number;
  left_eye: LayoutRect;
  right_eye: LayoutRect;
  center_gap_px: number;
  video_margin_px: number;
  eye_offset: { dx: number; dy: number };
  safe_margin: { mx: number; my: number };
}

/** Display: 2560 width × 1440 height; two streams side-by-side. */
export const DEFAULT_LAYOUT: DisplayLayout = {
  panel_width: 2560,
  panel_height: 1440,
  left_eye: { x: 0, y: 0, w: 1280, h: 1440 },
  right_eye: { x: 1280, y: 0, w: 1280, h: 1440 },
  center_gap_px: 24,
  video_margin_px: 16,
  eye_offset: { dx: 0, dy: 0 },
  safe_margin: { mx: 48, my: 64 },
};
