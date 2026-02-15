import { useCallback, useEffect, useRef, useState } from "react";
import type { HudState } from "../types";

const DEFAULT_STATE: HudState = {
  bpm: 0,
  temp_c: 0,
  quality: 1,
  heading_deg: 0,
  mic_level: 0,
  drawer_open: false,
  mic_on: true,
  light_on: false,
  thermal_on: false,
  alert_banner: null,
  mode: "NORM",
  reasoning_text: "",
  vlm_text: "",
  world_context: "",
  gps_lat: 0,
  gps_lon: 0,
  anchor_heading_deg: 0,
};

const MAX_BACKOFF_MS = 10000;
const INITIAL_RECONNECT_MS = 500;

function num(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function parseState(data: string): HudState {
  const raw = JSON.parse(data) as Record<string, unknown>;
  return {
    bpm: num(raw.bpm, DEFAULT_STATE.bpm),
    temp_c: num(raw.temp_c, DEFAULT_STATE.temp_c),
    quality: num(raw.quality, DEFAULT_STATE.quality),
    heading_deg: num(raw.heading_deg, DEFAULT_STATE.heading_deg),
    mic_level: num(raw.mic_level, DEFAULT_STATE.mic_level),
    drawer_open: Boolean(raw.drawer_open),
    mic_on: Boolean(raw.mic_on),
    light_on: Boolean(raw.light_on),
    thermal_on: Boolean(raw.thermal_on),
    alert_banner: raw.alert_banner != null ? String(raw.alert_banner) : null,
    mode: String(raw.mode ?? DEFAULT_STATE.mode),
    reasoning_text: String(raw.reasoning_text ?? DEFAULT_STATE.reasoning_text),
    vlm_text: String(raw.vlm_text ?? DEFAULT_STATE.vlm_text),
    world_context: String(raw.world_context ?? DEFAULT_STATE.world_context),
    gps_lat: num(raw.gps_lat, DEFAULT_STATE.gps_lat),
    gps_lon: num(raw.gps_lon, DEFAULT_STATE.gps_lon),
    anchor_heading_deg: num(raw.anchor_heading_deg, DEFAULT_STATE.anchor_heading_deg),
  };
}

/** WebSocket hook for HUD state. Use when VITE_WS_URL is set (external backend). */
export function useHudState(wsUrl: string): {
  state: HudState;
  connected: boolean;
  /** Send a frame (base64 JPEG) to the backend for VLM. No-op if not connected. */
  sendFrame: (base64: string) => void;
} {
  const [state, setState] = useState<HudState>(DEFAULT_STATE);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const sendFrame = useCallback((base64: string) => {
    const w = wsRef.current;
    if (w?.readyState === 1) w.send(JSON.stringify({ type: "frame", data: base64 }));
  }, []);

  useEffect(() => {
    if (!wsUrl) return;
    let ws: WebSocket | null = null;
    let backoff = INITIAL_RECONNECT_MS;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      try {
        ws = new WebSocket(wsUrl);
        wsRef.current = ws;
      } catch {
        scheduleReconnect();
        return;
      }
      ws.onopen = () => {
        setConnected(true);
        backoff = INITIAL_RECONNECT_MS;
      };
      ws.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        try {
          setState(parseState(event.data));
        } catch {
          // ignore
        }
      };
      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        ws = null;
        scheduleReconnect();
      };
      ws.onerror = () => {
        ws?.close();
      };
    }

    function scheduleReconnect() {
      timeoutId = setTimeout(() => {
        timeoutId = null;
        connect();
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      }, backoff);
    }
    connect();
    return () => {
      if (timeoutId != null) clearTimeout(timeoutId);
      wsRef.current = null;
      ws?.close();
    };
  }, [wsUrl]);

  return { state, connected, sendFrame };
}
