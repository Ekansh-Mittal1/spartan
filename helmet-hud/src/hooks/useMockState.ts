import { useCallback, useEffect, useRef, useState } from "react";
import type { HudState } from "../types";

const TICK_MS = 1000 / 30;

const INITIAL: HudState = {
  bpm: 72,
  temp_c: 36.6,
  quality: 0.95,
  heading_deg: 0,
  mic_level: 0.3,
  drawer_open: false,
  mic_on: true,
  light_on: false,
  thermal_on: false,
  alert_banner: null,
  mode: "NORM",
  reasoning_text: "—",
  vlm_text: "",
  world_context: "",
  gps_lat: 37.7749,
  gps_lon: -122.4194,
  anchor_heading_deg: 120,
};

export function useMockState(): { state: HudState; connected: true } {
  const [state, setState] = useState<HudState>(INITIAL);
  const startRef = useRef(performance.now() / 1000);
  const stressRef = useRef(0);
  const targetStressRef = useRef(0);
  const alertUntilRef = useRef(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  const triggerKey = useCallback((key: string) => {
    const now = performance.now() / 1000;
    setState((prev) => {
      const next = { ...prev };
      if (key === "r" || key === "R") next.drawer_open = !prev.drawer_open;
      else if (key === "h" || key === "H") targetStressRef.current = targetStressRef.current > 0.5 ? 0 : 1;
      else if (key === "a" || key === "A") {
        next.alert_banner = "HIGH HEAT";
        alertUntilRef.current = now + 5;
      } else if (key === "m" || key === "M") next.mic_on = !prev.mic_on;
      else if (key === "l" || key === "L") next.light_on = !prev.light_on;
      else if (key === "t" || key === "T") next.thermal_on = !prev.thermal_on;
      return next;
    });
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      const t = performance.now() / 1000;
      const elapsed = t - startRef.current;
      stressRef.current += 0.02 * (targetStressRef.current - stressRef.current);
      const bpm = 72 + stressRef.current * 45 + 2 * Math.sin(elapsed * 0.5);
      const temp = Math.max(35.5, Math.min(37.5, 36.5 + 0.4 * Math.sin(elapsed * 0.15)));
      const heading = (elapsed * 12) % 360;
      const micLevel = Math.max(0, Math.min(1, 0.2 + 0.3 * (0.5 + 0.5 * Math.sin(elapsed * 2))));
      const gpsLat = 37.7749 + 0.0001 * Math.sin(elapsed * 0.1);
      const gpsLon = -122.4194 + 0.0001 * Math.cos(elapsed * 0.12);
      setState((prev) => {
        let alert = prev.alert_banner;
        if (alert != null && t >= alertUntilRef.current) alert = null;
        return {
          ...prev,
          bpm,
          temp_c: temp,
          heading_deg: heading,
          mic_level: micLevel,
          quality: 0.95,
          alert_banner: alert,
          gps_lat: gpsLat,
          gps_lon: gpsLon,
        };
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.length === 1) triggerKey(e.key);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [triggerKey]);

  return { state, connected: true as const };
}
