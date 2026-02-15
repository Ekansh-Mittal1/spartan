import { Panel } from "./Panel";
import type { HudState } from "../types";

interface ImuPanelProps {
  state: HudState;
}

/** Format lat/lon for display (e.g. "37.7749° N, 122.4194° W"). */
function formatGps(lat: number, lon: number): string {
  const latStr = `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

export function ImuPanel({ state }: ImuPanelProps) {
  const anchor = state.anchor_heading_deg ?? 120;
  const heading = state.heading_deg;
  const arrowDeg = (anchor - heading + 360) % 360;
  const gpsStr = formatGps(state.gps_lat ?? 0, state.gps_lon ?? 0);

  return (
    <Panel className="w-[440px] py-5 px-7">
      <div className="flex flex-col items-center">
        <div
          className="flex items-center justify-center rounded-lg bg-amber-500/10 p-3 ring-1 ring-amber-500/30"
          style={{ transform: `rotate(${arrowDeg}deg)` }}
          title={`Anchor at ${anchor}°; you at ${Math.round(heading)}°`}
        >
          <svg
            className="h-20 w-20 shrink-0 text-amber-400"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 19V5M12 5l-6 8M12 5l6 8" />
          </svg>
        </div>
        <div className="mt-3 text-3xl font-bold tracking-tight text-neutral-100">{Math.round(heading)}°</div>
        <div className="mt-1 text-xs uppercase tracking-[0.2em] text-amber-500/90">gps coordinates</div>
        <div className="mt-1 max-w-full truncate text-center text-base text-neutral-300" title={gpsStr}>
          {gpsStr}
        </div>
      </div>
    </Panel>
  );
}
