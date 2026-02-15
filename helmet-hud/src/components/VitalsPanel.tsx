import { Panel } from "./Panel";
import type { HudState } from "../types";

function HeartIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
    </svg>
  );
}

function ThermometerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
    </svg>
  );
}

function SignalIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M2 20h.01M7 20v-4M12 20v-8M17 20V8M22 20V4" />
    </svg>
  );
}

interface VitalsPanelProps {
  state: HudState;
}

export function VitalsPanel({ state }: VitalsPanelProps) {
  const quality = Math.max(0, Math.min(1, state.quality));
  return (
    <Panel className="w-[520px] p-7">
      <div className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-200/90">Vitals</div>
      <div className="my-3 border-t border-amber-500/30" />
      <div className="flex items-end gap-6">
        <div className="flex items-center gap-3">
          <HeartIcon className="h-9 w-9 shrink-0 text-amber-400/90" />
          <div>
            <div className="text-xs uppercase tracking-widest text-neutral-500">Heart Rate</div>
            <div className="flex items-baseline gap-2">
              <span className="text-5xl font-bold tracking-tight text-amber-400">{Math.round(state.bpm)}</span>
              <span className="text-base text-neutral-400">bpm</span>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <ThermometerIcon className="h-8 w-8 shrink-0 text-amber-400/90" />
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500">Temp</div>
          <span className="text-2xl font-medium text-amber-400">{state.temp_c.toFixed(1)}<span className="text-lg text-neutral-400"> °C</span></span>
        </div>
      </div>
      <div className="mt-5 flex items-center gap-3">
        <SignalIcon className="h-7 w-7 shrink-0 text-amber-400/80" />
        <div className="flex-1">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs uppercase tracking-widest text-neutral-500">Signal</span>
            <span className="text-sm font-medium text-amber-300">{Math.round(quality * 100)}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-neutral-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all"
              style={{ width: `${quality * 100}%` }}
            />
          </div>
        </div>
      </div>
    </Panel>
  );
}
