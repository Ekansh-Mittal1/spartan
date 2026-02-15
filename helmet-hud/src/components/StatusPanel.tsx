import { Panel } from "./Panel";
import type { HudState } from "../types";

interface StatusPanelProps {
  state: HudState;
}

function StatusRow({
  icon,
  label,
  active,
  activeLabel,
  inactiveLabel,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  activeLabel: string;
  inactiveLabel: string;
}) {
  return (
    <div className="flex items-center gap-4">
      <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${active ? "bg-amber-400/15" : "bg-neutral-800/60"}`}>
        <div className={active ? "text-amber-400" : "text-neutral-500"}>{icon}</div>
      </div>
      <div className="flex-1">
        <div className="text-xs uppercase tracking-widest text-neutral-500">{label}</div>
        <div className={`text-lg font-medium ${active ? "text-amber-300" : "text-neutral-400"}`}>
          {active ? activeLabel : inactiveLabel}
        </div>
      </div>
      <div className={`h-2.5 w-2.5 rounded-full ${active ? "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.6)]" : "bg-neutral-600"}`} />
    </div>
  );
}

function LightIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6M10 22h4M12 2v1M4.22 4.22l.71.71M1 12h1M4.22 19.78l.71-.71M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z" />
    </svg>
  );
}

function ModeIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  );
}

function PowerIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18.36 6.64a9 9 0 1 1-12.73 0M12 2v10" />
    </svg>
  );
}

export function StatusPanel({ state }: StatusPanelProps) {
  return (
    <Panel className="w-[520px] py-5 px-7">
      <div className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-200/90">System</div>
      <div className="my-3 border-t border-amber-500/30" />
      <div className="space-y-4">
        <StatusRow
          icon={<LightIcon />}
          label="Lights"
          active={state.light_on}
          activeLabel="ACTIVE"
          inactiveLabel="OFF"
        />
        <StatusRow
          icon={<ModeIcon />}
          label="Mode"
          active
          activeLabel={state.mode}
          inactiveLabel="—"
        />
        <StatusRow
          icon={<PowerIcon />}
          label="Device"
          active={state.mic_on}
          activeLabel="ONLINE"
          inactiveLabel="OFFLINE"
        />
      </div>
    </Panel>
  );
}
