import { Panel } from "./Panel";
import type { HudState } from "../types";

const MAX_PREVIEW = 200;

function truncate(s: string, max: number): string {
  const t = (s ?? "").trim();
  if (!t) return "—";
  return t.length > max ? `${t.slice(0, max - 3)}...` : t;
}

function EyeIcon() {
  return (
    <svg className="mr-1.5 inline h-4 w-4 align-[-2px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function GlobeIcon() {
  return (
    <svg className="mr-1.5 inline h-4 w-4 align-[-2px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg className="mr-1.5 inline h-4 w-4 align-[-2px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7zM9 22h6M12 17v5" />
    </svg>
  );
}

interface ReasoningPanelProps {
  state: HudState;
}

export function ReasoningPanel({ state }: ReasoningPanelProps) {
  const vlm = (state.vlm_text ?? "").trim();
  const world = (state.world_context ?? "").trim();
  const fallback = (state.reasoning_text ?? "").trim();
  const showVlmWorld = vlm !== "" || world !== "";

  return (
    <Panel className="max-w-[600px] py-5 px-7">
      {showVlmWorld ? (
        <>
          <div className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-200/90">
            <EyeIcon />VLM
          </div>
          <div className="my-2 border-t border-amber-500/30" />
          <div className="text-lg leading-relaxed text-neutral-200 break-words">
            {vlm ? truncate(vlm, MAX_PREVIEW) : "—"}
          </div>
          <div className="mt-4 text-sm font-semibold uppercase tracking-[0.2em] text-amber-200/90">
            <GlobeIcon />World
          </div>
          <div className="my-2 border-t border-amber-500/30" />
          <div className="text-lg leading-relaxed text-neutral-200 break-words whitespace-pre-wrap">
            {world ? truncate(world, MAX_PREVIEW) : "—"}
          </div>
        </>
      ) : (
        <>
          <div className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-200/90">
            <BrainIcon />Reasoning
          </div>
          <div className="my-2 border-t border-amber-500/30" />
          <div className="text-lg leading-relaxed text-neutral-200 break-words">
            {truncate(fallback, 80)}
          </div>
        </>
      )}
    </Panel>
  );
}
