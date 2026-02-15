import { type ReactNode } from "react";

const PANEL_CLASS =
  "rounded-xl border-2 border-amber-400/80 bg-neutral-950/80 shadow-lg shadow-amber-500/20 backdrop-blur-sm";

interface PanelProps {
  children: ReactNode;
  className?: string;
}

export function Panel({ children, className = "" }: PanelProps) {
  return <div className={`${PANEL_CLASS} ${className}`.trim()}>{children}</div>;
}
