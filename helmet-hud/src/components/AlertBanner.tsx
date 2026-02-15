interface AlertBannerProps {
  message: string;
}

export function AlertBanner({ message }: AlertBannerProps) {
  return (
    <div className="absolute left-6 right-6 top-1/2 -translate-y-1/2 rounded-lg border-2 border-amber-500/80 bg-red-950/90 px-4 py-3 text-center text-lg font-bold text-red-300 shadow-lg shadow-amber-500/20">
      {message}
    </div>
  );
}
