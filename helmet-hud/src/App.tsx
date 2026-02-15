import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCameraDevices, useCameraStream } from "./hooks/useCameraStream";
import { useHudState } from "./hooks/useHudState";
import { useMockState } from "./hooks/useMockState";
import { buildWarpMesh } from "./lib/warpMesh";
import { leftViewport, rightViewport } from "./layout";
import { DEFAULT_LAYOUT } from "./types";
import { VitalsPanel } from "./components/VitalsPanel";
import { StatusPanel } from "./components/StatusPanel";
import { ImuPanel } from "./components/ImuPanel";
import { ReasoningPanel } from "./components/ReasoningPanel";
import { AlertBanner } from "./components/AlertBanner";
import { MockVideoCanvas } from "./components/MockVideoCanvas";
import { WarpLayer } from "./components/WarpLayer";
import type { HudState } from "./types";

const PAD = 16;
const LAYOUT_W = 2560;
const LAYOUT_H = 1440;

function useScaleToFit() {
  const [scale, setScale] = useState(() =>
    Math.min(window.innerWidth / LAYOUT_W, window.innerHeight / LAYOUT_H, 1)
  );
  useEffect(() => {
    const onResize = () =>
      setScale(Math.min(window.innerWidth / LAYOUT_W, window.innerHeight / LAYOUT_H, 1));
    window.addEventListener("resize", onResize);
    onResize();
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return scale;
}

function getWsUrl(): string {
  const base = import.meta.env.VITE_WS_URL ?? "";
  if (base) {
    const url = new URL("/ws/state", base);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  }
  return "";
}

function getStreamBase(): string | null {
  const base = import.meta.env.VITE_STREAM_BASE ?? "";
  return base || null;
}

function StereoDotViewport() {
  return (
    <div className="relative flex h-full w-full items-center justify-center bg-black">
      <div
        className="h-3 w-3 rounded-full bg-white shrink-0"
        style={{ boxShadow: "0 0 0 2px rgba(255,255,255,0.5)" }}
      />
    </div>
  );
}

function Viewport({
  streamUrl,
  cameraStream,
  viewportWidth,
  viewportHeight,
  state,
  connected,
  stereoDotTest,
  warpK1,
  warpK2,
  separationOffset,
  zoom,
  cameraVideoRef,
}: {
  streamUrl: string | null;
  cameraStream: MediaStream | null;
  viewportWidth: number;
  viewportHeight: number;
  state: HudState;
  connected: boolean;
  stereoDotTest: boolean;
  warpK1: number;
  warpK2: number;
  separationOffset: number;
  zoom: number;
  /** When set (e.g. left viewport), this ref is assigned to the video element for frame upload to backend. */
  cameraVideoRef?: React.RefObject<HTMLVideoElement | null>;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const setVideoRef = useCallback(
    (el: HTMLVideoElement | null) => {
      videoRef.current = el;
      if (cameraVideoRef != null) (cameraVideoRef as React.MutableRefObject<HTMLVideoElement | null>).current = el;
    },
    [cameraVideoRef]
  );
  const mesh = useMemo(
    () => buildWarpMesh(viewportWidth, viewportHeight, warpK1, warpK2),
    [viewportWidth, viewportHeight, warpK1, warpK2]
  );

  useEffect(() => {
    const el = videoRef.current;
    if (!cameraStream || !el) return;
    el.srcObject = cameraStream;
    return () => {
      if (el) el.srcObject = null;
    };
  }, [cameraStream]);

  if (stereoDotTest) {
    return <StereoDotViewport />;
  }

  const sourceRef = streamUrl ? imgRef : videoRef;

  return (
    <div className="relative h-full w-full overflow-hidden bg-black">
      {streamUrl && (
        <>
          <img
            ref={imgRef}
            src={streamUrl}
            alt=""
            crossOrigin="anonymous"
            className="absolute opacity-0 pointer-events-none object-cover"
            style={{ width: viewportWidth, height: viewportHeight }}
            aria-hidden
          />
          <WarpLayer
            width={viewportWidth}
            height={viewportHeight}
            mesh={mesh}
            sourceRef={sourceRef as React.RefObject<HTMLImageElement | null>}
            zoom={zoom}
            uvOffsetX={separationOffset}
            className="absolute inset-0 w-full h-full"
          />
        </>
      )}
      {cameraStream != null && streamUrl == null && (
        <>
          <video
            ref={setVideoRef}
            autoPlay
            playsInline
            muted
            className="absolute opacity-0 pointer-events-none object-cover"
            style={{ width: viewportWidth, height: viewportHeight }}
            aria-hidden
          />
          <WarpLayer
            width={viewportWidth}
            height={viewportHeight}
            mesh={mesh}
            sourceRef={sourceRef as React.RefObject<HTMLVideoElement | null>}
            zoom={zoom}
            uvOffsetX={separationOffset}
            className="absolute inset-0 w-full h-full"
          />
        </>
      )}
      {streamUrl == null && cameraStream == null && (
        <div className="absolute inset-0 flex items-center justify-center">
          <MockVideoCanvas
            width={viewportWidth}
            height={viewportHeight}
            className="max-h-full max-w-full"
          />
        </div>
      )}
      <div className="absolute inset-0 pointer-events-none" aria-hidden>
        <div className="absolute" style={{ top: PAD, left: PAD }}>
          <VitalsPanel state={state} />
        </div>
        <div className="absolute" style={{ bottom: PAD, left: PAD }}>
          <StatusPanel state={state} />
        </div>
        <div className="absolute" style={{ top: PAD, right: PAD }}>
          <ImuPanel state={state} />
        </div>
        <div className="absolute" style={{ bottom: PAD, right: PAD }}>
          <ReasoningPanel state={state} />
        </div>
        {state.alert_banner ? <AlertBanner message={state.alert_banner} /> : null}
      </div>
      {!connected && (
        <div className="absolute bottom-2 left-2 rounded border border-amber-500/40 bg-neutral-950/80 px-2 py-1 text-xs text-amber-400">
          Disconnected
        </div>
      )}
    </div>
  );
}

function useCameraFromQuery(): boolean {
  return useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("source") === "camera" || params.get("camera") === "1";
  })[0];
}

/** Single-dot stereo test: ?test=dot — black viewports, one white dot centered per eye. */
function useStereoDotTest(): boolean {
  return useState(() => new URLSearchParams(window.location.search).get("test") === "dot")[0];
}

function App() {
  const layout = DEFAULT_LAYOUT;
  const left = leftViewport(layout);
  const right = rightViewport(layout);
  const streamBase = getStreamBase();
  const cameraFromQuery = useCameraFromQuery();
  const stereoDotTest = useStereoDotTest();
  const videoSource: "stream" | "camera" | "mock" = streamBase
    ? "stream"
    : cameraFromQuery
      ? "camera"
      : "mock";
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const { stream: cameraStream, error: cameraError, status: cameraStatus } = useCameraStream(
    videoSource === "camera",
    selectedCameraId
  );
  const cameraDevices = useCameraDevices(!!cameraStream);
  // Camera dropdown (optional): uncomment in JSX and use setSelectedCameraId + cameraDevices
  void setSelectedCameraId;
  void cameraDevices;

  const useExternalWs = Boolean(import.meta.env.VITE_WS_URL);
  const wsState = useHudState(useExternalWs ? getWsUrl() : "");
  const mockState = useMockState();
  const { state, connected, sendFrame, switchCamera, setThermal } = useExternalWs
    ? wsState
    : { ...mockState, sendFrame: () => {}, switchCamera: () => {}, setThermal: () => {} };
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);

  // Keyboard shortcut: T to toggle thermal mode
  useEffect(() => {
    if (!useExternalWs) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "t" || e.key === "T") {
        setThermal(!state.thermal_on);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [useExternalWs, setThermal, state.thermal_on]);

  const scale = useScaleToFit();

  const [warpK1, setWarpK1] = useState(0);
  const [warpK2, setWarpK2] = useState(0);
  const [separation, setSeparation] = useState(0.5);
  const [zoom, setZoom] = useState(1);

  const streamUrlLeft = videoSource === "stream" && streamBase ? `${streamBase}/stream/left` : null;
  const streamUrlRight = videoSource === "stream" && streamBase ? `${streamBase}/stream/right` : null;
  const camStream = videoSource === "camera" ? cameraStream : null;

  useEffect(() => {
    if (!useExternalWs || videoSource !== "camera" || !connected || !sendFrame) return;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const FRAME_INTERVAL_MS = 400;
    const id = setInterval(() => {
      const video = cameraVideoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (!blob) return;
          const r = new FileReader();
          r.onloadend = () => {
            const dataUrl = r.result;
            if (typeof dataUrl !== "string") return;
            const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
            if (base64) sendFrame(base64);
          };
          r.readAsDataURL(blob);
        },
        "image/jpeg",
        0.8
      );
    }, FRAME_INTERVAL_MS);
    return () => clearInterval(id);
  }, [useExternalWs, videoSource, connected, sendFrame]);

  return (
    <div className="flex h-screen w-screen items-center justify-center overflow-hidden bg-black">
      {!stereoDotTest && (streamBase ?? cameraFromQuery) && (
        <div className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 flex-wrap items-center justify-center gap-4 rounded border border-amber-500/40 bg-neutral-950/95 px-4 py-3 shadow-lg shadow-amber-500/10">
          <label className="flex items-center gap-2 text-sm text-neutral-200">
            <span className="w-6 font-mono">k1</span>
            <input
              type="range"
              min={-0.3}
              max={0.3}
              step={0.01}
              value={warpK1}
              onChange={(e) => setWarpK1(Number(e.target.value))}
              className="w-24 accent-amber-500"
            />
            <span className="w-10 font-mono text-amber-300">{warpK1.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-2 text-sm text-neutral-200">
            <span className="w-6 font-mono">k2</span>
            <input
              type="range"
              min={-0.15}
              max={0.15}
              step={0.01}
              value={warpK2}
              onChange={(e) => setWarpK2(Number(e.target.value))}
              className="w-24 accent-amber-500"
            />
            <span className="w-10 font-mono text-amber-300">{warpK2.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-2 text-sm text-neutral-200">
            <span className="w-16">Separation</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={separation}
              onChange={(e) => setSeparation(Number(e.target.value))}
              className="w-24 accent-amber-500"
            />
            <span className="w-10 font-mono text-amber-300">{separation.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-2 text-sm text-neutral-200">
            <span className="w-12">Zoom</span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.05}
              value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
              className="w-24 accent-amber-500"
            />
            <span className="w-10 font-mono text-amber-300">{zoom.toFixed(2)}×</span>
          </label>
        </div>
      )}
      {stereoDotTest && (
        <div className="absolute top-4 left-1/2 z-10 -translate-x-1/2 rounded border border-amber-500/30 bg-neutral-950/90 px-3 py-1.5 text-xs text-neutral-200">
          Stereo test: one dot per eye. Use divider + close one eye to check isolation.
        </div>
      )}
      {!stereoDotTest && videoSource === "camera" && cameraError && (
        <div className="absolute top-4 left-1/2 z-10 -translate-x-1/2 rounded border border-red-500/50 bg-red-950/90 px-4 py-2 text-sm text-red-300">
          Camera: {cameraError}
        </div>
      )}
      {!stereoDotTest && videoSource === "camera" && cameraStatus === "loading" && (
        <div className="absolute top-4 left-1/2 z-10 -translate-x-1/2 rounded border border-amber-500/40 bg-neutral-950/90 px-4 py-2 text-sm text-amber-200">
          Requesting camera…
        </div>
      )}
      {/* Jetson camera selector — shown when the backend reports multiple cameras (from camera_feeder.py) */}
      {!stereoDotTest && useExternalWs && state.camera_ids.length > 1 && (
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 rounded border border-cyan-500/40 bg-neutral-950/90 px-3 py-2 shadow-lg">
          <span className="text-xs text-cyan-400 font-medium">CAM</span>
          {state.camera_ids.map((camId) => (
            <button
              key={camId}
              onClick={() => switchCamera(camId)}
              className={`rounded px-2 py-1 text-xs font-mono transition-colors ${
                camId === state.active_camera_id
                  ? "bg-cyan-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200"
              }`}
            >
              {camId.replace("jetson-cam-", "cam ")}
            </button>
          ))}
        </div>
      )}
      {/* Single camera indicator (when only one Jetson camera is connected) */}
      {!stereoDotTest && useExternalWs && state.camera_ids.length === 1 && state.active_camera_id && (
        <div className="absolute top-4 left-4 z-10 rounded border border-cyan-500/30 bg-neutral-950/80 px-2 py-1 text-xs text-cyan-400 font-mono">
          {state.active_camera_id.replace("jetson-cam-", "cam ")}
        </div>
      )}
      {/* Thermal mode indicator + toggle (press T to toggle) */}
      {!stereoDotTest && useExternalWs && (
        <button
          onClick={() => setThermal(!state.thermal_on)}
          className={`absolute top-4 right-4 z-10 flex items-center gap-2 rounded border px-3 py-2 text-xs font-bold uppercase tracking-wider shadow-lg transition-colors ${
            state.thermal_on
              ? "border-red-500/60 bg-red-950/90 text-red-300 shadow-red-500/20"
              : "border-neutral-600/40 bg-neutral-950/80 text-neutral-500 hover:border-neutral-500 hover:text-neutral-300"
          }`}
          title="Toggle thermal / infrared view (T)"
        >
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
          </svg>
          {state.thermal_on ? "IR ON" : "IR"}
        </button>
      )}
      <div
        className="flex shrink-0 origin-center bg-black"
        style={{
          width: LAYOUT_W,
          height: LAYOUT_H,
          transform: `scale(${scale})`,
        }}
      >
        <div
          className="flex-shrink-0 overflow-hidden"
          style={{ width: left.w, height: left.h }}
        >
          <Viewport
            streamUrl={streamUrlLeft}
            cameraStream={camStream}
            viewportWidth={left.w}
            viewportHeight={left.h}
            state={state}
            connected={connected}
            stereoDotTest={stereoDotTest}
            warpK1={warpK1}
            warpK2={warpK2}
            separationOffset={(0.5 - separation) * 0.12}
            zoom={zoom}
            cameraVideoRef={cameraVideoRef}
          />
        </div>
        <div
          className="flex-shrink-0 bg-black"
          style={{ width: layout.center_gap_px, height: layout.panel_height }}
        />
        <div
          className="flex-shrink-0 overflow-hidden"
          style={{ width: right.w, height: right.h }}
        >
          <Viewport
            streamUrl={streamUrlRight}
            cameraStream={camStream}
            viewportWidth={right.w}
            viewportHeight={right.h}
            state={state}
            connected={connected}
            stereoDotTest={stereoDotTest}
            warpK1={warpK1}
            warpK2={warpK2}
            separationOffset={(separation - 0.5) * 0.12}
            zoom={zoom}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
