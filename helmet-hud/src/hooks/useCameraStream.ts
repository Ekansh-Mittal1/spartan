import { useEffect, useState } from "react";

export type CameraStatus = "idle" | "loading" | "active";

export function useCameraStream(
  enabled: boolean,
  deviceId: string | null
): {
  stream: MediaStream | null;
  error: string | null;
  status: CameraStatus;
} {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");

  useEffect(() => {
    if (!enabled) {
      setStream((prev) => {
        prev?.getTracks().forEach((t) => t.stop());
        return null;
      });
      setError(null);
      setStatus("idle");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera not supported (need HTTPS or localhost)");
      setStatus("idle");
      return;
    }

    setStatus("loading");
    setError(null);

    const controller = new AbortController();
    const constraints: MediaStreamConstraints = {
      video: {
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((mediaStream) => {
        if (controller.signal.aborted) {
          mediaStream.getTracks().forEach((t) => t.stop());
          return;
        }
        setStream(mediaStream);
        setStatus("active");
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const message = err instanceof Error ? err.message : "Camera access failed";
        setError(message);
        setStatus("idle");
      });

    return () => {
      controller.abort();
      setStream((prev) => {
        prev?.getTracks().forEach((t) => t.stop());
        return null;
      });
      setStatus("idle");
    };
  }, [enabled, deviceId]);

  return { stream, error, status };
}

export interface CameraDevice {
  deviceId: string;
  label: string;
}

/** Enumerate video input devices. Call after permission is granted for accurate labels. */
export function useCameraDevices(hasPermission: boolean): CameraDevice[] {
  const [devices, setDevices] = useState<CameraDevice[]>([]);

  useEffect(() => {
    if (!hasPermission || !navigator.mediaDevices?.enumerateDevices) return;
    navigator.mediaDevices.enumerateDevices().then((list) => {
      setDevices(
        list
          .filter((d) => d.kind === "videoinput")
          .map((d) => ({ deviceId: d.deviceId, label: d.label || `Camera ${d.deviceId.slice(0, 8)}` }))
      );
    });
  }, [hasPermission]);

  return devices;
}
