#!/usr/bin/env python3
"""Camera feeder for Jetson: captures CSI + USB cameras, serves MJPEG streams,
and sends VLM frames to the helmet-backend via WebSocket.

Streams (served over HTTP):
  /stream/left   — Left eye  (CSI-0 in normal mode, USB-IR in thermal mode)
  /stream/right  — Right eye (CSI-1 in normal mode, USB-IR in thermal mode)
  /stream/csi0   — Always CSI camera 0
  /stream/csi1   — Always CSI camera 1
  /stream/usb    — Always USB infrared camera

Endpoints:
  GET  /state    — JSON camera state
  POST /thermal  — Toggle thermal mode: {"thermal_on": true/false}

VLM integration:
  Connects to the Node backend WebSocket, sends frames from the VLM camera
  (default: csi0) at a configurable rate, and receives state broadcasts
  (including thermal_on toggled from the HUD frontend).

Environment variables:
  BACKEND_WS_URL   WebSocket URL for helmet-backend (default: ws://127.0.0.1:8765/ws/state)
  FEEDER_PORT      MJPEG/HTTP server port (default: 8090)
  VLM_CAMERA       Camera to feed VLM: csi0 | csi1 | usb (default: csi0)
  VLM_FPS          VLM frame send rate in fps (default: 2.5)
  CSI_WIDTH        CSI camera width (default: 1280)
  CSI_HEIGHT       CSI camera height (default: 720)
  CSI_FPS          CSI camera framerate (default: 30)
  USB_DEVICE       USB camera device path (default: /dev/video2)
  JPEG_QUALITY     MJPEG stream JPEG quality 0-100 (default: 85)
  CSI_SATURATION   CSI color saturation multiplier (default: 1.8)
  CSI_CONTRAST     CSI contrast multiplier (default: 1.1)
"""

import http.server
import io
import json
import logging
import os
import signal
import socketserver
import sys
import threading
import time

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("camera_feeder")

# ── Configuration ────────────────────────────────────────────────────────────

BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "ws://127.0.0.1:8765/ws/state")
FEEDER_PORT = int(os.environ.get("FEEDER_PORT", "8090"))
VLM_CAMERA = os.environ.get("VLM_CAMERA", "csi0")
VLM_FPS = float(os.environ.get("VLM_FPS", "2.5"))
VLM_INTERVAL = 1.0 / VLM_FPS if VLM_FPS > 0 else 0.4

CSI_W = int(os.environ.get("CSI_WIDTH", "1280"))
CSI_H = int(os.environ.get("CSI_HEIGHT", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "30"))
USB_DEVICE = os.environ.get("USB_DEVICE", "/dev/video2")
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))

CSI_SATURATION = float(os.environ.get("CSI_SATURATION", "1.8"))
CSI_CONTRAST = float(os.environ.get("CSI_CONTRAST", "1.1"))
CSI_BRIGHTNESS = 0

# ── Shared State ─────────────────────────────────────────────────────────────

shutdown_event = threading.Event()

# Frame buffers (written by capture threads, read by MJPEG server + VLM sender)
frames = {"csi0": None, "csi1": None, "usb": None}
frame_lock = threading.Lock()

# Thermal mode (toggled via backend state or /thermal endpoint)
thermal_on = False
thermal_lock = threading.Lock()


def get_thermal() -> bool:
    with thermal_lock:
        return thermal_on


def set_thermal(value: bool):
    global thermal_on
    with thermal_lock:
        thermal_on = value
    logger.info("Thermal mode: %s", "ON" if value else "OFF")


# ── GStreamer Pipelines ──────────────────────────────────────────────────────

def csi_pipeline(sensor_id: int, w: int = CSI_W, h: int = CSI_H, fps: int = CSI_FPS) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=RGBA ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def usb_pipeline(dev: str = USB_DEVICE) -> str:
    return (
        f"v4l2src device={dev} io-mode=2 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


# ── Image Enhancement ────────────────────────────────────────────────────────

def enhance_csi(bgr: np.ndarray) -> np.ndarray:
    """Apply contrast and saturation boost to CSI camera frames."""
    out = cv2.convertScaleAbs(bgr, alpha=CSI_CONTRAST, beta=CSI_BRIGHTNESS)
    if abs(CSI_SATURATION - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * CSI_SATURATION, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


# ── Camera Capture Threads ───────────────────────────────────────────────────

def capture_loop(cam_key: str, pipeline: str, enhance_fn=None):
    """Continuously capture frames from a GStreamer pipeline."""
    logger.info("[%s] Opening pipeline: %s", cam_key, pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.error("[%s] FAILED to open camera.", cam_key)
        return
    logger.info("[%s] Camera opened OK.", cam_key)

    while not shutdown_event.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            if enhance_fn is not None:
                frame = enhance_fn(frame)
            with frame_lock:
                frames[cam_key] = frame
        else:
            time.sleep(0.01)

    cap.release()
    logger.info("[%s] Camera released.", cam_key)


def start_capture_threads():
    """Start background threads for all three cameras."""
    threads = [
        threading.Thread(
            target=capture_loop,
            args=("csi0", csi_pipeline(0), enhance_csi),
            daemon=True,
            name="capture-csi0",
        ),
        threading.Thread(
            target=capture_loop,
            args=("csi1", csi_pipeline(1), enhance_csi),
            daemon=True,
            name="capture-csi1",
        ),
        threading.Thread(
            target=capture_loop,
            args=("usb", usb_pipeline(USB_DEVICE), None),
            daemon=True,
            name="capture-usb",
        ),
    ]
    for t in threads:
        t.start()
    return threads


# ── Stream Mapping ───────────────────────────────────────────────────────────

def resolve_stream(name: str) -> np.ndarray | None:
    """Get the frame for a given stream name, accounting for thermal mode.

    Mapping:
      normal mode:  left → csi0,  right → csi1
      thermal mode: left → usb,   right → usb
      csi0/csi1/usb: always direct
    """
    if name in ("csi0", "csi1", "usb"):
        with frame_lock:
            return frames.get(name)

    thermal = get_thermal()
    if name == "left":
        key = "usb" if thermal else "csi0"
    elif name == "right":
        key = "usb" if thermal else "csi1"
    else:
        return None

    with frame_lock:
        return frames.get(key)


# ── MJPEG HTTP Server ────────────────────────────────────────────────────────

class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streams and camera state."""

    # Suppress per-request log spam
    def log_message(self, fmt, *args):
        pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/stream/"):
            stream_name = self.path.split("/stream/")[1].split("?")[0]
            self._serve_mjpeg(stream_name)
        elif self.path == "/state":
            self._serve_state()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/thermal":
            self._handle_thermal_toggle()
        else:
            self.send_error(404)

    def _serve_mjpeg(self, stream_name: str):
        """Stream MJPEG frames for the given stream."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self._cors_headers()
        self.end_headers()

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        target_interval = 1.0 / 30  # ~30 fps cap

        while not shutdown_event.is_set():
            frame = resolve_stream(stream_name)
            if frame is None:
                time.sleep(0.05)
                continue

            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                time.sleep(0.01)
                continue

            jpeg_bytes = jpeg.tobytes()
            try:
                self.wfile.write(b"----frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                break

            time.sleep(target_interval)

    def _serve_state(self):
        """Return JSON camera state."""
        thermal = get_thermal()
        with frame_lock:
            cam_status = {k: (v is not None) for k, v in frames.items()}
        state = {
            "thermal_on": thermal,
            "cameras": cam_status,
            "vlm_camera": VLM_CAMERA,
            "streams": {
                "left": "usb" if thermal else "csi0",
                "right": "usb" if thermal else "csi1",
            },
        }
        body = json.dumps(state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_thermal_toggle(self):
        """Toggle thermal mode via POST."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw)
            if "thermal_on" in data:
                set_thermal(bool(data["thermal_on"]))
            else:
                set_thermal(not get_thermal())
        except Exception:
            set_thermal(not get_thermal())

        body = json.dumps({"thermal_on": get_thermal()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_mjpeg_server():
    """Start the MJPEG HTTP server in a background thread."""
    server = ThreadedHTTPServer(("0.0.0.0", FEEDER_PORT), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-server")
    thread.start()
    logger.info("MJPEG server listening on http://0.0.0.0:%d", FEEDER_PORT)
    logger.info("  Streams: /stream/left  /stream/right  /stream/csi0  /stream/csi1  /stream/usb")
    logger.info("  State:   GET /state    POST /thermal")
    return server


# ── WebSocket Client (Backend Connection) ────────────────────────────────────

def backend_ws_loop():
    """Connect to the helmet-backend WebSocket, send VLM frames, receive state."""
    try:
        import websocket  # websocket-client package
    except ImportError:
        logger.error(
            "websocket-client not installed. Install with: pip install websocket-client\n"
            "VLM frame feeding disabled; MJPEG streams still work."
        )
        return

    last_vlm_send = 0.0
    reconnect_delay = 1.0

    while not shutdown_event.is_set():
        try:
            logger.info("Connecting to backend: %s", BACKEND_WS_URL)
            ws = websocket.WebSocket()
            ws.settimeout(2.0)
            ws.connect(BACKEND_WS_URL)
            logger.info("Connected to backend.")
            reconnect_delay = 1.0

            while not shutdown_event.is_set():
                # ── Receive state broadcasts (non-blocking) ──
                try:
                    ws.settimeout(0.05)
                    raw = ws.recv()
                    if raw:
                        state = json.loads(raw)
                        if "thermal_on" in state:
                            new_thermal = bool(state["thermal_on"])
                            if new_thermal != get_thermal():
                                set_thermal(new_thermal)
                except websocket.WebSocketTimeoutException:
                    pass
                except Exception:
                    break

                # ── Send VLM frame at configured rate ──
                now = time.time()
                if now - last_vlm_send >= VLM_INTERVAL:
                    with frame_lock:
                        vlm_frame = frames.get(VLM_CAMERA)
                    if vlm_frame is not None:
                        ok, jpeg = cv2.imencode(
                            ".jpg", vlm_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        if ok:
                            import base64
                            b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
                            msg = json.dumps({
                                "type": "frame",
                                "data": b64,
                                "camera_id": f"jetson-{VLM_CAMERA}",
                            })
                            try:
                                ws.settimeout(5.0)
                                ws.send(msg)
                                last_vlm_send = now
                            except Exception:
                                break

        except Exception as exc:
            logger.warning("Backend WS error: %s (reconnecting in %.0fs)", exc, reconnect_delay)

        if not shutdown_event.is_set():
            shutdown_event.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 15.0)


def start_backend_ws():
    """Start the backend WebSocket client in a background thread."""
    thread = threading.Thread(target=backend_ws_loop, daemon=True, name="backend-ws")
    thread.start()
    return thread


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Camera feeder starting...")
    logger.info("  Backend WS:   %s", BACKEND_WS_URL)
    logger.info("  MJPEG port:   %d", FEEDER_PORT)
    logger.info("  VLM camera:   %s (%.1f fps)", VLM_CAMERA, VLM_FPS)
    logger.info("  CSI:          %dx%d @ %d fps", CSI_W, CSI_H, CSI_FPS)
    logger.info("  USB device:   %s", USB_DEVICE)

    # Start components
    capture_threads = start_capture_threads()
    mjpeg_server = start_mjpeg_server()
    ws_thread = start_backend_ws()

    # Wait for cameras to produce first frames
    time.sleep(1.0)
    with frame_lock:
        ready = {k: (v is not None) for k, v in frames.items()}
    logger.info("Camera status: %s", ready)

    # Block until shutdown
    def handle_signal(sig, _frame):
        logger.info("Shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    except KeyboardInterrupt:
        pass

    shutdown_event.set()
    mjpeg_server.shutdown()
    logger.info("Camera feeder stopped.")


if __name__ == "__main__":
    main()
