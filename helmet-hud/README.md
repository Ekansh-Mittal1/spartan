# helmet-hud

Fullscreen HUD for a 1440×2560 helmet display with two side-by-side eye viewports. Pure JavaScript (Vite + React + Tailwind); no Python. Runs in the browser with in-browser mock video and state by default, or connect to an external stream and WebSocket when you run a backend elsewhere.

## Features

- **Dual eye viewports**: same HUD in left and right viewports; layout embedded in app (panel size, eye rects, center gap).
- **Four corner panels**: vitals (top-left), system status (bottom-left), IMU/heading (top-right), reasoning (bottom-right); alert banner in center when set.
- **Default: in-browser mock**: synthetic canvas video and mock state (BPM, temp, heading, etc.) at ~30 Hz. No backend required.
- **Optional external backend**: set `VITE_WS_URL` and `VITE_STREAM_BASE` to use an external server for video (MJPEG) and state (WebSocket).

## Key bindings (mock mode)

| Key | Action |
|-----|--------|
| **R** | Toggle drawer |
| **H** | Start/stop stress spike (BPM ramp) |
| **A** | Trigger alert banner "HIGH HEAT" (5 s) |
| **M** | Toggle mic / device |
| **L** | Toggle light |
| **T** | Toggle thermal |

## Install

```bash
cd helmet-hud
npm install
```

## Run

**Dev (mock video + mock state):**

```bash
npm run dev
```

Open http://localhost:5173. No backend needed.

**Production build:**

```bash
npm run build
npm run preview
```

**External backend:** If you run a server that exposes WebSocket at `/ws/state` (and optionally MJPEG at `/stream/left`, `/stream/right`), create a `.env` (see `.env.example`) and set:

- `VITE_WS_URL` — base URL of the backend (e.g. `http://localhost:8765`). App will connect to `ws://host/ws/state` and, when using the camera, send frames to the same WebSocket for VLM + world-context.
- `VITE_STREAM_BASE` — optional; same base URL for video streams.

The **helmet-backend** (Node) in this repo provides `/ws/state`: it receives camera frames from the HUD, runs VLM (vision) and world-context (OpenAI), and pushes `vlm_text` and `world_context` to the bottom-right Reasoning panel. Run it from `helmet-backend/` with `OPENAI_API_KEY` set.

## Repo layout

- `src/App.tsx` — Main layout, viewports, mock vs external source selection.
- `src/components/` — VitalsPanel, StatusPanel, ImuPanel, ReasoningPanel, AlertBanner, MockVideoCanvas.
- `src/hooks/useMockState.ts` — In-browser mock state + key triggers.
- `src/hooks/useHudState.ts` — WebSocket state (when `VITE_WS_URL` is set).
- `src/mock/drawFrame.ts` — Synthetic canvas frame for mock video.
- `src/types.ts` — HudState, DisplayLayout, DEFAULT_LAYOUT.
- `src/layout.ts` — leftViewport(), rightViewport() from layout.
