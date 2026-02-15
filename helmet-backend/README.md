# helmet-backend

Node server for the helmet HUD: receives camera frames from the client, runs VLM (vision) and world-context (OpenAI), pushes state over WebSocket.

## Setup

```bash
cd helmet-backend
npm install
```

Create `.env` (copy from `.env.example`). Required for world-context: `OPENAI_API_KEY`. For VLM: use OpenAI (default) or local VLM service.

Optional:

- `PORT` — default 8765
- `OPENAI_WORLD_MODEL` — default `gpt-5-mini`
- `VLM_MODE` — `openai` or `local` (default `openai`)
- `VLM_LOCAL_URL` — e.g. `http://127.0.0.1:5000` when `VLM_MODE=local`
- `VLM_PROMPT` — default "Describe this image briefly."
- `WORLD_LOG_FILE` — default `world_log.jsonl`

## Run

```bash
npm start
```

WebSocket: `ws://localhost:8765/ws/state`. Clients receive JSON HUD state (including `vlm_text`, `world_context`). Clients can send frames: JSON `{ "type": "frame", "data": "<base64>" }` or binary message (raw image bytes).

## VLM

- **OpenAI** (`VLM_MODE=openai`): Uses gpt-4o-mini for vision. Requires `OPENAI_API_KEY`.
- **Local** (`VLM_MODE=local`): POSTs frames to the Python VLM service at `VLM_LOCAL_URL`. Start the service first (e.g. `python -m vlm_service` from repo `vlm_service/`). See root README and `docs/JETSON_DEPLOY.md` for the edge pipeline.
