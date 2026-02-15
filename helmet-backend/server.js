import "dotenv/config";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import OpenAI from "openai";
import fs from "fs";

const PORT = Number(process.env.PORT) || 8765;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_WORLD_MODEL ?? "gpt-5-mini";
const VLM_MODE = (process.env.VLM_MODE || "openai").toLowerCase();
const VLM_LOCAL_URL = process.env.VLM_LOCAL_URL?.replace(/\/$/, "") ?? "";
const VLM_PROMPT = process.env.VLM_PROMPT || "Describe this image briefly.";
const LOG_FILE = process.env.WORLD_LOG_FILE || "world_log.jsonl";

/** True when using a GPT-5 reasoning model (Responses API: reasoning.effort, max_output_tokens; no temperature/max_tokens). */
const isGpt5 = /^gpt-5(-|$)/.test(OPENAI_MODEL);

const WORLD_SYSTEM_PROMPT = `You are a real-time people tracker. Track ONLY people: who is present, what they are wearing, and what they are doing. Ignore objects, furniture, and background.

OUTPUT FORMAT (use exactly this, no prose):
PEOPLE:
- <id>: wearing <clothing description> | doing <current activity> | status: present/absent | last seen: <timestamp>

CHANGES:
- <+added / -removed / ~changed> <id>: <what changed>

RULES:
1. Track ONLY people. Do not list objects, animals (unless you treat them as "person" by convention), or scenery.
2. For each person: describe clothing (e.g. "red shirt, dark pants") and current action (e.g. "sitting", "looking at camera", "holding phone").
3. ADD a new person when first seen. KEEP people not in frame as status: absent. Only REMOVE after many consecutive observations without them.
4. UPDATE clothing or activity when the observation clearly indicates a change. Do not invent or hallucinate changes.
5. If the observation says the image is black, blank, or has no visible content, return the previous registry UNCHANGED.
6. Be terse. No filler. No prose. Only the structured format above.
7. Respond with ONLY the updated registry.`;

const defaultHudState = () => ({
  bpm: 0,
  temp_c: 0,
  quality: 1,
  heading_deg: 0,
  mic_level: 0,
  drawer_open: false,
  mic_on: true,
  light_on: false,
  thermal_on: false,
  alert_banner: null,
  mode: "NORM",
  reasoning_text: "",
  vlm_text: "",
  world_context: "",
  gps_lat: 0,
  gps_lon: 0,
  anchor_heading_deg: 0,
  /** Currently active camera ID (the one being sent to VLM). */
  active_camera_id: "",
  /** All known camera IDs that have sent at least one frame. */
  camera_ids: [],
});

let hudState = defaultHudState();
let worldUnderstanding = "";
let clients = new Set();
let processing = false;

// ── Multi-camera state ──────────────────────────────────────────────────────
/** Per-camera latest frame buffer. Key = camera_id (e.g. "jetson-cam-0", "browser"). */
const cameraFrames = new Map();
/** Which camera the VLM pipeline reads from. Empty string = accept any / first seen. */
let activeCameraId = "";
/** Convenience: the frame the pipeline should process next (from the active camera). */
let latestFrame = null;

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

function broadcast() {
  const payload = JSON.stringify(hudState);
  for (const ws of clients) {
    if (ws.readyState === 1) ws.send(payload);
  }
}

async function runVLM(imageBuffer) {
  const b64 = imageBuffer.toString("base64");
  if (VLM_MODE === "local") {
    if (!VLM_LOCAL_URL) return { text: "[VLM_MODE=local but VLM_LOCAL_URL not set]", elapsed: 0 };
    const start = Date.now();
    try {
      const res = await fetch(`${VLM_LOCAL_URL}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: b64, prompt: VLM_PROMPT }),
      });
      const raw = await res.text();
      let data;
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        console.error("VLM error: response not JSON", res.status, raw?.slice(0, 200));
        return { text: `[VLM error: ${res.status} non-JSON]`, elapsed: (Date.now() - start) / 1000 };
      }
      if (!res.ok) {
        console.error("VLM error:", res.status, data?.text ?? data?.detail ?? raw?.slice(0, 200));
        return {
          text: typeof data?.text === "string" ? data.text : `[VLM ${res.status}]`,
          elapsed: (Date.now() - start) / 1000,
        };
      }
      const text = typeof data?.text === "string" ? data.text : String(data?.text ?? "");
      const elapsed = typeof data?.elapsed_s === "number" ? data.elapsed_s : (Date.now() - start) / 1000;
      return { text: text.trim(), elapsed };
    } catch (e) {
      console.error("VLM error:", e.message);
      return { text: `[VLM error: ${e.message}]`, elapsed: 0 };
    }
  }
  if (!openai) return { text: "[No OPENAI_API_KEY]", elapsed: 0 };
  const start = Date.now();
  try {
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 150,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: VLM_PROMPT },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } },
          ],
        },
      ],
    });
    const text = (res.choices?.[0]?.message?.content ?? "").trim();
    const elapsed = (Date.now() - start) / 1000;
    return { text, elapsed };
  } catch (e) {
    console.error("VLM error:", e.message);
    return { text: `[VLM error: ${e.message}]`, elapsed: 0 };
  }
}

function extractResponsesOutputText(res) {
  if (typeof res.output_text === "string") return res.output_text.trim();
  const out = res.output;
  if (!Array.isArray(out)) return "";
  for (const item of out) {
    if (item.type === "message" && Array.isArray(item.content)) {
      for (const block of item.content) {
        if (block?.type === "output_text" && typeof block.text === "string")
          return block.text.trim();
      }
    }
  }
  return "";
}

async function runWorld(timestamp, frameSummary, vlmElapsed) {
  if (!openai) {
    worldUnderstanding = "[No OPENAI_API_KEY]";
    hudState.world_context = worldUnderstanding;
    hudState.vlm_text = frameSummary;
    broadcast();
    return;
  }
  const userMsg = worldUnderstanding
    ? `REGISTRY:\n${worldUnderstanding}\n\nNEW [${timestamp}]: ${frameSummary}`
    : `FIRST [${timestamp}]: ${frameSummary}`;
  try {
    const start = Date.now();
    let newUnderstanding;
    if (isGpt5) {
      const res = await openai.responses.create({
        model: OPENAI_MODEL,
        instructions: WORLD_SYSTEM_PROMPT,
        input: [{ role: "user", content: userMsg }],
        reasoning: { effort: "low" },
        max_output_tokens: 4096,
      });
      newUnderstanding = extractResponsesOutputText(res);
    } else {
      const res = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [
          { role: "system", content: WORLD_SYSTEM_PROMPT },
          { role: "user", content: userMsg },
        ],
        max_completion_tokens: 4096,
      });
      newUnderstanding = (res.choices?.[0]?.message?.content ?? "").trim();
    }
    const openaiElapsed = (Date.now() - start) / 1000;
    worldUnderstanding = newUnderstanding;
    hudState.world_context = newUnderstanding;
    hudState.vlm_text = frameSummary;
    broadcast();
    const record = {
      timestamp,
      frame_summary: frameSummary,
      world_understanding: newUnderstanding,
      model: OPENAI_MODEL,
      vlm_latency_s: Math.round(vlmElapsed * 1000) / 1000,
      openai_latency_s: Math.round(openaiElapsed * 1000) / 1000,
    };
    fs.appendFileSync(LOG_FILE, JSON.stringify(record) + "\n");
  } catch (e) {
    console.error("World error:", e.message);
    hudState.vlm_text = frameSummary;
    const is429 = e.status === 429 || String(e.message).includes("429");
    hudState.world_context =
      worldUnderstanding ?? `[World error: ${e.message}]`;
    if (is429) {
      console.warn("World rate limited (429). Keeping previous state; try again in ~20s or add billing.");
    }
    broadcast();
  }
}

async function pipeline() {
  if (processing || !latestFrame) return;
  processing = true;
  const frame = latestFrame;
  latestFrame = null;
  try {
    const { text: vlmText, elapsed: vlmElapsed } = await runVLM(frame);
    const timestamp = new Date().toISOString();
    hudState.vlm_text = vlmText;
    broadcast();
    await runWorld(timestamp, vlmText, vlmElapsed);
  } finally {
    processing = false;
    if (latestFrame) setImmediate(pipeline);
  }
}

const httpServer = createServer((_req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("helmet-backend");
});
const wss = new WebSocketServer({ server: httpServer, path: "/ws/state" });
httpServer.listen(PORT);

/** Register a camera_id and update the HUD camera list. */
function registerCamera(cameraId) {
  if (!cameraId) return;
  if (!hudState.camera_ids.includes(cameraId)) {
    hudState.camera_ids.push(cameraId);
    console.log(`Camera registered: ${cameraId}  (${hudState.camera_ids.length} total)`);
  }
  // Auto-set the first camera as active if none is set yet.
  if (!activeCameraId) {
    activeCameraId = cameraId;
    hudState.active_camera_id = cameraId;
    console.log(`Active camera auto-set: ${cameraId}`);
  }
}

/** Accept a frame, store per-camera, and trigger the VLM pipeline if it's the active camera. */
function ingestFrame(frameBuffer, cameraId) {
  cameraId = cameraId || "browser";
  registerCamera(cameraId);
  cameraFrames.set(cameraId, frameBuffer);

  // Only feed the VLM pipeline from the active camera.
  if (cameraId === activeCameraId || !activeCameraId) {
    latestFrame = frameBuffer;
    setImmediate(pipeline);
  }
}

wss.on("connection", (ws) => {
  clients.add(ws);
  ws.send(JSON.stringify(hudState));
  ws.on("message", (data, isBinary) => {
    if (isBinary) {
      // Raw binary frame (e.g. direct JPEG bytes) — no camera_id, use "browser"
      ingestFrame(data, "browser");
      return;
    }
    // Text frame — ws@8 still delivers a Buffer; convert to string for JSON parsing
    const str = Buffer.isBuffer(data) ? data.toString("utf-8") : String(data);
    try {
      const msg = JSON.parse(str);

      // Frame from browser or camera feeder: { type: "frame", data: "<base64>", camera_id?: "..." }
      if (msg.type === "frame" && typeof msg.data === "string") {
        ingestFrame(Buffer.from(msg.data, "base64"), msg.camera_id || "browser");
        return;
      }

      // Switch active camera: { type: "set_active_camera", camera_id: "jetson-cam-2" }
      if (msg.type === "set_active_camera" && typeof msg.camera_id === "string") {
        activeCameraId = msg.camera_id;
        hudState.active_camera_id = msg.camera_id;
        console.log(`Active camera switched to: ${msg.camera_id}`);
        broadcast();
        return;
      }

      // Toggle thermal mode: { type: "set_thermal", thermal_on: true/false }
      if (msg.type === "set_thermal" && typeof msg.thermal_on === "boolean") {
        hudState.thermal_on = msg.thermal_on;
        console.log(`Thermal mode: ${msg.thermal_on ? "ON" : "OFF"}`);
        broadcast();
        return;
      }
    } catch {
      // ignore malformed messages
    }
  });
  ws.on("close", () => clients.delete(ws));
});

console.log(`helmet-backend: ws://localhost:${PORT}/ws/state VLM=${VLM_MODE}${VLM_MODE === "local" ? ` url=${VLM_LOCAL_URL || "(none)"}` : ""} OPENAI_API_KEY=${OPENAI_API_KEY ? "set" : "not set"}`);
if (VLM_MODE === "local" && !VLM_LOCAL_URL) console.warn("VLM_MODE=local but VLM_LOCAL_URL not set. Start the VLM service (e.g. python -m vlm_service) and set VLM_LOCAL_URL.");
if (VLM_MODE !== "local" && !OPENAI_API_KEY) console.warn("Set OPENAI_API_KEY for VLM (vision) and world-context.");
