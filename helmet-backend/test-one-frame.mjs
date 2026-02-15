#!/usr/bin/env node
/**
 * Quick test: send one frame to the backend and print the first state that has vlm_text.
 * Usage: node test-one-frame.mjs [wsUrl]
 * Requires: backend running (npm start), OPENAI_API_KEY in .env
 */
import WebSocket from "ws";

const WS_URL = process.argv[2] ?? "ws://localhost:8765/ws/state";

// Minimal 1x1 pixel JPEG (valid baseline)
const MINI_JPEG_BASE64 =
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMAAAQEAwEAAAAAAAAAAAAAAAUGBwABAgT/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMAAAQEAwEAAAAAAAAAAAAAAAUGBwABAgT/2Q==";

const ws = new WebSocket(WS_URL);
let done = false;

ws.on("open", () => {
  ws.send(JSON.stringify({ type: "frame", data: MINI_JPEG_BASE64 }));
});

const timeout = setTimeout(() => {
  if (!done) {
    done = true;
    console.log("Timeout (15s) – check backend is running and OPENAI_API_KEY is set.");
    ws.close();
  }
}, 15000);

ws.on("message", (data) => {
  if (done) return;
  const msg = JSON.parse(data.toString());
  if (msg.vlm_text != null && msg.vlm_text !== "") {
    console.log("VLM:", msg.vlm_text);
    console.log("World:", msg.world_context ?? "(waiting…)");
    if (msg.world_context != null && msg.world_context !== "") {
      done = true;
      clearTimeout(timeout);
      ws.close();
    }
  }
});

ws.on("error", (err) => {
  console.error("WS error:", err.message);
  process.exit(1);
});

ws.on("close", () => {
  if (!done) console.log("Closed before receiving vlm_text (wait a few seconds and check backend logs).");
});
