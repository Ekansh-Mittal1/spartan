#!/usr/bin/env python3
"""
Continuous webcam VLM inference with persistent world understanding (Jetson / NanoLLM).

Same logic as test_qwen_webcam.py but uses nano_llm for Jetson Nano/Orin instead of mlx_vlm.

Architecture (3 threads):
  - Main thread:  capture frames, display overlay, handle input
  - VLM thread:   run NanoLLM vision model on latest frame, overwrite latest summary (no queue)
  - World thread: use most recent summary when calling OpenAI; old summaries discarded
"""

import datetime
import json
import os
import resource
import sys
import textwrap
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# NanoLLM (Jetson); add NanoLLM repo to PYTHONPATH when running
from nano_llm import NanoLLM, ChatHistory

# ── Configuration ────────────────────────────────────────────────────────────

# Vision model for Jetson (NanoLLM-tested VLMs: VILA, LLaVA, etc.)
MODEL_PATH = "Efficient-Large-Model/VILA1.5-3b"
DEFAULT_PROMPT = "Describe this image briefly."

# Max pixel dimension for frames sent to the model (lower = faster)
MAX_FRAME_DIM = 768

# Camera index: 0 = first camera, /dev/video0 on Jetson
CAMERA_INDEX = 0

# OpenAI model for world understanding
OPENAI_MODEL = "gpt-5-mini"

# JSONL log file
LOG_FILE = "world_log.jsonl"

# System prompt for the world-understanding model
WORLD_SYSTEM_PROMPT = """\
You are a real-time people tracker. Track ONLY people: who is present, what they are wearing, and what they are doing. Ignore objects, furniture, and background.

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
7. Respond with ONLY the updated registry."""

# ── Shared state ─────────────────────────────────────────────────────────────

latest_frame = None           # webcam frame (main -> VLM thread)
frame_lock = threading.Lock()

latest_result = ""            # latest VLM summary for status line
result_lock = threading.Lock()

# Latest VLM summary only (old ones discarded); world thread uses most recent when it runs
latest_summary = None         # (timestamp, text, elapsed, put_time) or None
summary_condition = threading.Condition()

world_understanding = ""      # latest world understanding (world thread -> main)
world_lock = threading.Lock()

world_updated_at = 0.0        # time.time() when world was last updated (for flash)
_last_world_done_at = 0.0     # for measuring interval between world updates
_world_intervals = []         # recent intervals (for avg); max 10

running = True                # graceful shutdown flag


# ── Overlay helpers ──────────────────────────────────────────────────────────

def draw_vlm_summary(frame, text, max_chars_per_line=50):
    """Draw the latest VLM summary in a multi-line panel at the top of the frame."""
    if not text:
        return frame
    display = frame.copy()
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=max_chars_per_line) or [""])
    lines = lines[:4]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 22
    padding = 10
    label = "VLM:"
    bar_height = padding * 2 + line_height * (len(lines) + 1)
    h, w = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
    cv2.putText(display, label, (padding, padding + 16), font, font_scale,
                (180, 255, 180), thickness, cv2.LINE_AA)
    y = padding + 16 + line_height
    for line in lines:
        cv2.putText(display, line, (padding, y), font, font_scale,
                    (220, 220, 220), thickness, cv2.LINE_AA)
        y += line_height
    return display


def draw_text_overlay(frame, text, max_chars_per_line=45):
    """Draw multi-line text with a dark background strip at the bottom."""
    if not text:
        return frame
    display = frame.copy()
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=max_chars_per_line) or [""])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    line_height = 32
    padding = 14

    bg_height = padding * 2 + line_height * len(lines)
    h, w = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - bg_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

    y = h - bg_height + padding + 15
    for line in lines:
        cv2.putText(display, line, (padding, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    return display


# ── VLM inference thread (NanoLLM) ────────────────────────────────────────────

def inference_loop(model, chat_history, prompt):
    """Continuously grab the latest frame, run NanoLLM VLM, overwrite latest summary (old ones discarded)."""
    global latest_result, latest_summary, running

    while running:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        # Skip mostly-black frames (camera warmup / lens cap)
        if np.mean(frame) < 15:
            with result_lock:
                latest_result = "VLM: [skipped - black frame]"
            time.sleep(0.1)
            continue

        # Resize for speed
        height, width = frame.shape[:2]
        scale = MAX_FRAME_DIM / max(height, width)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # NanoLLM accepts np.ndarray; many VLMs expect RGB
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        try:
            start_t = time.time()
            chat_history.append("user", image=frame_rgb)
            chat_history.append("user", prompt, use_cache=True)
            embedding, _ = chat_history.embed_chat()

            reply = model.generate(
                embedding,
                kv_cache=chat_history.kv_cache,
                max_new_tokens=150,
                stop_tokens=getattr(chat_history.template, "stop", None),
            )
            # NanoLLM returns a stream-like object; .text may be set after iteration
            text = getattr(reply, "text", None)
            if not text:
                text = "".join(reply)
            text = (text or "").strip()

            chat_history.reset()

            elapsed = time.time() - start_t

            with result_lock:
                latest_result = f"VLM: {text}  [{elapsed:.2f}s]"

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with summary_condition:
                latest_summary = (timestamp, text, elapsed, time.time())
                summary_condition.notify()

        except Exception as e:
            print(f"VLM error: {e}")
            try:
                chat_history.reset()
            except Exception:
                pass


# ── World understanding thread ───────────────────────────────────────────────

def world_understanding_loop(client: OpenAI):
    """Use the most recent VLM summary when calling OpenAI; old summaries are discarded."""
    global world_understanding, world_updated_at, latest_summary, running
    global _last_world_done_at, _world_intervals

    while running:
        with summary_condition:
            summary_condition.wait(timeout=0.5)
            snapshot = latest_summary
            latest_summary = None  # consume so we don't reprocess same summary
        if snapshot is None:
            continue
        timestamp, frame_summary, vlm_elapsed, put_time = snapshot
        summary_age = time.time() - put_time
        print(f"[DEBUG] Using most recent summary (age {summary_age:.1f}s)")

        with world_lock:
            current_state = world_understanding

        if current_state:
            user_msg = (
                f"REGISTRY:\n{current_state}\n\n"
                f"NEW [{timestamp}]: {frame_summary}"
            )
        else:
            user_msg = f"FIRST [{timestamp}]: {frame_summary}"

        try:
            print("[DEBUG] Sending request to OpenAI...")
            start_t = time.time()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": WORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=4096,
            )
            openai_elapsed = time.time() - start_t

            print(f"[WORLD {openai_elapsed:.2f}s]")
            raw_content = None
            for choice in response.choices:
                msg = choice.message
                print(f"  role: {getattr(msg, 'role', '')}")
                c = getattr(msg, "content", "") or ""
                if raw_content is None:
                    raw_content = c
                print(f"  content: {c or '(empty)'}")
            print()

            new_understanding = (raw_content or "").strip()
            with world_lock:
                world_understanding = new_understanding
                world_updated_at = time.time()
            record = {
                "timestamp": timestamp,
                "frame_summary": frame_summary,
                "world_understanding": new_understanding,
                "model": OPENAI_MODEL,
                "vlm_latency_s": round(vlm_elapsed, 3),
                "openai_latency_s": round(openai_elapsed, 3),
            }
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            now = time.time()
            if _last_world_done_at > 0:
                interval = now - _last_world_done_at
                _world_intervals.append(interval)
                if len(_world_intervals) > 10:
                    _world_intervals.pop(0)
                avg_interval = sum(_world_intervals) / len(_world_intervals)
                print(f"[DEBUG] World update interval: {interval:.1f}s (avg: {avg_interval:.1f}s)")
            _last_world_done_at = now
        except Exception as e:
            print(f"[DEBUG] Request failed: {e}")
            print(f"OpenAI error: {e}")


# ── Memory monitor ───────────────────────────────────────────────────────────

def get_current_rss_mb():
    """Current process RSS in MB. Linux: /proc/self/status; macOS: ps."""
    try:
        if sys.platform == "linux":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        elif sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
            )
            return int(out.strip()) / 1024  # KB -> MB
    except Exception:
        pass
    return 0.0


def get_max_rss_mb():
    """Process max RSS in MB so far (Unix/macOS). resource.getrusage."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "linux":
            rss *= 1024  # KB -> bytes
        return rss / (1024 * 1024)
    except Exception:
        return 0.0


def memory_monitor_loop(interval_sec=5.0):
    """Print current and max RSS every interval_sec until running is False."""
    global running
    while running:
        time.sleep(interval_sec)
        if not running:
            break
        current_mb = get_current_rss_mb()
        max_mb = get_max_rss_mb()
        print(f"[MEM] current: {current_mb:.1f} MB  max: {max_mb:.1f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global latest_frame, running

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    prompt = DEFAULT_PROMPT
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print(f"Loading {MODEL_PATH} (NanoLLM for Jetson)...")
    model = NanoLLM.from_pretrained(
        MODEL_PATH,
        api="mlc",
        quantization="q4f16_ft",
    )
    if not getattr(model, "has_vision", True):
        print("Warning: model may not have vision; continuing anyway.")
    chat_history = ChatHistory(model)

    print("Model loaded. Starting continuous inference + world understanding.")
    print(f"Camera: index {CAMERA_INDEX}")
    print(f"VLM prompt: {prompt}")
    print(f"World model: {OPENAI_MODEL}")
    print(f"Log file: {LOG_FILE}")
    print("Press Q to quit.\n")

    vlm_thread = threading.Thread(
        target=inference_loop, args=(model, chat_history, prompt), daemon=True
    )
    world_thread = threading.Thread(
        target=world_understanding_loop, args=(client,), daemon=True
    )
    memory_thread = threading.Thread(
        target=memory_monitor_loop, kwargs={"interval_sec": 5.0}, daemon=True
    )
    vlm_thread.start()
    world_thread.start()
    memory_thread.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera (index {CAMERA_INDEX}).")
        running = False
        sys.exit(1)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        with frame_lock:
            latest_frame = frame.copy()

        with result_lock:
            vlm_text = latest_result

        with world_lock:
            world_text = world_understanding

        with world_lock:
            updated_at = world_updated_at

        display = frame.copy()
        display = draw_text_overlay(display, world_text)
        display = draw_vlm_summary(display, vlm_text)

        flash_age = time.time() - updated_at
        if 0 < updated_at and flash_age < 1.0:
            h, w = display.shape[:2]
            alpha = max(0.0, 1.0 - flash_age)
            radius = int(8 + 4 * alpha)
            color = (0, int(255 * alpha), 0)
            cx, cy = w - 30, 30
            cv2.circle(display, (cx, cy), radius, color, -1, cv2.LINE_AA)
            label_alpha = int(255 * alpha)
            cv2.putText(display, "UPDATED", (cx - 70, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, label_alpha, 0), 1, cv2.LINE_AA)

        cv2.imshow("NanoLLM Vision + World Understanding (Q to quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    running = False
    cap.release()
    cv2.destroyAllWindows()
    vlm_thread.join(timeout=2)
    world_thread.join(timeout=2)
    memory_thread.join(timeout=2)

    print(f"\nSession log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
