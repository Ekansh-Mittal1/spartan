#!/usr/bin/env python3
"""
Continuous webcam VLM inference with persistent world understanding.

Architecture (3 threads):
  - Main thread:  capture frames, display overlay, handle input
  - VLM thread:   run mlx-vlm on latest frame, overwrite latest summary (no queue)
  - World thread:  use most recent summary when calling OpenAI; old summaries discarded
                   world understanding, persist to JSONL log
"""

import datetime
import json
import os
import resource
import sys
import tempfile
import textwrap
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

# 4-bit quantized VLM for speed on Apple Silicon
MODEL_PATH = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
DEFAULT_PROMPT = "Describe this image briefly."

# Max pixel dimension for frames sent to the model (lower = faster)
MAX_FRAME_DIM = 768

# Camera index: 0 = first camera, 1 = second camera
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
    # Limit to 4 lines so the panel doesn't cover too much of the video
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


# ── VLM inference thread ─────────────────────────────────────────────────────

def inference_loop(model, processor, prompt):
    """Continuously grab the latest frame, run VLM, overwrite latest summary (old ones discarded)."""
    global latest_result, latest_summary, running

    formatted_prompt = apply_chat_template(
        processor, model.config, prompt, num_images=1
    )

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

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, resized)

        try:
            start_t = time.time()
            output = generate(
                model,
                processor,
                formatted_prompt,
                temp_path,
                verbose=False,
                max_tokens=150,
            )
            elapsed = time.time() - start_t
            text = output.text if hasattr(output, "text") else str(output)
            text = text.strip()

            # Update status line
            with result_lock:
                latest_result = f"VLM: {text}  [{elapsed:.2f}s]"

            # Overwrite latest summary (world thread will use most recent when it runs; old ones discarded)
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with summary_condition:
                latest_summary = (timestamp, text, elapsed, time.time())
                summary_condition.notify()

        except Exception as e:
            print(f"VLM error: {e}")
        finally:
            os.unlink(temp_path)


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

        # Build the user message
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
                max_completion_tokens=4096,  # reasoning models need room for reasoning + output; 500 was all reasoning, 0 content
            )
            openai_elapsed = time.time() - start_t

            # Print message(s) only
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

            # Update shared state and world log
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

            # Measure interval between world updates
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

    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    prompt = DEFAULT_PROMPT
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print(f"Loading {MODEL_PATH} (this might take a minute initially)...")
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    print("Model loaded. Starting continuous inference + world understanding.")
    print(f"Camera: index {CAMERA_INDEX}")
    print(f"VLM prompt: {prompt}")
    print(f"World model: {OPENAI_MODEL}")
    print(f"Log file: {LOG_FILE}")
    print("Press Q to quit.\n")

    # Start background threads
    vlm_thread = threading.Thread(
        target=inference_loop, args=(model, processor, prompt), daemon=True
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

        # Store latest frame for VLM thread
        with frame_lock:
            latest_frame = frame.copy()

        # Read latest VLM summary for top status bar
        with result_lock:
            vlm_text = latest_result

        # Read latest world understanding for bottom overlay
        with world_lock:
            world_text = world_understanding

        # Check if world understanding was recently updated (flash for ~1s)
        with world_lock:
            updated_at = world_updated_at

        # Draw overlays: VLM summary at top, world understanding at bottom
        display = frame.copy()
        display = draw_text_overlay(display, world_text)
        display = draw_vlm_summary(display, vlm_text)

        # Flash indicator when world summary was just updated
        flash_age = time.time() - updated_at
        if 0 < updated_at and flash_age < 1.0:
            h, w = display.shape[:2]
            # Pulsing alpha: bright at 0s, fades over 1s
            alpha = max(0.0, 1.0 - flash_age)
            # Green dot
            radius = int(8 + 4 * alpha)
            color = (0, int(255 * alpha), 0)
            cx, cy = w - 30, 30
            cv2.circle(display, (cx, cy), radius, color, -1, cv2.LINE_AA)
            # "UPDATED" label
            label_alpha = int(255 * alpha)
            cv2.putText(display, "UPDATED", (cx - 70, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, label_alpha, 0), 1, cv2.LINE_AA)

        cv2.imshow("MLX Vision + World Understanding (Q to quit)", display)

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
