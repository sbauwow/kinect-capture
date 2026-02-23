#!/usr/bin/env python3
"""Kinect RGB + Scarlett Audio capture tool.

Live preview from Xbox 360 Kinect or webcam with audio recording.
Press R to record, S to snapshot, V to cycle video source, A to cycle audio device,
B to toggle background removal, [/] to adjust depth threshold, Q to quit.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import sounddevice as sd

freenect = None   # lazy import — only when Kinect is connected
open3d = None     # lazy import — only when cloud mode is activated
mp_pose = None    # lazy import — only when skeleton overlay is activated
mp_drawing = None

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FPS = 30
AUDIO_SAMPLERATE = 48000
AUDIO_CHANNELS = 2
AUDIO_BLOCKSIZE = 1024
TOOLBAR_H = 40

# Kinect v1 depth camera intrinsics
KINECT_FX = 594.21
KINECT_FY = 591.04
KINECT_CX = 339.31
KINECT_CY = 242.74
CLOUD_SUBSAMPLE = 2  # take every Nth pixel (2 → ~76k points)

# Visual effects list (cycled with E key)
EFFECTS = [
    "off", "ascii", "pixelate", "edge_glow", "thermal", "trail",
    "vhs", "matrix", "kaleidoscope", "posterize", "chroma_shift",
    "predator", "wireframe", "particle",
    "silhouette", "xray", "depth_fog",
    "crt", "gameboy", "old_film", "sepia", "negative", "solarize",
    "emboss", "cartoon", "glitch", "hue_rotate", "mirror",
    "tilt_shift", "oil_paint", "color_isolate",
    "bass_pulse", "audio_glow", "depth_layers",
]

# ASCII art character ramp (dark → bright)
ASCII_CHARS = " .:-=+*#%@"


def find_audio_input_devices():
    """Return list of (device_index, short_name) for real hardware input devices."""
    devices = []
    seen = set()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        # Filter to hw: devices (real ALSA hardware)
        if "hw:" not in name:
            continue
        # Shorten name: "Scarlett 4i4 USB: Audio (hw:2,0)" → "Scarlett 4i4 USB"
        short = name.split(":")[0].strip() if ":" in name else name
        if short not in seen:
            seen.add(short)
            devices.append((i, short))
    return devices


def _kinect_available():
    """Check if a Kinect is connected by looking for its USB VID:PID."""
    try:
        result = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=3,
        )
        # Kinect camera is 045e:02ae
        return "045e:02ae" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def find_video_sources():
    """Return list of video sources: Kinect modes (if connected) + V4L2 webcams.

    Each entry is a dict with keys: type, name, and device (for v4l2).
    Types: 'kinect_rgb', 'kinect_depth', 'kinect_cloud', 'v4l2'
    """
    sources = []
    if _kinect_available():
        sources.append({"type": "kinect_rgb", "name": "Kinect RGB"})
        sources.append({"type": "kinect_depth", "name": "Kinect Depth"})
        sources.append({"type": "kinect_cloud", "name": "Kinect Cloud"})
    # Scan for V4L2 webcams
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            current_name = None
            for line in lines:
                if not line.startswith("\t"):
                    # Device name line: "icspring camera: icspring camer (usb-...)"
                    current_name = line.split("(")[0].split(":")[0].strip()
                else:
                    dev = line.strip()
                    # Only take the first /dev/videoN per device (skip metadata nodes)
                    if dev.startswith("/dev/video") and current_name:
                        sources.append({
                            "type": "v4l2",
                            "name": current_name,
                            "device": dev,
                        })
                        current_name = None  # only first node
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return sources


class KinectCapture:
    def __init__(self, output_dir=".", audio_device=None, duration=None,
                 countdown=3, mp4=False, bg_remove=False, bokeh=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.countdown = countdown
        self.mp4_mode = mp4  # True=H.264 MP4, False=FFV1 MKV

        # Countdown state
        self.countdown_end = None  # monotonic time when countdown finishes

        # Video sources
        self.video_sources = find_video_sources()
        if not self.video_sources:
            print("ERROR: No video sources found")
            sys.exit(1)
        self.video_src_idx = 0
        self.webcam_cap = None  # cv2.VideoCapture for V4L2 sources
        # Import freenect only if Kinect is connected
        self.has_kinect = any(s["type"].startswith("kinect") for s in self.video_sources)
        if self.has_kinect:
            global freenect
            import freenect as _freenect
            freenect = _freenect
        # If starting on a V4L2 source, open it now
        if self._current_video_source["type"] == "v4l2":
            self.webcam_cap = cv2.VideoCapture(self._current_video_source["device"])

        # Audio devices
        self.input_devices = find_audio_input_devices()
        if not self.input_devices:
            print("WARNING: No hardware input devices found, using default")
            self.input_devices = [(None, "Default")]

        # Pick initial device
        self.audio_dev_idx = 0  # index into self.input_devices
        if audio_device is not None:
            try:
                dev_id = int(audio_device)
                for idx, (did, _) in enumerate(self.input_devices):
                    if did == dev_id:
                        self.audio_dev_idx = idx
                        break
            except ValueError:
                needle = audio_device.lower()
                for idx, (_, name) in enumerate(self.input_devices):
                    if needle in name.lower():
                        self.audio_dev_idx = idx
                        break
        else:
            # Default to Scarlett if available
            for idx, (_, name) in enumerate(self.input_devices):
                if "scarlett" in name.lower():
                    self.audio_dev_idx = idx
                    break

        self.audio_stream = None
        self.audio_stream_lock = threading.Lock()

        # State
        self.running = True
        self.recording = False
        self.record_start_time = None

        # Queues
        self.video_queue = Queue(maxsize=90)  # ~3s buffer at 30fps
        self.audio_queue = Queue(maxsize=200)

        # Recording state
        self.video_writer = None
        self.audio_chunks = []
        self.temp_video_path = None
        self.temp_audio_path = None
        self.output_path = None
        self.frame_count = 0

        # Latest frame for preview (shared between video thread and main)
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Toolbar buttons: (label, x1, x2) — y is always 0..TOOLBAR_H
        # Positions computed in _draw_toolbar based on current state
        self._pending_click = None

        # Point cloud visualization (Open3D, Kinect only)
        self.o3d_vis = None
        self.o3d_pcd = None
        self.o3d_initialized = False
        self.cloud_points = None
        self.cloud_colors = None
        self.cloud_lock = threading.Lock()
        self.cloud_ready = False

        # Background removal (depth-based, Kinect only)
        # Modes: "off", "green", "bokeh"
        self.bg_mode = "bokeh" if bokeh else "green" if bg_remove else "off"
        self.bg_threshold = 1200  # depth cutoff in mm

        # Audio level monitoring (always active, not just when recording)
        self.audio_level = 0.0  # peak amplitude 0..1
        self.audio_waveform = np.zeros(640, dtype=np.float32)  # downsampled waveform

        # Snapshot flash overlay
        self.snap_flash_until = 0  # monotonic time when flash fades

        # Skeleton overlay (MediaPipe Pose)
        self.skeleton_on = False
        self.pose = None  # lazily initialized MediaPipe Pose instance

        # Visual effects
        self.effect_idx = 0  # index into EFFECTS
        self.trail_buffer = None  # for trail/ghosting effect
        self.matrix_cols = None  # for matrix rain effect
        self.particles = None  # for particle dissolve effect
        self.prev_frame_gray = None  # for particle motion detection
        self.latest_depth = None  # raw depth for depth-powered effects
        self.depth_lock = threading.Lock()
        self.hue_offset = 0  # for hue_rotate effect
        self.color_channel = 0  # for color_isolate: 0=B, 1=G, 2=R
        self.film_scratches = []  # for old_film effect
        self.glitch_seed = 0  # for glitch effect

    @property
    def _current_video_source(self):
        return self.video_sources[self.video_src_idx]

    @property
    def _current_video_name(self):
        return self._current_video_source["name"]

    def _switch_video_source(self):
        """Cycle to next video source."""
        old_type = self._current_video_source["type"]
        # Close current webcam if open
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.webcam_cap = None
        # Close cloud viewer if leaving cloud mode
        if old_type == "kinect_cloud":
            self._close_cloud_viewer()
        self.video_src_idx = (self.video_src_idx + 1) % len(self.video_sources)
        src = self._current_video_source
        print(f"Video → {src['name']}")
        # Open webcam if switching to V4L2
        if src["type"] == "v4l2":
            self.webcam_cap = cv2.VideoCapture(src["device"])
            if not self.webcam_cap.isOpened():
                print(f"  Failed to open {src['device']}, skipping")
                self.webcam_cap = None
                self._switch_video_source()
        # Open cloud viewer if entering cloud mode
        elif src["type"] == "kinect_cloud":
            if not self._open_cloud_viewer():
                print("  Skipping cloud mode (open3d unavailable)")
                self._switch_video_source()

    @property
    def _current_audio_device(self):
        return self.input_devices[self.audio_dev_idx]

    @property
    def _current_audio_name(self):
        return self._current_audio_device[1]

    def _open_audio_stream(self):
        """Create and start an audio InputStream for the current device."""
        dev_id, dev_name = self._current_audio_device
        dev_info = sd.query_devices(dev_id)
        channels = min(AUDIO_CHANNELS, dev_info["max_input_channels"])
        samplerate = AUDIO_SAMPLERATE
        try:
            stream = sd.InputStream(
                device=dev_id,
                samplerate=samplerate,
                channels=channels,
                blocksize=AUDIO_BLOCKSIZE,
                dtype="float32",
                callback=self.audio_callback,
            )
            stream.start()
            return stream
        except sd.PortAudioError as e:
            print(f"Failed to open {dev_name}: {e}")
            return None

    def _switch_audio_device(self):
        """Cycle to next audio input device."""
        self.audio_dev_idx = (self.audio_dev_idx + 1) % len(self.input_devices)
        dev_id, dev_name = self._current_audio_device
        print(f"Audio → {dev_name}")
        with self.audio_stream_lock:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
            self.audio_stream = self._open_audio_stream()

    def _cycle_bg_mode(self):
        """Cycle depth-based background mode: off → green → bokeh → off."""
        if self._current_video_source["type"] != "kinect_rgb":
            print("BG modes only work with Kinect RGB source")
            return
        modes = ["off", "green", "bokeh"]
        idx = modes.index(self.bg_mode)
        self.bg_mode = modes[(idx + 1) % len(modes)]
        print(f"BG mode → {self.bg_mode} (threshold: {self.bg_threshold}mm)")

    def _toggle_skeleton(self):
        """Toggle MediaPipe Pose skeleton overlay."""
        global mp_pose, mp_drawing
        self.skeleton_on = not self.skeleton_on
        if self.skeleton_on and self.pose is None:
            try:
                import mediapipe as mp
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("Skeleton → ON")
            except ImportError:
                print("ERROR: mediapipe not installed — pip install mediapipe")
                self.skeleton_on = False
                return
        else:
            print(f"Skeleton → {'ON' if self.skeleton_on else 'OFF'}")

    def _draw_skeleton(self, frame):
        """Run pose estimation and draw skeleton on frame (in-place)."""
        if not self.skeleton_on or self.pose is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3,
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 220, 0), thickness=2,
                ),
            )

    # ── Visual Effects ────────────────────────────────────────────────────────

    def _cycle_effect(self):
        """Cycle to next visual effect."""
        self.effect_idx = (self.effect_idx + 1) % len(EFFECTS)
        name = EFFECTS[self.effect_idx]
        print(f"Effect → {name}")
        # Reset stateful effects when switching away
        if name == "off":
            self.trail_buffer = None
            self.matrix_cols = None
            self.particles = None

    def _apply_effect(self, frame):
        """Apply the current visual effect to a frame. Returns new frame."""
        name = EFFECTS[self.effect_idx]
        if name == "off":
            return frame
        method = getattr(self, f"_fx_{name}", None)
        if method is None:
            return frame
        try:
            return method(frame)
        except Exception:
            return frame

    def _fx_ascii(self, frame):
        """Render frame as colored ASCII characters."""
        h, w = frame.shape[:2]
        cell_w, cell_h = 6, 10
        cols, rows = w // cell_w, h // cell_h
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(frame)
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                region = gray[y1:y2, x1:x2]
                brightness = int(np.mean(region))
                char_idx = brightness * (len(ASCII_CHARS) - 1) // 255
                color_region = frame[y1:y2, x1:x2]
                avg_color = tuple(int(v) for v in np.mean(color_region, axis=(0, 1)))
                cv2.putText(out, ASCII_CHARS[char_idx], (x1, y2 - 2),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, avg_color, 1)
        return out

    def _fx_pixelate(self, frame):
        """Chunky pixel mosaic effect."""
        block = 10
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _fx_edge_glow(self, frame):
        """Neon edge glow — bright edges on black."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Dilate edges for glow
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Color the edges using the original frame colors
        out = np.zeros_like(frame)
        mask = edges > 0
        # Brighten edge pixels from original colors
        out[mask] = np.clip(frame[mask].astype(np.int16) + 80, 0, 255).astype(np.uint8)
        # Add glow via blur
        glow = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
        out = cv2.add(out, glow)
        return out

    def _fx_thermal(self, frame):
        """Thermal/predator vision — false color heat map."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use depth if available for true thermal-like effect
        with self.depth_lock:
            depth = self.latest_depth
        if depth is not None:
            depth_clipped = np.clip(depth, 0, 2047)
            # Invert so closer = hotter
            thermal = (255 - (depth_clipped / 2047.0 * 255)).astype(np.uint8)
        else:
            # Fallback: use luminance as "heat"
            thermal = gray
        return cv2.applyColorMap(thermal, cv2.COLORMAP_INFERNO)

    def _fx_trail(self, frame):
        """Motion trail / ghosting effect."""
        if self.trail_buffer is None:
            self.trail_buffer = frame.astype(np.float32)
        # Blend: 70% old trail, 30% new frame
        self.trail_buffer = self.trail_buffer * 0.7 + frame.astype(np.float32) * 0.3
        return self.trail_buffer.astype(np.uint8)

    def _fx_vhs(self, frame):
        """VHS glitch — scanlines, color bleed, tracking noise."""
        out = frame.copy()
        h, w = out.shape[:2]
        # Scanlines — darken every other row
        out[::2] = (out[::2].astype(np.int16) * 7 // 10).clip(0, 255).astype(np.uint8)
        # Color channel horizontal offset (chromatic aberration)
        shift = 3
        out[:, shift:, 2] = frame[:, :-shift, 2]   # red shift right
        out[:, :-shift, 0] = frame[:, shift:, 0]    # blue shift left
        # Random horizontal glitch bands
        t = int(time.monotonic() * 1000)
        rng = np.random.RandomState(t % 10000)
        for _ in range(rng.randint(1, 4)):
            y = rng.randint(0, h - 8)
            band_h = rng.randint(2, 8)
            offset = rng.randint(-15, 15)
            if offset != 0:
                band = out[y:y + band_h].copy()
                out[y:y + band_h] = np.roll(band, offset, axis=1)
        # Slight color tint (warm VHS look)
        tint = np.full_like(out, (15, 5, 25))
        out = cv2.add(out, tint)
        # Noise
        noise = rng.randint(0, 20, (h, w), dtype=np.uint8)
        noise_3ch = cv2.merge([noise, noise, noise])
        out = cv2.add(out, noise_3ch)
        return out

    def _fx_matrix(self, frame):
        """Matrix digital rain overlay."""
        h, w = frame.shape[:2]
        if self.matrix_cols is None:
            # Each column: current y position, speed, char
            n_cols = w // 10
            self.matrix_cols = []
            for i in range(n_cols):
                self.matrix_cols.append({
                    "x": i * 10,
                    "y": np.random.randint(-h, 0),
                    "speed": np.random.randint(8, 25),
                    "length": np.random.randint(8, 25),
                })
        # Dark green-tinted version of frame as background
        out = (frame * 0.15).astype(np.uint8)
        out[:, :, 1] = np.clip(out[:, :, 1].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        # Draw rain columns
        for col in self.matrix_cols:
            for j in range(col["length"]):
                cy = col["y"] - j * 14
                if 0 <= cy < h:
                    ch = chr(np.random.randint(0x30, 0x5B))
                    brightness = max(0, 255 - j * (255 // col["length"]))
                    color = (0, brightness, 0)
                    if j == 0:
                        color = (180, 255, 180)  # bright head
                    cv2.putText(out, ch, (col["x"], cy),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
            # Advance
            col["y"] += col["speed"]
            if col["y"] - col["length"] * 14 > h:
                col["y"] = np.random.randint(-h // 2, 0)
                col["speed"] = np.random.randint(8, 25)
                col["length"] = np.random.randint(8, 25)
        return out

    def _fx_kaleidoscope(self, frame):
        """Mirror frame into 4-way symmetric kaleidoscope."""
        h, w = frame.shape[:2]
        # Take top-left quadrant
        quad = frame[:h // 2, :w // 2]
        # Mirror horizontally and vertically
        top = np.hstack([quad, cv2.flip(quad, 1)])
        bottom = cv2.flip(top, 0)
        result = np.vstack([top, bottom])
        # Ensure exact size match
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h))
        return result

    def _fx_posterize(self, frame):
        """Pop art posterize — reduce to limited color palette."""
        levels = 4
        div = 256 // levels
        out = (frame // div) * div + div // 2
        # Boost saturation for pop-art look
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 80, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _fx_chroma_shift(self, frame):
        """Chromatic aberration — offset RGB channels."""
        out = np.zeros_like(frame)
        shift = 5
        # Blue stays centered, green shifts up-left, red shifts down-right
        out[:, :, 0] = frame[:, :, 0]  # blue
        out[:-shift, :-shift, 1] = frame[shift:, shift:, 1]  # green shift
        out[shift:, shift:, 2] = frame[:-shift, :-shift, 2]  # red shift
        return out

    def _fx_predator(self, frame):
        """Predator cloak — depth-based refraction/transparency."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            return frame
        # Foreground mask from depth
        fg_mask = ((depth > 0) & (depth < self.bg_threshold)).astype(np.float32)
        fg_mask = cv2.GaussianBlur(fg_mask, (15, 15), 0)
        # Create refraction by warping foreground pixels
        h, w = frame.shape[:2]
        # Distortion map based on depth edges
        edges = cv2.Canny((fg_mask * 255).astype(np.uint8), 50, 150)
        edges_f = cv2.GaussianBlur(edges.astype(np.float32), (11, 11), 0) / 255.0
        # Offset map
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        t = time.monotonic() * 3
        ripple = (np.sin(map_y * 0.1 + t) * 4 * edges_f).astype(np.float32)
        map_x_distort = (map_x + ripple).astype(np.float32)
        map_y_distort = (map_y + ripple * 0.5).astype(np.float32)
        warped = cv2.remap(frame, map_x_distort, map_y_distort, cv2.INTER_LINEAR)
        # Blend: foreground gets warped (semi-transparent), background stays
        fg_3ch = fg_mask[:, :, np.newaxis]
        out = (warped * fg_3ch * 0.4 + frame * (1.0 - fg_3ch * 0.6)).astype(np.uint8)
        # Edge highlight in cyan
        edge_mask = edges > 0
        out[edge_mask] = [255, 255, 0]
        return out

    def _fx_wireframe(self, frame):
        """Wireframe mesh — render depth as a green grid."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            # Fallback: edge wireframe from RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            out = np.zeros_like(frame)
            out[:, :, 1] = edges
            return out
        out = np.zeros_like(frame)
        h, w = out.shape[:2]
        step = 8
        depth_norm = np.clip(depth, 0, 2047).astype(np.float32) / 2047.0
        # Draw horizontal grid lines
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                d1 = depth_norm[y, x]
                d2 = depth_norm[y, x + step]
                if d1 > 0.01 and d2 > 0.01:
                    bright = int((1.0 - (d1 + d2) / 2) * 255)
                    cv2.line(out, (x, y), (x + step, y), (0, bright, 0), 1)
                # Vertical line
                d3 = depth_norm[min(y + step, h - 1), x]
                if d1 > 0.01 and d3 > 0.01:
                    bright = int((1.0 - (d1 + d3) / 2) * 255)
                    cv2.line(out, (x, y), (x, y + step), (0, bright, 0), 1)
        return out

    def _fx_particle(self, frame):
        """Particle dissolve — person breaks into particles at depth edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        # Detect motion or use depth edges for particle spawning
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray.copy()
            self.particles = []
        # Motion detection
        diff = cv2.absdiff(gray, self.prev_frame_gray)
        self.prev_frame_gray = gray.copy()
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Spawn new particles from motion edges
        spawn_points = np.argwhere(motion_mask > 0)
        if len(spawn_points) > 0:
            # Sample up to 50 new particles per frame
            indices = np.random.choice(len(spawn_points), min(50, len(spawn_points)), replace=False)
            for idx in indices:
                py, px = spawn_points[idx]
                color = tuple(int(v) for v in frame[py, px])
                self.particles.append({
                    "x": float(px), "y": float(py),
                    "vx": np.random.uniform(-2, 2),
                    "vy": np.random.uniform(-4, -1),
                    "life": np.random.randint(15, 40),
                    "color": color,
                    "size": np.random.randint(1, 4),
                })
        # Update and draw particles
        out = (frame * 0.4).astype(np.uint8)  # dim the base frame
        alive = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.15  # gravity
            p["life"] -= 1
            if p["life"] > 0 and 0 <= int(p["x"]) < w and 0 <= int(p["y"]) < h:
                alpha = p["life"] / 40.0
                color = tuple(int(v * alpha) for v in p["color"])
                cv2.circle(out, (int(p["x"]), int(p["y"])), p["size"], color, -1)
                alive.append(p)
        self.particles = alive[-3000:]  # cap at 3000 particles
        return out

    def _fx_silhouette(self, frame):
        """Solid color silhouette of foreground on gradient background."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            # Fallback: threshold on brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            mask_f = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
        else:
            mask_f = ((depth > 0) & (depth < self.bg_threshold)).astype(np.float32)
            mask_f = cv2.GaussianBlur(mask_f, (9, 9), 0)
        h, w = frame.shape[:2]
        # Gradient background: dark blue at top → purple at bottom
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h
            bg[y, :] = (int(60 * (1 - t) + 30 * t),
                        int(10 * (1 - t) + 10 * t),
                        int(80 * (1 - t) + 160 * t))
        # Foreground: solid warm color with slight edge glow
        fg_color = np.full((h, w, 3), (50, 180, 255), dtype=np.uint8)  # orange-amber
        # Edge glow
        edges = cv2.Canny((mask_f * 255).astype(np.uint8), 50, 150)
        glow = cv2.GaussianBlur(edges.astype(np.float32), (11, 11), 0) / 255.0
        mask_3ch = mask_f[:, :, np.newaxis]
        glow_3ch = glow[:, :, np.newaxis]
        out = (fg_color * mask_3ch + bg * (1.0 - mask_3ch)).astype(np.uint8)
        # Add white edge glow
        edge_color = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
        out = np.clip(out.astype(np.float32) + edge_color * glow_3ch * 0.8,
                      0, 255).astype(np.uint8)
        return out

    def _fx_xray(self, frame):
        """X-ray vision — inverted depth with bone-like contrast."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            # Fallback: invert grayscale with high contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            inv = cv2.equalizeHist(inv)
            return cv2.applyColorMap(inv, cv2.COLORMAP_BONE)
        # Real depth x-ray
        depth_clipped = np.clip(depth, 0, 2047).astype(np.float32)
        # Closer objects = brighter (like x-ray film)
        valid = depth_clipped > 0
        xray = np.zeros_like(depth_clipped)
        xray[valid] = 255.0 - (depth_clipped[valid] / 2047.0 * 255.0)
        xray = xray.astype(np.uint8)
        # Enhance contrast
        xray = cv2.equalizeHist(xray)
        # Apply bone colormap for that medical x-ray look
        colored = cv2.applyColorMap(xray, cv2.COLORMAP_BONE)
        # Darken areas with no depth data
        no_data = ~valid
        colored[no_data] = [0, 0, 0]
        return colored

    def _fx_depth_fog(self, frame):
        """Depth fog — distant objects fade into atmospheric haze."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            # Fallback: vertical gradient fog (top = foggy)
            h, w = frame.shape[:2]
            fog_strength = np.linspace(0.6, 0.0, h).reshape(-1, 1, 1).astype(np.float32)
            fog_color = np.full_like(frame, (220, 210, 200), dtype=np.uint8)
            out = (frame.astype(np.float32) * (1.0 - fog_strength) +
                   fog_color.astype(np.float32) * fog_strength)
            return out.astype(np.uint8)
        h, w = frame.shape[:2]
        depth_f = np.clip(depth, 0, 2047).astype(np.float32)
        # Fog ramps from 0 (close) to 1 (far)
        # Start fog at 30% of threshold, full fog at 2x threshold
        fog_near = self.bg_threshold * 0.3
        fog_far = self.bg_threshold * 2.0
        fog_amount = np.clip((depth_f - fog_near) / (fog_far - fog_near), 0, 1)
        # Zero depth (no data) gets full fog
        fog_amount[depth_f == 0] = 0.9
        fog_3ch = fog_amount[:, :, np.newaxis].astype(np.float32)
        # Cool bluish-white fog
        fog_color = np.full((h, w, 3), (230, 220, 200), dtype=np.uint8)
        out = (frame.astype(np.float32) * (1.0 - fog_3ch) +
               fog_color.astype(np.float32) * fog_3ch)
        return out.astype(np.uint8)

    def _fx_crt(self, frame):
        """CRT monitor — barrel distortion, scanlines, phosphor glow, vignette."""
        h, w = frame.shape[:2]
        # Barrel distortion (vectorized)
        cx, cy = w / 2, h / 2
        xs = np.arange(w, dtype=np.float32) - cx
        ys = np.arange(h, dtype=np.float32) - cy
        dx, dy = np.meshgrid(xs, ys)
        r2 = dx * dx + dy * dy
        k = -0.00015
        factor = 1 + k * r2
        map_x = (cx + dx * factor).astype(np.float32)
        map_y = (cy + dy * factor).astype(np.float32)
        out = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # Scanlines
        out[::3] = (out[::3].astype(np.int16) * 6 // 10).clip(0, 255).astype(np.uint8)
        # Phosphor RGB sub-pixel tint
        out[::1, 0::3, 2] = np.clip(out[::1, 0::3, 2].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        out[::1, 1::3, 1] = np.clip(out[::1, 1::3, 1].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        out[::1, 2::3, 0] = np.clip(out[::1, 2::3, 0].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        # Vignette
        Y, X = np.ogrid[:h, :w]
        vignette = 1.0 - 0.5 * (((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        vignette = np.clip(vignette, 0, 1).astype(np.float32)
        out = (out.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        # Rounded corners (black out corners)
        corner_r = 40
        for cy_, cx_ in [(0, 0), (0, w), (h, 0), (h, w)]:
            cv2.circle(out, (cx_, cy_), corner_r, (0, 0, 0), -1)
        return out

    def _fx_gameboy(self, frame):
        """Gameboy — 4-shade green palette at low resolution."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample to 160x144 (original Gameboy resolution)
        small = cv2.resize(gray, (160, 144), interpolation=cv2.INTER_AREA)
        # Quantize to 4 shades
        palette_gray = [15, 86, 172, 224]
        palette_bgr = [(48, 98, 15), (48, 115, 56), (139, 172, 48), (155, 188, 15)]
        out = np.zeros((144, 160, 3), dtype=np.uint8)
        for i, threshold in enumerate([0, 64, 128, 192]):
            next_t = 256 if i == 3 else [64, 128, 192, 256][i]
            mask = (small >= threshold) & (small < next_t)
            out[mask] = palette_bgr[i]
        # Scale up blocky to 640x480
        return cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)

    def _fx_old_film(self, frame):
        """Old film — scratches, dust, flicker, vignette, desaturated."""
        h, w = frame.shape[:2]
        # Desaturate
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Warm sepia tint
        out = out.astype(np.float32)
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.15 + 15, 0, 255)  # R boost
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.05, 0, 255)       # G slight
        out[:, :, 0] = np.clip(out[:, :, 0] * 0.85, 0, 255)       # B reduce
        out = out.astype(np.uint8)
        # Random flicker
        rng = np.random.RandomState(int(time.monotonic() * 30) % 100000)
        flicker = rng.uniform(0.85, 1.05)
        out = np.clip(out.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
        # Vertical scratches
        for _ in range(rng.randint(0, 4)):
            sx = rng.randint(0, w)
            brightness = rng.randint(180, 255)
            thickness = rng.randint(1, 2)
            cv2.line(out, (sx, 0), (sx, h), (brightness, brightness, brightness), thickness)
        # Dust spots
        for _ in range(rng.randint(0, 8)):
            dx, dy = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(1, 3)
            cv2.circle(out, (dx, dy), r, (200, 200, 200), -1)
        # Vignette
        cx, cy = w / 2, h / 2
        Y, X = np.ogrid[:h, :w]
        vignette = 1.0 - 0.6 * (((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        vignette = np.clip(vignette, 0, 1).astype(np.float32)
        out = (out.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
        return out

    def _fx_sepia(self, frame):
        """Warm sepia vintage tone."""
        # Sepia matrix transform
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ], dtype=np.float32)
        out = cv2.transform(frame, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _fx_negative(self, frame):
        """Color inversion."""
        return 255 - frame

    def _fx_solarize(self, frame):
        """Solarize — partial inversion at brightness threshold."""
        threshold = 128
        out = frame.copy()
        mask = out > threshold
        out[mask] = 255 - out[mask]
        # Boost saturation for visual punch
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _fx_emboss(self, frame):
        """3D emboss/relief effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]], dtype=np.float32)
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = np.clip(embossed + 128, 0, 255).astype(np.uint8)
        # Tint with original colors at low opacity
        emboss_bgr = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(emboss_bgr, 0.7, frame, 0.3, 0)

    def _fx_cartoon(self, frame):
        """Cartoon/cel-shade — edge lines + color quantization."""
        # Color quantization via bilateral filter
        color = frame.copy()
        for _ in range(3):
            color = cv2.bilateralFilter(color, 9, 75, 75)
        # Reduce colors
        div = 64
        color = (color // div) * div + div // 2
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 5)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Combine: color where edges are white, black edges on top
        return cv2.bitwise_and(color, edges_bgr)

    def _fx_glitch(self, frame):
        """Digital glitch — block corruption, channel swaps, JPEG artifacts."""
        h, w = frame.shape[:2]
        out = frame.copy()
        self.glitch_seed = (self.glitch_seed + 1) % 100000
        rng = np.random.RandomState(int(time.monotonic() * 15) % 100000)
        # Random block displacement
        n_blocks = rng.randint(3, 10)
        for _ in range(n_blocks):
            by = rng.randint(0, h - 30)
            bh = rng.randint(5, 30)
            bx_src = rng.randint(0, w - 50)
            bx_dst = rng.randint(0, w - 50)
            bw = rng.randint(20, min(100, w - max(bx_src, bx_dst)))
            out[by:by + bh, bx_dst:bx_dst + bw] = frame[by:by + bh, bx_src:bx_src + bw]
        # Channel swap on random bands
        for _ in range(rng.randint(1, 4)):
            by = rng.randint(0, h - 20)
            bh = rng.randint(3, 20)
            channels = list(out[by:by + bh].transpose(2, 0, 1))
            rng.shuffle(channels)
            out[by:by + bh] = np.stack(channels, axis=2).transpose(1, 2, 0)
        # Horizontal shift on random rows
        for _ in range(rng.randint(2, 8)):
            ry = rng.randint(0, h - 5)
            rh = rng.randint(1, 5)
            shift = rng.randint(-30, 30)
            if shift != 0:
                out[ry:ry + rh] = np.roll(out[ry:ry + rh], shift, axis=1)
        # Occasional full-frame color tint flash
        if rng.random() < 0.15:
            tint_ch = rng.randint(0, 3)
            out[:, :, tint_ch] = np.clip(
                out[:, :, tint_ch].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        return out

    def _fx_hue_rotate(self, frame):
        """Continuously shifting rainbow hue rotation."""
        self.hue_offset = (self.hue_offset + 3) % 180
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + self.hue_offset) % 180
        # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _fx_mirror(self, frame):
        """Horizontal mirror — left side reflected to right."""
        h, w = frame.shape[:2]
        left = frame[:, :w // 2]
        mirrored = np.hstack([left, cv2.flip(left, 1)])
        if mirrored.shape[1] != w:
            mirrored = cv2.resize(mirrored, (w, h))
        return mirrored

    def _fx_tilt_shift(self, frame):
        """Tilt-shift miniature effect — sharp center band, blurred top/bottom."""
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=10)
        # Create gradient mask: sharp in middle band, blurred at top/bottom
        mask = np.zeros((h, w), dtype=np.float32)
        center = h // 2
        band = h // 5  # sharp band height
        for y in range(h):
            dist = abs(y - center)
            if dist < band:
                mask[y, :] = 1.0
            else:
                mask[y, :] = max(0, 1.0 - (dist - band) / (h * 0.25))
        mask_3ch = mask[:, :, np.newaxis]
        out = (frame.astype(np.float32) * mask_3ch +
               blurred.astype(np.float32) * (1.0 - mask_3ch))
        # Boost saturation for toy-like look
        result = out.astype(np.uint8)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 50, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _fx_oil_paint(self, frame):
        """Oil painting — heavy bilateral filter for brush stroke look."""
        out = frame.copy()
        # Multiple passes of bilateral filter for strong painterly effect
        for _ in range(5):
            out = cv2.bilateralFilter(out, 9, 100, 100)
        # Slight edge enhancement for brush definition
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out = cv2.subtract(out, edges_bgr // 3)
        return out

    def _fx_color_isolate(self, frame):
        """Show one color channel, rest as grayscale. Cycles B→G→R each call."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Replace one channel with the original color data
        ch = self.color_channel
        out[:, :, ch] = frame[:, :, ch]
        # Auto-cycle channel slowly
        if int(time.monotonic() * 0.5) % 3 != self.color_channel:
            self.color_channel = int(time.monotonic() * 0.5) % 3
        return out

    def _fx_bass_pulse(self, frame):
        """Audio-reactive zoom pulse — frame scales with audio level."""
        level = self.audio_level
        # Map audio level to zoom amount (1.0 = no zoom, up to 1.15)
        zoom = 1.0 + level * 0.15
        h, w = frame.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        zoomed = cv2.resize(frame, (new_w, new_h))
        # Crop center to original size
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        out = zoomed[y_off:y_off + h, x_off:x_off + w]
        # Color tint on beats (high audio)
        if level > 0.5:
            intensity = (level - 0.5) * 2  # 0..1
            tint = np.zeros_like(out)
            tint[:, :, 2] = int(80 * intensity)  # red pulse
            tint[:, :, 0] = int(40 * intensity)  # slight blue
            out = cv2.add(out, tint)
        return out

    def _fx_audio_glow(self, frame):
        """Audio-reactive edge glow — edge brightness tied to audio peaks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Scale edge brightness with audio level
        level = max(self.audio_level, 0.1)  # minimum dim glow
        glow_strength = min(level * 2.5, 1.0)
        # Color based on audio level: green → yellow → red
        if level < 0.4:
            edge_color = np.array([0, 255, 0])
        elif level < 0.7:
            t = (level - 0.4) / 0.3
            edge_color = np.array([0, int(255 * (1 - t)), int(255 * t)])
        else:
            edge_color = np.array([0, 0, 255])
        out = (frame * 0.3).astype(np.uint8)  # dim base
        edge_layer = np.zeros_like(frame)
        edge_mask = edges > 0
        edge_layer[edge_mask] = edge_color
        # Glow via blur
        glow = cv2.GaussianBlur(edge_layer, (0, 0), sigmaX=4)
        out = cv2.add(out, (glow.astype(np.float32) * glow_strength).astype(np.uint8))
        out = cv2.add(out, (edge_layer.astype(np.float32) * glow_strength).astype(np.uint8))
        return out

    def _fx_depth_layers(self, frame):
        """Color-coded depth bands — near=red, mid=green, far=blue."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            # Fallback: tinted brightness bands
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out = np.zeros_like(frame)
            out[gray < 85, 2] = gray[gray < 85] * 3      # dark=red
            out[(gray >= 85) & (gray < 170), 1] = gray[(gray >= 85) & (gray < 170)]  # mid=green
            out[gray >= 170, 0] = gray[gray >= 170]       # bright=blue
            return out
        h, w = frame.shape[:2]
        depth_f = np.clip(depth, 0, 2047).astype(np.float32)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        # Define depth bands
        band1 = self.bg_threshold * 0.4   # near
        band2 = self.bg_threshold * 0.8   # mid-near
        band3 = self.bg_threshold * 1.2   # mid-far
        band4 = self.bg_threshold * 2.0   # far
        valid = depth_f > 0
        # Near: red
        near = valid & (depth_f < band1)
        intensity = np.clip((1.0 - depth_f / band1) * 255, 0, 255).astype(np.uint8)
        out[near, 2] = intensity[near]
        # Mid-near: yellow
        mid_near = valid & (depth_f >= band1) & (depth_f < band2)
        intensity = np.clip(200 - ((depth_f - band1) / (band2 - band1)) * 100, 100, 200).astype(np.uint8)
        out[mid_near, 2] = intensity[mid_near]
        out[mid_near, 1] = intensity[mid_near]
        # Mid-far: green
        mid_far = valid & (depth_f >= band2) & (depth_f < band3)
        intensity = np.clip(200 - ((depth_f - band2) / (band3 - band2)) * 100, 100, 200).astype(np.uint8)
        out[mid_far, 1] = intensity[mid_far]
        # Far: blue
        far = valid & (depth_f >= band3) & (depth_f < band4)
        intensity = np.clip(200 - ((depth_f - band3) / (band4 - band3)) * 100, 80, 200).astype(np.uint8)
        out[far, 0] = intensity[far]
        # Very far: dim purple
        very_far = valid & (depth_f >= band4)
        out[very_far, 0] = 60
        out[very_far, 2] = 40
        # Blend with dimmed original for context
        blended = cv2.addWeighted(out, 0.7, (frame * 0.3).astype(np.uint8), 1.0, 0)
        return blended

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the toolbar."""
        if event == cv2.EVENT_LBUTTONDOWN and y < TOOLBAR_H:
            self._pending_click = (x, y)
            self._pending_click = (x, y)

    def _draw_toolbar(self, display):
        """Draw toolbar with buttons on top of the display frame. Returns button rects."""
        bar = display[:TOOLBAR_H]
        bar[:] = (40, 40, 40)  # dark gray background

        buttons = []
        x = 8

        # Record button
        if self.recording:
            label, color = "STOP", (60, 60, 220)
        else:
            label, color = "REC", (50, 50, 180)
        bw = 80
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, label, (x + 12, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        buttons.append(("record", x, x + bw))
        x += bw + 12

        # Video source button
        vid_label = self._current_video_name
        if len(vid_label) > 14:
            vid_label = vid_label[:13] + ".."
        src_type = self._current_video_source["type"]
        if src_type == "kinect_depth":
            color = (160, 80, 0)
        elif src_type == "kinect_cloud":
            color = (160, 100, 0)
        elif src_type == "v4l2":
            color = (130, 0, 130)
        else:
            color = (0, 130, 0)
        bw = max(len(vid_label) * 10 + 16, 90)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, vid_label, (x + 8, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        buttons.append(("video", x, x + bw))
        x += bw + 12

        # Audio device button
        audio_label = self._current_audio_name
        # Truncate long names to fit
        if len(audio_label) > 14:
            audio_label = audio_label[:13] + ".."
        bw = max(len(audio_label) * 10 + 16, 90)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (100, 80, 40), -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, audio_label, (x + 8, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        buttons.append(("audio", x, x + bw))
        x += bw + 12

        # Format toggle button (MKV / MP4)
        fmt_label = "MP4" if self.mp4_mode else "MKV"
        color = (0, 120, 120) if self.mp4_mode else (90, 90, 90)
        bw = 60
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, fmt_label, (x + 8, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buttons.append(("format", x, x + bw))
        x += bw + 12

        # BG mode button
        bg_label = self.bg_mode.upper() if self.bg_mode != "off" else "BG"
        color = (180, 130, 0) if self.bg_mode == "green" else (180, 80, 50) if self.bg_mode == "bokeh" else (90, 90, 90)
        bw = 60
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, bg_label, (x + 8, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buttons.append(("bg", x, x + bw))
        x += bw + 12

        # Snapshot button
        bw = 60
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (130, 100, 0), -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, "SNAP", (x + 4, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buttons.append(("snap", x, x + bw))
        x += bw + 12

        # Skeleton button
        skel_label = "SKEL" if not self.skeleton_on else "SKEL"
        color = (0, 140, 0) if self.skeleton_on else (90, 90, 90)
        bw = 60
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, skel_label, (x + 4, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buttons.append(("skel", x, x + bw))
        x += bw + 12

        # FX button (visual effects)
        fx_name = EFFECTS[self.effect_idx]
        fx_label = fx_name[:5].upper() if fx_name != "off" else "FX"
        color = (180, 0, 180) if fx_name != "off" else (90, 90, 90)
        bw = 60
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), color, -1)
        cv2.rectangle(display, (x, 5), (x + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, fx_label, (x + 4, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buttons.append(("fx", x, x + bw))
        x += bw + 12

        # Quit button (right-aligned)
        bw = 70
        qx = 640 - bw - 8
        cv2.rectangle(display, (qx, 5), (qx + bw, TOOLBAR_H - 5), (80, 80, 80), -1)
        cv2.rectangle(display, (qx, 5), (qx + bw, TOOLBAR_H - 5), (200, 200, 200), 1)
        cv2.putText(display, "QUIT", (qx + 10, TOOLBAR_H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        buttons.append(("quit", qx, qx + bw))

        # Recording status
        if self.recording:
            elapsed = time.monotonic() - self.record_start_time
            status = f"REC {elapsed:.1f}s [{self.frame_count}f]"
            cv2.circle(display, (x + 10, TOOLBAR_H // 2), 6, (0, 0, 255), -1)
            cv2.putText(display, status, (x + 22, TOOLBAR_H - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            x += 160

        # VU meter — fill remaining space before quit button
        vu_x1 = x + 4
        vu_x2 = qx - 8
        if vu_x2 - vu_x1 > 30:
            vu_y1, vu_y2 = 10, TOOLBAR_H - 10
            # Background
            cv2.rectangle(display, (vu_x1, vu_y1), (vu_x2, vu_y2), (25, 25, 25), -1)
            cv2.rectangle(display, (vu_x1, vu_y1), (vu_x2, vu_y2), (80, 80, 80), 1)
            # Level fill
            level = min(self.audio_level, 1.0)
            fill_w = int((vu_x2 - vu_x1 - 2) * level)
            if fill_w > 0:
                # Green → yellow → red gradient
                if level < 0.6:
                    vu_color = (0, 200, 0)
                elif level < 0.85:
                    vu_color = (0, 200, 200)
                else:
                    vu_color = (0, 0, 220)
                cv2.rectangle(display, (vu_x1 + 1, vu_y1 + 1),
                              (vu_x1 + 1 + fill_w, vu_y2 - 1), vu_color, -1)

        return buttons

    def _handle_click(self, buttons):
        """Process a pending toolbar click."""
        if self._pending_click is None:
            return None
        cx, cy = self._pending_click
        self._pending_click = None
        for name, x1, x2 in buttons:
            if x1 <= cx <= x2:
                return name
        return None

    def _draw_waveform(self, display):
        """Draw audio waveform overlay on the bottom of the display."""
        wave_h = 60
        y_base = display.shape[0] - 5  # 5px from bottom
        y_top = y_base - wave_h
        y_mid = y_top + wave_h // 2

        # Semi-transparent background
        overlay = display[y_top:y_base, :].copy()
        cv2.rectangle(display, (0, y_top), (640, y_base), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display[y_top:y_base, :], 0.7, 0,
                        display[y_top:y_base, :])

        # Center line
        cv2.line(display, (0, y_mid), (640, y_mid), (60, 60, 60), 1)

        # Draw waveform as connected line segments
        waveform = self.audio_waveform
        half_h = wave_h // 2 - 2
        points = []
        for i in range(640):
            y = int(y_mid - waveform[i] * half_h)
            y = max(y_top + 1, min(y_base - 1, y))
            points.append((i, y))

        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 220, 0), 1, cv2.LINE_AA)

    def _depth_to_bgr(self, depth):
        """Convert 11-bit depth array to colorized BGR frame."""
        # Normalize to 8-bit, clip far values
        depth_clipped = np.clip(depth, 0, 2047)
        depth8 = (depth_clipped / 2047.0 * 255).astype(np.uint8)
        return cv2.applyColorMap(depth8, cv2.COLORMAP_JET)

    def _ensure_cloud_uv_grid(self):
        """Precompute subsampled pixel coordinate arrays for depth→world projection."""
        if hasattr(self, "_cloud_us"):
            return
        s = CLOUD_SUBSAMPLE
        vs, us = np.mgrid[0:480:s, 0:640:s]
        self._cloud_us = us.astype(np.float64).ravel()
        self._cloud_vs = vs.astype(np.float64).ravel()

    def _compute_cloud(self, depth_raw, rgb):
        """Convert raw depth + RGB into point cloud arrays. Called from video thread."""
        self._ensure_cloud_uv_grid()
        s = CLOUD_SUBSAMPLE
        depth_sub = depth_raw[::s, ::s].ravel().astype(np.float64)
        rgb_sub = rgb[::s, ::s].reshape(-1, 3)

        valid = (depth_sub > 0) & (depth_sub < 2047)
        d = depth_sub[valid]
        us = self._cloud_us[valid]
        vs = self._cloud_vs[valid]
        c = rgb_sub[valid]

        z = 1.0 / (d * -0.0030711016 + 3.3309495161)
        finite = (z > 0) & (z < 10.0)
        x = (us[finite] - KINECT_CX) * z[finite] / KINECT_FX
        y = (vs[finite] - KINECT_CY) * z[finite] / KINECT_FY

        points = np.column_stack((x, y, z[finite]))
        colors = c[finite].astype(np.float64) / 255.0

        with self.cloud_lock:
            self.cloud_points = points
            self.cloud_colors = colors
            self.cloud_ready = True

    def _open_cloud_viewer(self):
        """Create the Open3D visualizer window. Returns False if open3d unavailable."""
        global open3d
        if open3d is None:
            try:
                print("Loading Open3D...")
                import open3d as _open3d
                open3d = _open3d
            except ImportError:
                print("WARNING: open3d not installed — pip install open3d")
                return False
        try:
            self.o3d_vis = open3d.visualization.Visualizer()
            self.o3d_vis.create_window(
                window_name="Kinect Point Cloud",
                width=800, height=600, left=660, top=50,
            )
        except Exception as e:
            print(f"Failed to create Open3D window: {e}")
            self.o3d_vis = None
            return False
        self.o3d_pcd = open3d.geometry.PointCloud()
        self.o3d_initialized = False
        opt = self.o3d_vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        return True

    def _update_cloud_viewer(self):
        """Push latest cloud data to Open3D. Must be called from main thread."""
        if self.o3d_vis is None:
            return
        with self.cloud_lock:
            if not self.cloud_ready:
                alive = self.o3d_vis.poll_events()
                self.o3d_vis.update_renderer()
                if not alive:
                    print("Cloud viewer closed")
                    self._close_cloud_viewer()
                    self._switch_video_source()
                return
            points = self.cloud_points
            colors = self.cloud_colors
            self.cloud_ready = False

        if points is None or len(points) == 0:
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            return

        self.o3d_pcd.points = open3d.utility.Vector3dVector(points)
        self.o3d_pcd.colors = open3d.utility.Vector3dVector(colors)

        if not self.o3d_initialized:
            self.o3d_vis.add_geometry(self.o3d_pcd)
            ctr = self.o3d_vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_lookat([0, 0, 2])
            ctr.set_zoom(0.5)
            self.o3d_initialized = True
        else:
            self.o3d_vis.update_geometry(self.o3d_pcd)

        alive = self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
        if not alive:
            print("Cloud viewer closed")
            self._close_cloud_viewer()
            self._switch_video_source()

    def _close_cloud_viewer(self):
        """Destroy the Open3D visualizer window."""
        if self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
            self.o3d_vis = None
            self.o3d_pcd = None
            self.o3d_initialized = False
        with self.cloud_lock:
            self.cloud_points = None
            self.cloud_colors = None
            self.cloud_ready = False

    def _grab_frame(self):
        """Grab a frame from the current video source. Returns BGR 640x480 or None."""
        src = self._current_video_source
        if src["type"] == "kinect_rgb":
            result = freenect.sync_get_video()
            depth_result = freenect.sync_get_depth()
            if result is None:
                return None
            # Stash raw depth for depth-powered effects
            if depth_result is not None:
                with self.depth_lock:
                    self.latest_depth = depth_result[0].copy()
            bgr = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)
            if self.bg_mode != "off" and depth_result is not None:
                depth = depth_result[0]
                # Foreground: valid depth within threshold
                mask = ((depth > 0) & (depth < self.bg_threshold)).astype(np.uint8) * 255
                # Smooth mask edges
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                mask_f = mask.astype(np.float32) / 255.0
                mask_3ch = mask_f[:, :, np.newaxis]
                if self.bg_mode == "green":
                    bg = np.full_like(bgr, (0, 255, 0))
                else:  # bokeh
                    bg = cv2.GaussianBlur(bgr, (0, 0), sigmaX=15)
                bgr = (bgr * mask_3ch + bg * (1.0 - mask_3ch)).astype(np.uint8)
            return bgr
        elif src["type"] == "kinect_depth":
            # Also fetch video to keep both streams alive
            freenect.sync_get_video()
            result = freenect.sync_get_depth()
            if result is None:
                return None
            return self._depth_to_bgr(result[0])
        elif src["type"] == "kinect_cloud":
            rgb_result = freenect.sync_get_video()
            depth_result = freenect.sync_get_depth()
            if rgb_result is None or depth_result is None:
                return None
            self._compute_cloud(depth_result[0], rgb_result[0])
            return self._depth_to_bgr(depth_result[0])
        elif src["type"] == "v4l2":
            if self.webcam_cap is None or not self.webcam_cap.isOpened():
                time.sleep(0.03)
                return None
            ret, frame = self.webcam_cap.read()
            if not ret:
                return None
            # Resize to 640x480 to match Kinect
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            return frame
        return None

    def video_thread_func(self):
        """Poll current video source for frames."""
        while self.running:
            try:
                bgr = self._grab_frame()
                if bgr is None:
                    time.sleep(0.03)
                    continue
                # Update preview frame
                with self.frame_lock:
                    self.latest_frame = bgr
                # Push to recording queue if recording
                if self.recording:
                    try:
                        self.video_queue.put_nowait(bgr)
                    except Exception:
                        pass  # drop frame if queue full
            except Exception as e:
                if self.running:
                    print(f"Video: {e}")
                    time.sleep(0.5)  # back off and retry

    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice input callback — pushes audio chunks to queue."""
        if status:
            print(f"Audio: {status}", file=sys.stderr)
        # Update level and waveform for visualization (always active)
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        self.audio_level = float(np.max(np.abs(mono)))
        # Downsample to 640 points for waveform display
        n = len(mono)
        if n >= 640:
            step = n // 640
            self.audio_waveform = mono[:step * 640:step].copy()
        else:
            # Stretch short buffer to fill 640
            indices = np.linspace(0, n - 1, 640).astype(int)
            self.audio_waveform = mono[indices].copy()
        if self.recording:
            try:
                self.audio_queue.put_nowait(indata.copy())
            except Exception:
                pass  # drop if queue full

    def writer_thread_func(self):
        """Pull from queues and write video frames / accumulate audio."""
        while self.running:
            if not self.recording:
                time.sleep(0.01)
                continue

            # Drain video queue
            try:
                frame = self.video_queue.get(timeout=0.03)
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1
            except Empty:
                pass

            # Drain audio queue
            drained = 0
            while drained < 10:
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.audio_chunks.append(chunk)
                    drained += 1
                except Empty:
                    break

    def take_snapshot(self):
        """Save the current preview frame as a PNG."""
        with self.frame_lock:
            frame = self.latest_frame
        if frame is None:
            print("No frame available for snapshot")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.output_dir / f"snap_{ts}.png"
        cv2.imwrite(str(path), frame)
        size_kb = path.stat().st_size / 1024
        print(f"Snapshot: {path} ({size_kb:.0f} KB)")
        self.snap_flash_until = time.monotonic() + 0.15

    def _begin_countdown(self):
        """Start the countdown timer. Recording begins when it expires."""
        if self.countdown <= 0:
            self.start_recording()
            return
        self.countdown_end = time.monotonic() + self.countdown
        print(f"Recording in {self.countdown}...")

    def _draw_countdown(self, display):
        """Draw countdown number centered on the video frame."""
        if self.countdown_end is None:
            return
        remaining = self.countdown_end - time.monotonic()
        if remaining <= 0:
            return
        num = str(int(remaining) + 1)
        # Large centered text with dark outline
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 4.0
        thickness = 8
        (tw, th), _ = cv2.getTextSize(num, font, scale, thickness)
        cx = (640 - tw) // 2
        cy = (TOOLBAR_H + display.shape[0]) // 2 + th // 2
        cv2.putText(display, num, (cx, cy), font, scale, (0, 0, 0), thickness + 4)
        cv2.putText(display, num, (cx, cy), font, scale, (255, 255, 255), thickness)

    def start_recording(self):
        """Begin a new recording."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "mp4" if self.mp4_mode else "mkv"
        self.temp_video_path = str(self.output_dir / f".tmp_video_{ts}.avi")
        self.temp_audio_path = str(self.output_dir / f".tmp_audio_{ts}.wav")
        self.output_path = str(self.output_dir / f"capture_{ts}.{ext}")

        # Video writer — FFV1 lossless in AVI container (temp)
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path, fourcc, TARGET_FPS, (640, 480)
        )
        if not self.video_writer.isOpened():
            # Fallback to MJPEG
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(
                self.temp_video_path, fourcc, TARGET_FPS, (640, 480)
            )

        self.audio_chunks = []
        self.frame_count = 0

        # Clear queues
        while not self.video_queue.empty():
            try:
                self.video_queue.get_nowait()
            except Empty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

        self.record_start_time = time.monotonic()
        self.recording = True
        print(f"Recording started → {self.output_path}")

    def stop_recording(self):
        """Stop recording and mux audio+video."""
        self.recording = False
        time.sleep(0.1)  # let writer drain

        # Drain remaining frames
        while not self.video_queue.empty():
            try:
                frame = self.video_queue.get_nowait()
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1
            except Empty:
                break

        # Drain remaining audio
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                self.audio_chunks.append(chunk)
            except Empty:
                break

        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        print(f"Recording stopped: {self.frame_count} frames")

        # Write audio WAV
        if self.audio_chunks:
            import wave

            audio_data = np.concatenate(self.audio_chunks, axis=0)
            # Convert float32 → int16
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            with wave.open(self.temp_audio_path, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLERATE)
                wf.writeframes(audio_int16.tobytes())
            print(f"Audio: {len(audio_data)} samples ({len(audio_data)/AUDIO_SAMPLERATE:.1f}s)")

        # Mux with ffmpeg
        self._mux()

        # Clean temp files
        for p in [self.temp_video_path, self.temp_audio_path]:
            if p and os.path.exists(p):
                os.unlink(p)

    def _mux(self):
        """Mux temp video + audio into final MKV using ffmpeg."""
        if not os.path.exists(self.temp_video_path):
            print("ERROR: temp video file missing")
            return

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
        cmd += ["-i", self.temp_video_path]

        has_audio = self.audio_chunks and os.path.exists(self.temp_audio_path)
        if has_audio:
            cmd += ["-i", self.temp_audio_path]

        if self.mp4_mode:
            cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "copy"]

        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]

        cmd += [self.output_path]

        print("Muxing with ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            print(f"Saved: {self.output_path} ({size_mb:.1f} MB)")

    def run(self):
        """Main entry point."""
        print("Kinect RGB + Scarlett Audio Capture")
        print("  R = record | S = snap | K = skeleton | E = effects | V = video | A = audio | Q = quit")
        print("  F = format | B = bg mode | [ / ] = depth threshold")
        print(f"  Video: {', '.join(s['name'] for s in self.video_sources)}")
        print(f"  Audio: {', '.join(n for _, n in self.input_devices)}")
        print(f"  Active: {self._current_video_name} + {self._current_audio_name}")
        if self.duration:
            print(f"  Auto-record for {self.duration}s then exit")
        print()
        auto_start_pending = self.duration is not None

        # Start audio stream
        self.audio_stream = self._open_audio_stream()

        # Start threads
        video_thread = threading.Thread(target=self.video_thread_func, daemon=True)
        writer_thread = threading.Thread(target=self.writer_thread_func, daemon=True)
        video_thread.start()
        writer_thread.start()

        cv2.namedWindow("Kinect Preview", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Kinect Preview", 640, 480 + TOOLBAR_H)
        cv2.setMouseCallback("Kinect Preview", self._mouse_callback)

        try:
            while self.running:
                # Get latest frame for preview
                with self.frame_lock:
                    frame = self.latest_frame

                if frame is not None:
                    # Auto-start recording once first frame arrives
                    if auto_start_pending:
                        auto_start_pending = False
                        self._begin_countdown()

                    # Check if countdown finished
                    if (
                        self.countdown_end is not None
                        and time.monotonic() >= self.countdown_end
                    ):
                        self.countdown_end = None
                        self.start_recording()

                    # Apply visual effect
                    frame = self._apply_effect(frame)

                    # Skeleton overlay (runs pose estimation on frame)
                    self._draw_skeleton(frame)

                    # Build display with toolbar above the video
                    toolbar = np.zeros((TOOLBAR_H, 640, 3), dtype=np.uint8)
                    display = np.vstack([toolbar, frame])
                    buttons = self._draw_toolbar(display)

                    # Countdown overlay
                    self._draw_countdown(display)

                    # Waveform overlay on bottom of video
                    self._draw_waveform(display)

                    # Snapshot flash
                    if time.monotonic() < self.snap_flash_until:
                        alpha = (self.snap_flash_until - time.monotonic()) / 0.15
                        white = np.full_like(display, 255)
                        cv2.addWeighted(white, alpha * 0.6, display, 1.0, 0, display)

                    cv2.imshow("Kinect Preview", display)

                    # Update Open3D cloud viewer if active
                    if self._current_video_source["type"] == "kinect_cloud":
                        self._update_cloud_viewer()

                # Handle toolbar clicks
                action = self._handle_click(buttons if frame is not None else [])
                if action == "record":
                    if self.recording:
                        self.stop_recording()
                    elif self.countdown_end is not None:
                        self.countdown_end = None
                        print("Countdown cancelled")
                    else:
                        self._begin_countdown()
                elif action == "video":
                    self._switch_video_source()
                elif action == "audio":
                    self._switch_audio_device()
                elif action == "format":
                    if not self.recording:
                        self.mp4_mode = not self.mp4_mode
                        print(f"Format → {'MP4' if self.mp4_mode else 'MKV'}")
                elif action == "bg":
                    self._cycle_bg_mode()
                elif action == "snap":
                    self.take_snapshot()
                elif action == "skel":
                    self._toggle_skeleton()
                elif action == "fx":
                    self._cycle_effect()
                elif action == "quit":
                    if self.recording:
                        self.stop_recording()
                    self.running = False
                    break

                # Handle keyboard shortcuts
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    if self.recording:
                        self.stop_recording()
                    self.running = False
                    break

                elif key == ord("r") or key == ord("R"):
                    if self.recording:
                        self.stop_recording()
                    elif self.countdown_end is not None:
                        self.countdown_end = None
                        print("Countdown cancelled")
                    else:
                        self._begin_countdown()

                elif key == ord("v") or key == ord("V"):
                    self._switch_video_source()

                elif key == ord("a") or key == ord("A"):
                    self._switch_audio_device()

                elif key == ord("f") or key == ord("F"):
                    if not self.recording:
                        self.mp4_mode = not self.mp4_mode
                        print(f"Format → {'MP4' if self.mp4_mode else 'MKV'}")

                elif key == ord("b") or key == ord("B"):
                    self._cycle_bg_mode()

                elif key == ord("s") or key == ord("S"):
                    self.take_snapshot()

                elif key == ord("k") or key == ord("K"):
                    self._toggle_skeleton()

                elif key == ord("e") or key == ord("E"):
                    self._cycle_effect()

                elif key == ord("["):
                    self.bg_threshold = max(100, self.bg_threshold - 100)
                    print(f"BG threshold → {self.bg_threshold}mm")

                elif key == ord("]"):
                    self.bg_threshold = min(2047, self.bg_threshold + 100)
                    print(f"BG threshold → {self.bg_threshold}mm")

                # Auto-stop by duration
                if (
                    self.recording
                    and self.duration
                    and (time.monotonic() - self.record_start_time) >= self.duration
                ):
                    print(f"Duration limit ({self.duration}s) reached")
                    self.stop_recording()
                    self.running = False
                    break

        except KeyboardInterrupt:
            if self.recording:
                self.stop_recording()
        finally:
            self.running = False
            self._close_cloud_viewer()
            if self.pose is not None:
                self.pose.close()
            with self.audio_stream_lock:
                if self.audio_stream is not None:
                    self.audio_stream.stop()
                    self.audio_stream.close()
            if self.webcam_cap is not None:
                self.webcam_cap.release()
            if self.has_kinect:
                freenect.sync_stop()
            cv2.destroyAllWindows()
            print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Kinect RGB + Scarlett Audio Capture")
    parser.add_argument("--duration", type=float, help="Auto-stop after N seconds")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Countdown seconds before recording (default: 3, 0 to disable)")
    parser.add_argument("--mp4", action="store_true",
                        help="Output H.264 MP4 instead of lossless MKV")
    parser.add_argument("--output", default=".", help="Output directory (default: .)")
    parser.add_argument("--audio-device", default=None,
                        help="sounddevice device name/index (default: auto-detect Scarlett)")
    parser.add_argument("--bg-remove", action="store_true",
                        help="Enable depth-based background removal (Kinect RGB only)")
    parser.add_argument("--bokeh", action="store_true",
                        help="Enable depth-based bokeh blur (Kinect RGB only)")
    args = parser.parse_args()

    cap = KinectCapture(
        output_dir=args.output,
        audio_device=args.audio_device,
        duration=args.duration,
        countdown=args.countdown,
        mp4=args.mp4,
        bg_remove=args.bg_remove,
        bokeh=args.bokeh,
    )
    cap.run()


if __name__ == "__main__":
    main()
