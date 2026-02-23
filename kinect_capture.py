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

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the toolbar."""
        if event == cv2.EVENT_LBUTTONDOWN and y < TOOLBAR_H:
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
        print("  R = record | S = snap | K = skeleton | V = video | A = audio | F = fmt | B = bg | Q = quit")
        print("  [ / ] = adjust BG depth threshold")
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
