#!/usr/bin/env python3
"""Kinect RGB + Scarlett Audio capture tool.

Live preview from Xbox 360 Kinect or webcam with audio recording.
Press R to record, V to cycle video source, A to cycle audio device, Q to quit.
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

freenect = None  # lazy import — only when Kinect is connected

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FPS = 30
AUDIO_SAMPLERATE = 48000
AUDIO_CHANNELS = 2
AUDIO_BLOCKSIZE = 1024
TOOLBAR_H = 40


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
    Types: 'kinect_rgb', 'kinect_depth', 'v4l2'
    """
    sources = []
    if _kinect_available():
        sources.append({"type": "kinect_rgb", "name": "Kinect RGB"})
        sources.append({"type": "kinect_depth", "name": "Kinect Depth"})
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
    def __init__(self, output_dir=".", audio_device=None, duration=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration

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

        # Audio level monitoring (always active, not just when recording)
        self.audio_level = 0.0  # peak amplitude 0..1
        self.audio_waveform = np.zeros(640, dtype=np.float32)  # downsampled waveform

    @property
    def _current_video_source(self):
        return self.video_sources[self.video_src_idx]

    @property
    def _current_video_name(self):
        return self._current_video_source["name"]

    def _switch_video_source(self):
        """Cycle to next video source."""
        # Close current webcam if open
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.webcam_cap = None
        self.video_src_idx = (self.video_src_idx + 1) % len(self.video_sources)
        src = self._current_video_source
        print(f"Video → {src['name']}")
        # Open webcam if switching to V4L2
        if src["type"] == "v4l2":
            self.webcam_cap = cv2.VideoCapture(src["device"])
            if not self.webcam_cap.isOpened():
                print(f"  Failed to open {src['device']}, skipping")
                self.webcam_cap = None
                # Skip to next source
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

    def _grab_frame(self):
        """Grab a frame from the current video source. Returns BGR 640x480 or None."""
        src = self._current_video_source
        if src["type"] == "kinect_rgb":
            result = freenect.sync_get_video()
            # Also fetch depth to keep both streams alive for smooth switching
            freenect.sync_get_depth()
            if result is None:
                return None
            return cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)
        elif src["type"] == "kinect_depth":
            # Also fetch video to keep both streams alive
            freenect.sync_get_video()
            result = freenect.sync_get_depth()
            if result is None:
                return None
            return self._depth_to_bgr(result[0])
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

    def start_recording(self):
        """Begin a new recording."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_video_path = str(self.output_dir / f".tmp_video_{ts}.avi")
        self.temp_audio_path = str(self.output_dir / f".tmp_audio_{ts}.wav")
        self.output_path = str(self.output_dir / f"capture_{ts}.mkv")

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

        if self.audio_chunks and os.path.exists(self.temp_audio_path):
            cmd += ["-i", self.temp_audio_path]
            cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-c:v", "copy"]

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
        print("  R = record | V = video source | A = audio device | Q = quit")
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
                        self.start_recording()

                    # Build display with toolbar above the video
                    toolbar = np.zeros((TOOLBAR_H, 640, 3), dtype=np.uint8)
                    display = np.vstack([toolbar, frame])
                    buttons = self._draw_toolbar(display)

                    # Waveform overlay on bottom of video
                    self._draw_waveform(display)

                    cv2.imshow("Kinect Preview", display)

                # Handle toolbar clicks
                action = self._handle_click(buttons if frame is not None else [])
                if action == "record":
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif action == "video":
                    self._switch_video_source()
                elif action == "audio":
                    self._switch_audio_device()
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
                    else:
                        self.start_recording()

                elif key == ord("v") or key == ord("V"):
                    self._switch_video_source()

                elif key == ord("a") or key == ord("A"):
                    self._switch_audio_device()

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
    parser.add_argument("--output", default=".", help="Output directory (default: .)")
    parser.add_argument("--audio-device", default=None,
                        help="sounddevice device name/index (default: auto-detect Scarlett)")
    args = parser.parse_args()

    cap = KinectCapture(
        output_dir=args.output,
        audio_device=args.audio_device,
        duration=args.duration,
    )
    cap.run()


if __name__ == "__main__":
    main()
