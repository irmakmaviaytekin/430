#!/usr/bin/env python3
"""
loop_test.py — single-command smoke test for the bullseye tracking / piezo
steering closed loop.

What it does, every cycle:
    1. grab a frame from the camera
    2. detect the concentric-ring "bullseye" target, get its sub-pixel centroid
    3. compute pixel error against the chosen setpoint (default: frame center)
    4. run an incremental (velocity-form) PID per pixel axis
    5. convert the pixel-domain correction into a voltage delta via the
       inverse response matrix (Jacobian), clamp it, and send it to the
       MDT693B over serial
    6. log everything to CSV and (optionally) show a live debug window

USAGE (single command, sensible defaults):
    python loop_test.py --port COM5

Dry run with no hardware attached (serial calls are just printed):
    python loop_test.py --dry-run --show

IMPORTANT — read before running on real hardware:
  * The default response matrix below is a PLACEHOLDER (small, conservative
    gain). It is NOT a calibrated Jacobian. Replace it with a measured
    matrix (see --matrix) before trusting closed-loop behavior.
  * --max-step-volts is an independent hard safety limit on how much any
    single axis can move per control cycle, regardless of gains or matrix
    errors. Keep it small until you've verified behavior.
  * Serial command syntax for the MDT693B is written to match the
    documented "<axis>voltage=<value>" / "<axis>voltage?" command family
    (Thorlabs MDT693B/694B remote command set, Chapter 7 of the manual).
    Confirm this against your unit's firmware before relying on it —
    minor syntax differences exist across firmware revisions.
"""

# Must be set BEFORE numpy/cv2/etc. are imported. Some Anaconda/Windows
# environments end up with two different libraries (commonly numpy/opencv's
# bundled MKL runtime and the FlyCapture2 SDK's own bundled runtime) each
# linking their own copy of Intel's OpenMP runtime (libiomp5md.dll). When
# that happens the *second* one to initialize aborts the whole process
# ("OMP: Error #15") — which is what killed the Spyder kernel. This env var
# tells OpenMP to tolerate the duplicate rather than hard-abort. For the
# lightweight 2x2 matrix math this script does, the accuracy/performance
# risk mentioned in Intel's own warning is negligible — this is the
# standard, widely-used workaround for exactly this situation.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


# ==========================================================================
# SETTINGS — edit these directly and press Run (F5) in Spyder.
#
# This file always uses the values below, regardless of how it's launched —
# command-line flags are intentionally ignored (see the parse_args([]) call
# in main()), so there's no ambiguity between "what's in this file" and
# "what Spyder happened to put in sys.argv".
# ==========================================================================

# --- hardware connection ---
PORT = "COM3"                  # MDT693B serial port, e.g. "COM3" or "/dev/ttyUSB0"
BAUD = 115200
CAMERA_BACKEND = "flycapture"  # "opencv" for a webcam, "flycapture" for the Firefly MV
CAMERA_INDEX = 0               # camera index (meaning depends on CAMERA_BACKEND)
DRY_RUN = False                # True = print serial commands instead of sending them (no hardware needed)

# --- run parameters ---
DURATION_S = 30.0              # seconds to run the tracking loop
TARGET_X = None                # setpoint px; None = frame center
TARGET_Y = None                # setpoint px; None = frame center
SHOW_WINDOW = True             # show the live OpenCV debug window
LOG_PATH = None                # CSV log path; None = auto-named with a timestamp

# --- voltage safety guardrails (hard limits, enforced in two places) ---
V_INIT = 75.0                  # starting voltage on x/y axes
V_MIN = 5.0
V_MAX = 145.0
MAX_STEP_VOLTS = 5.0           # hard cap on voltage change per axis per control cycle

# --- calibration (auto-computes the response matrix at startup) ---
CALIBRATE = True               # False = load MATRIX_PATH instead of recalibrating
MATRIX_PATH = None             # used only when CALIBRATE = False; path to a saved calib_matrix_*.json
CALIB_AMPLITUDE_V = 25.0       # volts to perturb each axis by during calibration
CALIB_SAMPLES = 9              # number of test points per axis during calibration
CALIB_SETTLE_S = 0.2           # seconds to wait after each voltage change before reading the camera
CALIB_SAVE_PATH = None         # where to save the computed matrix; None = auto-named with a timestamp

# --- feedforward (one calibrated jump toward the target before the loop starts) ---
FEEDFORWARD = True
FEEDFORWARD_MAX_VOLTS = 30.0   # cap on the one-shot jump per axis

# --- controller ("direct" recommended; "pid" kept only for comparison) ---
CONTROLLER = "direct"          # "direct" or "pid"
GAIN = 0.6                     # direct mode: fraction of the fully-corrective voltage applied per cycle
DEADBAND_PX = 0.5              # direct mode: no correction applied within this many px of the target
KP = 0.6                       # pid mode only
KI = 0.05                      # pid mode only
KD = 0.02                      # pid mode only

# --- bullseye detection strictness ---
MIN_RINGS = 3                  # minimum concentric circles required to count as a valid detection
MIN_RADIUS_SPREAD_PX = 10.0    # minimum radius spread within a cluster to count as valid

# --- centroid smoothing (temporal EMA applied to the detected centroid, ---
# --- before computing pixel error / driving the controller) ---
EMA_ALPHA = 0.5                 # 1.0 = no smoothing (raw detection each frame); lower = more
                                # smoothing (and more lag). filtered = alpha*new + (1-alpha)*filtered


# --------------------------------------------------------------------------
# MDT693B serial interface
# --------------------------------------------------------------------------

class MDT693BController:
    """
    Thin wrapper around the MDT693B ASCII serial command set.

    Uses "fire and forget" writes (no read-back / verification) in the hot
    loop, since the vision feedback loop is what actually verifies motion.
    Call get_voltage() only outside the control loop (e.g. at startup) if
    you want a verified read.
    """

    AXES = ("x", "y", "z")

    def __init__(self, port, baud=115200, timeout=0.05, dry_run=False, vmin=5.0, vmax=145.0):
        self.dry_run = dry_run or serial is None
        self.port_name = port
        self.vmin = vmin
        self.vmax = vmax
        if self.dry_run:
            self._ser = None
            if serial is None:
                print("[MDT693B] pyserial not installed — forcing dry-run mode.")
            else:
                print(f"[MDT693B] DRY RUN — would open {port} @ {baud} baud")
            return

        self._ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # give the virtual COM port a moment to settle after opening
        time.sleep(0.2)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # The MDT693B replies with '*' after every accepted command (per
        # the manual's Chapter 7), not just queries. Since set_voltage()
        # below never reads those replies, they'd otherwise pile up in the
        # OS receive buffer over a long run and eventually cause the port
        # to misbehave. Disable echo mode (reduces reply volume) and do an
        # initial buffer clear; set_voltage() clears it again after every
        # write so it can never accumulate.
        self._ser.write(b"echo=0\r")
        time.sleep(0.05)
        self._ser.reset_input_buffer()

    def set_voltage(self, axis, volts):
        assert axis in self.AXES, f"unknown axis {axis!r}"
        # Hard safety clamp — deliberately re-applied here, independent of
        # any clamping done by the caller, so a bug or a future code path
        # that calls set_voltage() directly can never exceed these bounds.
        clamped = float(np.clip(volts, self.vmin, self.vmax))
        if abs(clamped - volts) > 1e-6:
            print(f"[MDT693B] WARNING: requested {axis}={volts:.3f}V is outside "
                  f"[{self.vmin:.1f}, {self.vmax:.1f}]V — clamped to {clamped:.3f}V")
        cmd = f"{axis}voltage={clamped:.3f}\r"
        if self.dry_run:
            print(f"[MDT693B] >> {cmd.strip()}")
            return
        self._ser.write(cmd.encode("ascii"))
        # Drain the '*' ready-prompt reply immediately so it never
        # accumulates across a long run. Non-blocking — reset_input_buffer
        # just discards whatever the OS has already received; we are not
        # waiting for the reply to arrive.
        self._ser.reset_input_buffer()

    def get_voltage(self, axis):
        assert axis in self.AXES, f"unknown axis {axis!r}"
        if self.dry_run:
            print(f"[MDT693B] >> {axis}voltage?  (dry run, returning 0.0)")
            return 0.0
        self._ser.reset_input_buffer()
        self._ser.write(f"{axis}voltage?\r".encode("ascii"))
        time.sleep(0.01)
        resp = self._ser.read(64).decode("ascii", errors="ignore")
        digits = "".join(c for c in resp if c.isdigit() or c in ".-")
        try:
            return float(digits)
        except ValueError:
            raise RuntimeError(f"could not parse voltage response: {resp!r}")

    def close(self):
        if not self.dry_run and self._ser is not None:
            self._ser.close()


# --------------------------------------------------------------------------
# Camera sources
# --------------------------------------------------------------------------

class OpenCVCameraSource:
    """Plain UVC webcam via OpenCV's VideoCapture."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

import ctypes
import os
import atexit
import time
import numpy as np
import cv2

class FlyCaptureSource:
    """
    FlyCapture2 C-API wrapper configured to swallow FC2_ERROR_TIMEOUT (err 41)
    and retry frame retrieval without dropping the stream.
    """

    def __init__(self, index=0):
        dll_dir = r"C:\Program Files\Point Grey Research\FlyCapture2\bin64\vs2015"
        dll_path = os.path.join(dll_dir, "FlyCapture2_C_v140.dll")

        if not os.path.exists(dll_path):
            raise RuntimeError(f"DLL not found at: {dll_path}")

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
            os.add_dll_directory(r"C:\Program Files\Point Grey Research\FlyCapture2\bin64")

        try:
            self.fc2 = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load {dll_path}: {e}") from e

        self.context = ctypes.c_void_p()
        atexit.register(self.release)

        # 64-bit signature declarations
        self.fc2.fc2CreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.fc2.fc2GetCameraFromIndex.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        self.fc2.fc2Connect.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.fc2.fc2StartCapture.argtypes = [ctypes.c_void_p]
        self.fc2.fc2CreateImage.argtypes = [ctypes.c_void_p]
        self.fc2.fc2RetrieveBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.fc2.fc2DestroyImage.argtypes = [ctypes.c_void_p]

        err = self.fc2.fc2CreateContext(ctypes.byref(self.context))
        if err != 0:
            raise RuntimeError(f"fc2CreateContext failed with code {err}")

        guid = (ctypes.c_uint32 * 4)()
        err = self.fc2.fc2GetCameraFromIndex(self.context, index, ctypes.byref(guid))
        if err != 0:
            raise RuntimeError(f"No FlyCapture camera found at index {index} (Error code {err}).")

        self.fc2.fc2StopCapture(self.context)
        self.fc2.fc2Disconnect(self.context)

        err = self.fc2.fc2Connect(self.context, ctypes.byref(guid))
        if err != 0:
            raise RuntimeError(f"Failed to connect to camera (Error code {err}).")

        err = self.fc2.fc2StartCapture(self.context)
        if err != 0:
            raise RuntimeError(f"Failed to start capture (Error code {err}).")

        self._opened = True

        # Warm-up check: consume until we get at least 1 valid frame (err == 0)
        valid_frames = 0
        for _ in range(15):
            ret, _ = self.read()
            if ret:
                valid_frames += 1
                if valid_frames >= 2:
                    break
            time.sleep(0.03)

        if valid_frames == 0:
            self.release()
            raise RuntimeError("Camera connected, but timed out continuously during initialization.")

    def isOpened(self):
        return getattr(self, "_opened", False) and self.context is not None

    def read(self, retries=5):
        if not self.isOpened():
            return False, None

        class Fc2Image(ctypes.Structure):
            _fields_ = [
                ("rows", ctypes.c_uint),
                ("cols", ctypes.c_uint),
                ("stride", ctypes.c_uint),
                ("pData", ctypes.c_void_p),
                ("dataSize", ctypes.c_uint),
                ("receivedDataSize", ctypes.c_uint),
                ("format", ctypes.c_uint),
                ("bayerFormat", ctypes.c_uint),
                ("imageImpl", ctypes.c_void_p),
            ]

        # Retry loop to absorb FC2_ERROR_TIMEOUT (err 41)
        for attempt in range(retries):
            raw_img = Fc2Image()
            self.fc2.fc2CreateImage(ctypes.byref(raw_img))
            err = self.fc2.fc2RetrieveBuffer(self.context, ctypes.byref(raw_img))

            if err == 0 and raw_img.pData and raw_img.rows > 0:
                buffer_size = raw_img.rows * raw_img.cols
                frame_gray = np.empty((raw_img.rows, raw_img.cols), dtype=np.uint8)
                ctypes.memmove(frame_gray.ctypes.data, raw_img.pData, buffer_size)
                self.fc2.fc2DestroyImage(ctypes.byref(raw_img))

                frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                return True, frame_bgr

            self.fc2.fc2DestroyImage(ctypes.byref(raw_img))
            time.sleep(0.01)  # Short 10ms wait before retrying on timeout

        return False, None

    def release(self):
        if hasattr(self, "context") and self.context:
            try:
                self.fc2.fc2StopCapture(self.context)
                self.fc2.fc2Disconnect(self.context)
                self.fc2.fc2DestroyContext(self.context)
            except Exception:
                pass
            self.context = None
            self._opened = False
# --------------------------------------------------------------------------
# Bullseye detection
# --------------------------------------------------------------------------

def _refine_center_subpixel(gray_like, x, y, r, band=4.0, max_shift=3.0):
    """
    Refine a Hough-detected circle center to sub-pixel precision.

    cv2.HoughCircles' own output is quantized by its accumulator resolution
    and gets further destroyed if it's rounded to an int (as this file used
    to do before averaging across rings), so two different real positions a
    fraction of a pixel apart can report the identical centroid. This finds
    the intensity-gradient-weighted centroid of the ring edge in a thin
    annulus around the Hough estimate, which is not restricted to integer
    (or even accumulator-resolution) coordinates.

    Falls back to the original (x, y) if there isn't enough edge signal in
    the annulus to trust the refit, or if the refined center would move
    implausibly far (a sign it locked onto some other nearby feature).
    """
    h, w = gray_like.shape
    pad = int(np.ceil(r + band + 2))
    x0, x1 = int(x - pad), int(x + pad)
    y0, y1 = int(y - pad), int(y + pad)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
        return float(x), float(y)  # too close to the frame edge to refine safely

    roi = gray_like[y0:y1, x0:x1].astype(np.float32)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    cx_local, cy_local = x - x0, y - y0
    dist = np.hypot(xx - cx_local, yy - cy_local)
    annulus = (dist > r - band) & (dist < r + band)

    weights = np.where(annulus, grad_mag, 0.0)
    total = weights.sum()
    if total < 1e-6:
        return float(x), float(y)

    x_refined = float((weights * xx).sum() / total) + x0
    y_refined = float((weights * yy).sum() / total) + y0

    if np.hypot(x_refined - x, y_refined - y) > max_shift:
        return float(x), float(y)

    return x_refined, y_refined


def detect_bullseye(frame_bgr, min_radius=8, max_radius=400, concentric_tol_px=10,
                     min_rings=3, min_radius_spread_px=15):
    """
    Detects concentric rings by sweeping through distinct radius ranges,
    bypassing HoughCircles' minDist center-collision limitation while remaining
    compatible with CLI arguments.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Enhance low-contrast image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

    # 2. Define radius bands for target rings bounded by min_radius & max_radius
    base_bands = [(10, 30), (30, 55), (55, 85), (85, 120), (120, 160), (160, 210)]
    radius_bands = [(max(min_r, min_radius), min(max_r, max_radius)) 
                    for min_r, max_r in base_bands 
                    if min_r < max_radius and max_r > min_radius]

    all_found_circles = []

    # 3. Sweep each radius band independently
    for min_r, max_r in radius_bands:
        if min_r >= max_r:
            continue
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,  # Suppresses duplicates within this specific size band
            param1=50,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r
        )
        if circles is not None:
            x, y, r = circles[0][0]
            # Refine on the sharp (unblurred) enhanced image — the Gaussian
            # blur used for Hough detection softens the very edge we want a
            # precise centroid from.
            x_ref, y_ref = _refine_center_subpixel(enhanced, x, y, r)
            all_found_circles.append((x_ref, y_ref, float(r)))

    debug = frame_bgr.copy()
    if len(all_found_circles) < min_rings:
        return None, None, debug

    all_found_circles = np.array(all_found_circles)
    radii = all_found_circles[:, 2]

    # Verify min radius spread requirement across detected rings
    radius_spread = radii.max() - radii.min()
    if radius_spread < min_radius_spread_px:
        return None, None, debug

    # Draw individual detected concentric rings (gray) — cv2 drawing needs
    # integer coordinates, but that's a display-only rounding now, not the
    # value used for the actual centroid math below.
    for (x, y, r) in all_found_circles:
        cv2.circle(debug, (int(round(x)), int(round(y))), int(round(r)),
                   (120, 120, 120), 1)

    # Calculate average center position across detected rings, at full
    # sub-pixel precision (no rounding of the individual ring centers).
    centers = all_found_circles[:, :2]
    cx, cy = centers.mean(axis=0)

    # Draw detected bullseye centroid (red marker)
    cv2.drawMarker(debug, (int(round(cx)), int(round(cy))), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return float(cx), float(cy), debug
# --------------------------------------------------------------------------
# Incremental (velocity-form) PID, one instance per pixel axis
# --------------------------------------------------------------------------

@dataclass
class IncrementalPID:
    kp: float
    ki: float
    kd: float
    e_prev: float = 0.0
    e_prev2: float = 0.0
    initialized: bool = False

    def step(self, error, dt):
        if dt <= 0:
            return 0.0
        if not self.initialized:
            # first call: no history yet, proportional-only to avoid a
            # derivative/integral spike from undefined prior error
            self.e_prev = error
            self.e_prev2 = error
            self.initialized = True
        d_error = error - self.e_prev
        dd_error = (error - self.e_prev) - (self.e_prev - self.e_prev2)
        delta_u = (
            self.kp * d_error
            + self.ki * error * dt
            + self.kd * (dd_error / dt)
        )
        self.e_prev2 = self.e_prev
        self.e_prev = error
        return delta_u


def grab_frame_safe(cap, attempts=20, delay=0.03):
    """
    cap.read() can transiently fail or return an empty frame (seen with the
    FlyCapture backend especially right after a voltage change, before the
    camera has a fresh frame ready). Retry briefly rather than handing a
    possibly-empty frame straight to OpenCV, which crashes instead of
    failing gracefully. Returns the frame, or None if all attempts failed.
    """
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(delay)
    return None


def compute_response_matrix(mdt, cap, base_v, vmin, vmax, amplitude=3.0, n_samples=5,
                             settle_s=0.15, min_rings=3, min_radius_spread_px=10.0,
                             show=False):
    """
    Auto-calibrate the pixel/volt Jacobian by perturbing x and y around the
    current operating point (base_v) and measuring the resulting centroid
    shift. Captures cross-axis coupling (x-voltage moving y-pixel, etc.) —
    does not assume a diagonal matrix.

    Only characterizes LOCAL sensitivity near the current voltages, so
    perturbations are small (a few volts) rather than a full-range sweep.
    All test voltages are clamped to [vmin, vmax], so this respects the same
    guardrails the live loop uses. Each axis is restored to its baseline
    voltage before moving to the next, and again before returning.

    Raises RuntimeError (leaving voltages restored to baseline) if the
    bullseye is lost at any point — calibration should not silently produce
    a matrix built on partial/bad data.
    """
    axes = ("x", "y")

    span = vmax - vmin
    if amplitude > 0.25 * span:
        print(f"[calib] NOTE: --calib-amplitude {amplitude:.1f}V is a large fraction of your "
              f"[{vmin:.0f}, {vmax:.0f}]V range. The first test point will be a single "
              f"~{amplitude:.1f}V jump from baseline. If you want smoother, smaller steps "
              f"between test points, increase --calib-samples rather than reducing amplitude.")

    frame = grab_frame_safe(cap)
    if frame is None:
        raise RuntimeError("Could not grab a valid frame from the camera to start "
                            "calibration — check the camera connection/backend.")
    cx0, cy0, debug0 = detect_bullseye(frame, min_rings=min_rings, min_radius_spread_px=min_radius_spread_px)
    if show:
        cv2.imshow("loop_test", debug0)
        cv2.waitKey(1)
    if cx0 is None:
        raise RuntimeError("Bullseye not visible at the start of calibration — "
                            "center the target in frame before calibrating.")

    print(f"[calib] starting from centroid ({cx0:.1f}, {cy0:.1f}) px, "
          f"baseline voltages vx={base_v['x']:.2f} vy={base_v['y']:.2f}")

    J = np.zeros((2, 2))  # rows: [dpx, dpy], cols: [x-axis volt, y-axis volt]
    deltas = np.linspace(-amplitude, amplitude, n_samples)

    for col, axis in enumerate(axes):
        measured_px, measured_py, applied_dv = [], [], []
        try:
            for d in deltas:
                v_test = float(np.clip(base_v[axis] + d, vmin, vmax))
                mdt.set_voltage(axis, v_test)
                time.sleep(settle_s)
                frame = grab_frame_safe(cap)
                if frame is None:
                    raise RuntimeError(
                        f"Could not grab a valid frame while calibrating the {axis}-axis "
                        f"at {v_test:.2f}V — aborting calibration."
                    )
                cx, cy, debug = detect_bullseye(frame, min_rings=min_rings,
                                             min_radius_spread_px=min_radius_spread_px)
                if show:
                    label = f"calibrating {axis}-axis  {v_test:+.2f}V (base {base_v[axis]:.2f}V)"
                    cv2.putText(debug, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 1)
                    cv2.imshow("loop_test", debug)
                    cv2.waitKey(1)
                if cx is None:
                    raise RuntimeError(
                        f"Lost the bullseye while calibrating the {axis}-axis "
                        f"at {v_test:.2f}V — aborting calibration."
                    )
                measured_px.append(cx)
                measured_py.append(cy)
                applied_dv.append(v_test - base_v[axis])
        finally:
            # always restore this axis to baseline before moving on / raising
            mdt.set_voltage(axis, base_v[axis])
            time.sleep(settle_s)

        applied_dv = np.array(applied_dv)
        if np.ptp(applied_dv) < 1e-6:
            raise RuntimeError(
                f"Calibration amplitude on {axis}-axis was entirely clamped by "
                f"[{vmin}, {vmax}]V bounds — baseline voltage is too close to a "
                f"limit. Move the mirror off the rail or reduce --calib-amplitude."
            )

        slope_x = np.polyfit(applied_dv, measured_px, 1)[0]
        slope_y = np.polyfit(applied_dv, measured_py, 1)[0]
        J[0, col] = slope_x
        J[1, col] = slope_y

    det = np.linalg.det(J)
    if abs(det) < 1e-9:
        raise RuntimeError("Calibration produced a singular response matrix — "
                            "try a larger --calib-amplitude or check the mirror "
                            "is actually connected and moving.")

    cond = np.linalg.cond(J)
    print("[calib] measured response matrix (px/volt):")
    print(f"           x-axis    y-axis")
    print(f"  dpx/dV  {J[0,0]:8.3f}  {J[0,1]:8.3f}")
    print(f"  dpy/dV  {J[1,0]:8.3f}  {J[1,1]:8.3f}")
    print(f"[calib] condition number: {cond:.1f}" +
          ("" if cond < 50 else "  (WARNING: poorly conditioned — consider a "
                                 "larger --calib-amplitude or more --calib-samples)"))

    return J


# --------------------------------------------------------------------------
# Response matrix (Jacobian): pixel_shift = J @ volt_delta  ->  volt_delta = J_inv @ pixel_shift
# --------------------------------------------------------------------------

DEFAULT_JACOBIAN = [[0.10, 0.0], [0.0, 0.10]]  # px/volt, PLACEHOLDER — see module docstring


def load_jacobian(path):
    if path is None:
        J = np.array(DEFAULT_JACOBIAN, dtype=float)
        print("[calib] using PLACEHOLDER response matrix — replace with a "
              "measured Jacobian before trusting this for real alignment.")
    else:
        with open(path, "r") as f:
            data = json.load(f)
        J = np.array(data["jacobian_px_per_volt"], dtype=float)
        print(f"[calib] loaded response matrix from {path}")
    if abs(np.linalg.det(J)) < 1e-9:
        raise ValueError("response matrix is singular / near-singular — "
                          "recalibrate before running")
    return J, np.linalg.inv(J)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=PORT, help="serial port for MDT693B (e.g. COM5 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="camera index (meaning depends on --camera-backend)")
    ap.add_argument("--camera-backend", choices=["opencv", "flycapture"], default=CAMERA_BACKEND,
                     help="'opencv' for a standard webcam, 'flycapture' for the Firefly MV / other "
                          "Point Grey cameras via PyCapture2")
    ap.add_argument("--duration", type=float, default=DURATION_S, help="seconds to run")
    ap.add_argument("--target-x", type=float, default=TARGET_X, help="setpoint px, default = frame center")
    ap.add_argument("--target-y", type=float, default=TARGET_Y, help="setpoint px, default = frame center")
    ap.add_argument("--controller", choices=["direct", "pid"], default=CONTROLLER,
                     help="'direct' (default): each cycle, compute the volts needed to fully null "
                          "the pixel error via the calibrated matrix and apply a fraction of it "
                          "(--gain). Appropriate for this system since there's no momentum/dynamics "
                          "to anticipate. 'pid': legacy incremental PID, kept for comparison.")
    ap.add_argument("--gain", type=float, default=GAIN,
                     help="fraction of the fully-corrective voltage applied per cycle in 'direct' "
                          "mode (0-1). Lower = smoother/more damped, higher = faster but more "
                          "sensitive to detection noise and any model mismatch.")
    ap.add_argument("--deadband-px", type=float, default=DEADBAND_PX,
                     help="in 'direct' mode, no correction is applied if both axes' pixel error are "
                          "within this deadband — prevents the controller from chasing detector jitter.")
    ap.add_argument("--kp", type=float, default=KP, help="(--controller pid only)")
    ap.add_argument("--ki", type=float, default=KI, help="(--controller pid only)")
    ap.add_argument("--kd", type=float, default=KD, help="(--controller pid only)")
    ap.add_argument("--v-init", type=float, default=V_INIT, help="starting voltage on x/y axes (mid-range)")
    ap.add_argument("--vmin", type=float, default=V_MIN, help="hard lower voltage guardrail (V)")
    ap.add_argument("--vmax", type=float, default=V_MAX, help="hard upper voltage guardrail (V)")
    ap.add_argument("--max-step-volts", type=float, default=MAX_STEP_VOLTS,
                     help="hard cap on voltage change per axis per cycle, independent of gains")
    ap.add_argument("--matrix", default=MATRIX_PATH, help="path to JSON file with a calibrated response matrix")
    ap.add_argument("--calibrate", action="store_true", default=CALIBRATE,
                     help="auto-compute the response matrix at startup instead of loading --matrix")
    ap.add_argument("--calib-amplitude", type=float, default=CALIB_AMPLITUDE_V,
                     help="volts to perturb each axis by during auto-calibration")
    ap.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES,
                     help="number of test points per axis during auto-calibration")
    ap.add_argument("--calib-settle", type=float, default=CALIB_SETTLE_S,
                     help="seconds to wait after each voltage change before reading the camera during calibration")
    ap.add_argument("--calib-save", default=CALIB_SAVE_PATH,
                     help="path to save the auto-calibrated matrix (default: auto-named with timestamp)")
    ap.add_argument("--feedforward", action="store_true", default=FEEDFORWARD,
                     help="apply one calibrated jump toward the target before starting the PID loop, "
                          "instead of relying on the incremental PID to slowly build up voltage")
    ap.add_argument("--feedforward-max-volts", type=float, default=FEEDFORWARD_MAX_VOLTS,
                     help="cap on the size of the one-shot feedforward jump per axis (independent of --max-step-volts)")
    ap.add_argument("--min-rings", type=int, default=MIN_RINGS,
                     help="minimum number of concentric Hough circles required to count as a valid detection")
    ap.add_argument("--min-radius-spread", type=float, default=MIN_RADIUS_SPREAD_PX,
                     help="minimum px spread between smallest/largest ring radius in a cluster to count as valid")
    ap.add_argument("--ema-alpha", type=float, default=EMA_ALPHA,
                     help="EMA smoothing weight on each new centroid measurement (0-1]. "
                          "1.0 disables smoothing (use the raw per-frame detection); lower "
                          "values trade responsiveness for less noise reaching the controller.")
    ap.add_argument("--log", default=LOG_PATH, help="CSV log path (default: auto-named with timestamp)")
    ap.add_argument("--show", action="store_true", default=SHOW_WINDOW, help="show a live debug window")
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN, help="print serial commands instead of sending them")
    # Deliberately parse an EMPTY argument list, not sys.argv. This script is
    # meant to be run from Spyder via the SETTINGS block above, and Spyder's
    # %runfile/run mechanism can inject its own content into sys.argv (as
    # seen when '--wdir' ended up interfering with --port). Ignoring argv
    # entirely guarantees the SETTINGS block is always what's used, no
    # matter how this file is launched.
    args = ap.parse_args([])

    if args.port is None and not args.dry_run:
        print("ERROR: --port is required unless --dry-run is set.", file=sys.stderr)
        sys.exit(1)

    if args.vmin < 0 or args.vmax > 150 or args.vmin >= args.vmax:
        print("ERROR: voltage bounds must satisfy 0 <= vmin < vmax <= 150.", file=sys.stderr)
        sys.exit(1)

    if args.calibrate and args.dry_run:
        print("ERROR: --calibrate requires real hardware (it measures actual "
              "mirror motion) and cannot be used with --dry-run.", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < args.ema_alpha <= 1.0):
        print("ERROR: --ema-alpha must be in (0, 1].", file=sys.stderr)
        sys.exit(1)

    log_path = args.log or f"loop_test_{datetime.now():%Y%m%d_%H%M%S}.csv"

    try:
        if args.camera_backend == "flycapture":
            cap = FlyCaptureSource(args.camera)
        else:
            cap = OpenCVCameraSource(args.camera)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera} (backend={args.camera_backend})",
              file=sys.stderr)
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("ERROR: could not read a frame from the camera.", file=sys.stderr)
        sys.exit(1)
    h, w = frame.shape[:2]
    target_x = args.target_x if args.target_x is not None else w / 2.0
    target_y = args.target_y if args.target_y is not None else h / 2.0

    if args.show:
        # Open the debug window now so calibration/feedforward are visible
        # too, instead of only appearing once the tracking loop starts.
        cv2.namedWindow("loop_test")
        cv2.imshow("loop_test", frame)
        cv2.waitKey(1)

    J, J_inv = load_jacobian(args.matrix)

    try:
        mdt = MDT693BController(args.port, baud=args.baud, dry_run=args.dry_run,
                                 vmin=args.vmin, vmax=args.vmax)
    except Exception as e:
        print(f"ERROR: could not open MDT693B on {args.port}: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)
    v = {"x": args.v_init, "y": args.v_init, "z": args.v_init}
    for axis in ("x", "y"):
        mdt.set_voltage(axis, v[axis])

    if args.calibrate:
        time.sleep(args.calib_settle)  # let the initial voltage settle before measuring
        try:
            J = compute_response_matrix(
                mdt, cap, v, args.vmin, args.vmax,
                amplitude=args.calib_amplitude, n_samples=args.calib_samples,
                settle_s=args.calib_settle, min_rings=args.min_rings,
                min_radius_spread_px=args.min_radius_spread,
                show=args.show,
            )
        except RuntimeError as e:
            print(f"ERROR: calibration failed: {e}", file=sys.stderr)
            mdt.close()
            cap.release()
            sys.exit(1)
        J_inv = np.linalg.inv(J)
        calib_path = args.calib_save or f"calib_matrix_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(calib_path, "w") as f:
            json.dump({
                "jacobian_px_per_volt": J.tolist(),
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                "amplitude_volts": args.calib_amplitude,
                "n_samples": args.calib_samples,
                "vmin": args.vmin, "vmax": args.vmax,
            }, f, indent=2)
        print(f"[calib] saved matrix to {calib_path} — reuse it later with --matrix {calib_path}")

    if args.feedforward:
        frame = grab_frame_safe(cap)
        cx, cy = (None, None)
        if frame is not None:
            cx, cy, debug = detect_bullseye(frame, min_rings=args.min_rings,
                                         min_radius_spread_px=args.min_radius_spread)
            if args.show:
                cv2.imshow("loop_test", debug)
                cv2.waitKey(1)
        if cx is None:
            print("[feedforward] bullseye not visible — skipping feedforward jump, "
                  "PID loop will start from current voltages.")
        else:
            ex = target_x - cx
            ey = target_y - cy
            dv0 = J_inv @ np.array([ex, ey])
            dv0 = np.clip(dv0, -args.feedforward_max_volts, args.feedforward_max_volts)
            print(f"[feedforward] error=({ex:.1f}, {ey:.1f})px -> requesting "
                  f"({dv0[0]:+.2f}, {dv0[1]:+.2f})V "
                  f"(capped at ±{args.feedforward_max_volts:.1f}V per axis)")
            v["x"] = float(np.clip(v["x"] + dv0[0], args.vmin, args.vmax))
            v["y"] = float(np.clip(v["y"] + dv0[1], args.vmin, args.vmax))
            mdt.set_voltage("x", v["x"])
            mdt.set_voltage("y", v["y"])
            time.sleep(args.calib_settle)
            print(f"[feedforward] applied — voltages now vx={v['x']:.2f} vy={v['y']:.2f}")

    pid_x = IncrementalPID(args.kp, args.ki, args.kd)
    pid_y = IncrementalPID(args.kp, args.ki, args.kd)

    if args.controller == "direct":
        print(f"[loop] controller=direct  gain={args.gain}  deadband={args.deadband_px}px")
    else:
        print(f"[loop] controller=pid  kp={args.kp} ki={args.ki} kd={args.kd}")

    print(f"[loop] target=({target_x:.1f}, {target_y:.1f}) px  "
          f"duration={args.duration}s  log={log_path}")
    print("[loop] press Ctrl+C to stop early (voltages will be safely held, not zeroed)")

    frame_count = 0
    lost_count = 0
    loop_times = []
    cx_filt, cy_filt = None, None  # EMA-smoothed centroid state; None until first detection
    t_start = time.perf_counter()
    t_prev = t_start

    try:
        with open(log_path, "w", newline="") as logf:
            writer = csv.writer(logf)
            writer.writerow(["t_s", "detected", "cx_px", "cy_px", "cx_filt_px", "cy_filt_px",
                              "ex_px", "ey_px", "vx", "vy", "loop_dt_s"])

            while (time.perf_counter() - t_start) < args.duration:
                t_now = time.perf_counter()
                dt = t_now - t_prev
                t_prev = t_now

                ok, frame = cap.read()
                if not ok:
                    print("[loop] warning: frame grab failed, skipping cycle")
                    continue

                cx, cy, debug = detect_bullseye(frame, min_rings=args.min_rings,
                                                 min_radius_spread_px=args.min_radius_spread)
                frame_count += 1
                dist_px = None  # set below when the bullseye is detected

                if cx is None:
                    lost_count += 1
                    writer.writerow([f"{t_now - t_start:.4f}", 0, "", "", "", "",
                                      "", "", f"{v['x']:.3f}", f"{v['y']:.3f}",
                                      f"{dt:.4f}"])
                else:
                    # EMA-smooth the raw per-frame centroid before it drives the
                    # controller, so single-frame detection noise (including the
                    # residual sub-pixel jitter the refinement above can't fully
                    # remove) doesn't translate directly into voltage commands.
                    if cx_filt is None:
                        cx_filt, cy_filt = cx, cy
                    else:
                        cx_filt = args.ema_alpha * cx + (1 - args.ema_alpha) * cx_filt
                        cy_filt = args.ema_alpha * cy + (1 - args.ema_alpha) * cy_filt

                    ex = target_x - cx_filt
                    ey = target_y - cy_filt
                    dist_px = float(np.hypot(ex, ey))

                    if args.controller == "direct":
                        if abs(ex) < args.deadband_px and abs(ey) < args.deadband_px:
                            # within the noise floor of detection — hold voltage rather
                            # than keep correcting against jitter
                            dv = np.zeros(2)
                        else:
                            # volts that would fully null the error, per the calibrated
                            # matrix, damped by --gain so a single noisy frame or small
                            # model mismatch doesn't cause overshoot/oscillation
                            dv = args.gain * (J_inv @ np.array([ex, ey]))
                    else:
                        du_px_x = pid_x.step(ex, dt)
                        du_px_y = pid_y.step(ey, dt)
                        dv = J_inv @ np.array([du_px_x, du_px_y])

                    dv = np.clip(dv, -args.max_step_volts, args.max_step_volts)

                    v["x"] = float(np.clip(v["x"] + dv[0], args.vmin, args.vmax))
                    v["y"] = float(np.clip(v["y"] + dv[1], args.vmin, args.vmax))
                    mdt.set_voltage("x", v["x"])
                    mdt.set_voltage("y", v["y"])

                    writer.writerow([f"{t_now - t_start:.4f}", 1, f"{cx:.2f}", f"{cy:.2f}",
                                      f"{cx_filt:.2f}", f"{cy_filt:.2f}",
                                      f"{ex:.2f}", f"{ey:.2f}", f"{v['x']:.3f}", f"{v['y']:.3f}",
                                      f"{dt:.4f}"])

                loop_times.append(dt)

                if args.show:
                    cv2.drawMarker(debug, (int(target_x), int(target_y)), (0, 255, 0),
                                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=1)
                    if cx_filt is not None:
                        # yellow circle = EMA-smoothed centroid actually driving the
                        # controller, alongside detect_bullseye's raw red cross
                        cv2.circle(debug, (int(round(cx_filt)), int(round(cy_filt))),
                                   6, (0, 255, 255), 1)
                    cv2.putText(debug, f"vx={v['x']:.1f}V vy={v['y']:.1f}V",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    dist_str = f"{dist_px:.1f}px" if dist_px is not None else "n/a"
                    cv2.putText(debug, f"dist={dist_str}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.imshow("loop_test", debug)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("\n[loop] interrupted by user")

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        mdt.close()

    elapsed = time.perf_counter() - t_start
    if loop_times:
        rate = 1.0 / (sum(loop_times) / len(loop_times))
    else:
        rate = 0.0
    print(f"[summary] ran {elapsed:.1f}s, {frame_count} frames, "
          f"{lost_count} lost ({100 * lost_count / max(frame_count,1):.1f}%), "
          f"avg loop rate {rate:.1f} Hz")
    print(f"[summary] final voltages: vx={v['x']:.2f}V vy={v['y']:.2f}V")
    print(f"[summary] log written to {log_path}")


if __name__ == "__main__":
    main()
