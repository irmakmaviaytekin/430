"""
make_demo_data.py: FAKE files in the same formats as the real data, for testing mirror_tools.py.

Makes (in the folder you choose):
   demo_dark.tif       200 dark frames (camera background only)
   demo_quiet.tif      quiet recording: fringes + drift + 50 Hz vibration + noise
   demo_loop_log.csv   a log in the same columns as Calum's loop_test.py
   demo_truth.json     the true values used, so the analysis can be checked
"""
import csv
import json
import os
import numpy as np
import tifffile


def make_all(folder="demo_data", wavelength_nm=532.0, fps=1000, n_frames=5000, seed=7):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    ny, nx = 64, 64
    y, x = np.mgrid[0:ny, 0:nx]
    background = 100.0                                    # camera offset (counts)
    beam = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 40.0 ** 2))   # brighter in the middle
    peak = 1800.0                                         # counts at the beam centre (stays below 4095)
    contrast = 0.80
    kx, ky = 2 * np.pi * 0.08, 2 * np.pi * 0.03           # stripe frequency (per pixel)

    # True mirror motion (OPD, nm): drift + 50 Hz vibration + small random jitter
    t = np.arange(n_frames) / fps
    opd = 20 * np.sin(2 * np.pi * 0.7 * t) + 8 * np.sin(2 * np.pi * 50 * t) + 3 * rng.standard_normal(n_frames)
    phase = 2 * np.pi * opd / wavelength_nm

    frames = np.empty((n_frames, ny, nx), dtype=np.uint16)
    for i in range(n_frames):
        light = peak * beam * (1 + contrast * np.cos(kx * x + ky * y + phase[i]))
        counts = rng.poisson(light) + background + 5 * rng.standard_normal((ny, nx))   # photon + read noise
        frames[i] = np.clip(counts, 0, 4095)
    dark = np.clip(background + 5 * rng.standard_normal((200, ny, nx)), 0, 4095).astype(np.uint16)

    tifffile.imwrite(os.path.join(folder, "demo_quiet.tif"), frames)
    tifffile.imwrite(os.path.join(folder, "demo_dark.tif"), dark)

    # Fake loop log (same columns as loop_test.py): starts 30 px off, settles, then jitters
    rows, tt, cxf, cyf = [], 0.0, None, None
    vx, vy = 75.0, 75.0
    ex0, ey0 = 30.0, -20.0
    for i in range(750):
        dt = 1 / 25 + 0.004 * rng.standard_normal()
        tt += max(dt, 0.01)
        decay = 0.6 ** i
        cx = 320 - ex0 * decay + 0.4 * rng.standard_normal() + 0.3 * np.sin(2 * np.pi * 3 * tt)
        cy = 240 - ey0 * decay + 0.4 * rng.standard_normal()
        lost = rng.random() < 0.02
        if lost:
            rows.append([f"{tt:.4f}", 0, "", "", "", "", "", "", f"{vx:.3f}", f"{vy:.3f}", f"{dt:.4f}"])
            continue
        cxf = cx if cxf is None else 0.5 * cx + 0.5 * cxf
        cyf = cy if cyf is None else 0.5 * cy + 0.5 * cyf
        ex, ey = 320 - cxf, 240 - cyf
        vx = 75.0 + 3.0 * (1 - decay) + 0.05 * rng.standard_normal()   # piezo voltages settle too
        vy = 75.0 - 2.0 * (1 - decay) + 0.05 * rng.standard_normal()
        rows.append([f"{tt:.4f}", 1, f"{cx:.2f}", f"{cy:.2f}", f"{cxf:.2f}", f"{cyf:.2f}",
                     f"{ex:.2f}", f"{ey:.2f}", f"{vx:.3f}", f"{vy:.3f}", f"{dt:.4f}"])
    with open(os.path.join(folder, "demo_loop_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "detected", "cx_px", "cy_px", "cx_filt_px", "cy_filt_px",
                    "ex_px", "ey_px", "vx", "vy", "loop_dt_s"])
        w.writerows(rows)

    truth = dict(wavelength_nm=wavelength_nm, fps=fps, contrast=contrast,
                 opd_rms_nm=float(opd.std()), vibration_hz=50.0, vibration_amp_nm=8.0,
                 drift_hz=0.7, drift_amp_nm=20.0, background_counts=background,
                 fringe_freq=float(np.hypot(kx, ky) / (2 * np.pi)))
    with open(os.path.join(folder, "demo_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return truth


if __name__ == "__main__":
    print(make_all())
