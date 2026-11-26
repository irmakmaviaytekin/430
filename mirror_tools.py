"""
mirror_tools.py: analysis tools for real bench data (Irmak Aytekin, AST430).

Two kinds of input:
  1. Camera recordings of fringes (a stack of frames) -> contrast, mirror position, wobble, vibration spectrum
  2. Calum's loop_test.py CSV logs                    -> loop rate, settling, pointing stability

Import it in a notebook with:   from mirror_tools import *
"""
import csv
import json
import numpy as np
from scipy.optimize import curve_fit


# ======================================================================
# 1. Loading files
# ======================================================================

def load_frames(path):
    """Load a recording as a float array of shape (n_frames, rows, columns).

    Supports .npy, .tif/.tiff (needs `tifffile`) and .fits/.fit (needs `astropy`).
    """
    p = str(path).lower()
    if p.endswith(".npy"):
        data = np.load(path)
    elif p.endswith((".tif", ".tiff")):
        import tifffile                      # pip install tifffile
        data = tifffile.imread(path)
    elif p.endswith((".fits", ".fit")):
        from astropy.io import fits          # pip install astropy
        data = fits.getdata(path)
    else:
        raise ValueError(f"Don't know how to read {path}: use .npy, .tif or .fits")
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:                       # a single frame -> make it a stack of 1
        data = data[np.newaxis]
    return data


# ======================================================================
# 2. Checking and cleaning frames
# ======================================================================

def check_frames(frames, saturation_level, name="recording"):
    """Print basic checks. saturation_level = the camera's maximum value (e.g. 4095 or 65535)."""
    n, ny, nx = frames.shape
    peak = frames.max()
    frac_sat = np.mean(frames >= saturation_level)
    print(f"{name}: {n} frames of {ny} x {nx} pixels")
    print(f"   brightest pixel {peak:.0f} of max {saturation_level}  ({peak / saturation_level:.0%})")
    print(f"   mean level {frames.mean():.1f}")
    if frac_sat > 0:
        print(f"   WARNING: {frac_sat:.2%} of pixels are saturated: shorten the exposure")
    elif peak > 0.9 * saturation_level:
        print("   WARNING: close to saturation: consider a shorter exposure")
    return dict(n_frames=n, peak=peak, saturated_fraction=frac_sat)


def dark_level(dark_frames):
    """Average dark frame: the camera's background, pixel by pixel."""
    return dark_frames.mean(axis=0)


def crop(frames, y0, x0, size=32):
    """Cut a size x size window starting at row y0, column x0, from every frame."""
    return frames[:, y0:y0 + size, x0:x0 + size]


# ======================================================================
# 3. Fitting fringes (contrast AND position)
# ======================================================================

def _fringe_model(xy, mean, amp, kx, ky, phase):
    x, y = xy
    return mean + amp * np.cos(kx * x + ky * y + phase)


def _centred_grid(ny, nx):
    """Pixel coordinates measured from the CENTRE of the window, so the fitted phase
    is the fringe position at the window centre (a tilt change then doesn't leak into it)."""
    y, x = np.mgrid[0:ny, 0:nx]
    return x - (nx - 1) / 2, y - (ny - 1) / 2


def _fft_guess(img):
    """Starting guess for the fit from the strongest peak of the 2D FFT."""
    ny, nx = img.shape
    F = np.fft.fft2(img - img.mean()); F[0, 0] = 0
    iy, ix = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    return [img.mean(), (img.max() - img.min()) / 2,
            2 * np.pi * np.fft.fftfreq(nx)[ix], 2 * np.pi * np.fft.fftfreq(ny)[iy], 0.0]


def fit_fringes(image, p0=None):
    """Fit straight fringes to one frame.

    Returns contrast (mod_eff), stripe frequency, AND the fringe phase at the window centre,
    each with a 1-sigma error. p0 = starting guess (e.g. the previous frame's result).
    """
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    x, y = _centred_grid(ny, nx)
    if p0 is None:
        p0 = _fft_guess(img)
    popt, pcov = curve_fit(_fringe_model, (x.ravel(), y.ravel()), img.ravel(), p0=p0)
    perr = np.sqrt(np.diag(pcov))
    mean, amp, kx, ky, phase = popt
    if amp < 0:                               # a negative amplitude is the same as a half-stripe shift
        amp, phase = -amp, phase + np.pi
    me = amp / mean
    k = np.hypot(kx, ky)
    return dict(mod_eff=me, mod_eff_err=me * np.hypot(perr[1] / amp, perr[0] / mean),
                fringe_freq=k / (2 * np.pi),
                fringe_freq_err=np.hypot(kx * perr[2], ky * perr[3]) / k / (2 * np.pi),
                phase=np.angle(np.exp(1j * phase)), phase_err=perr[4],
                kx=kx, ky=ky, popt=np.array([mean, amp, kx, ky, phase]),
                fit=_fringe_model((x, y), mean, amp, kx, ky, phase))


def fit_all_frames(frames, max_frames=None):
    """Fit every frame. Each fit starts from the previous frame's answer (faster, and it keeps
    the stripe direction consistent so the phase can be followed from frame to frame)."""
    if max_frames is not None:
        frames = frames[:max_frames]
    keys = ("mod_eff", "mod_eff_err", "fringe_freq", "fringe_freq_err", "phase", "phase_err")
    out = {k: np.full(len(frames), np.nan) for k in keys}
    p0, k_ref, n_failed = None, None, 0
    for i, img in enumerate(frames):
        try:
            r = fit_fringes(img, p0=p0)
        except RuntimeError:                  # fit didn't converge: skip this frame, restart from FFT
            p0, n_failed = None, n_failed + 1
            continue
        k = np.array([r["kx"], r["ky"]])
        if k_ref is None:
            k_ref = k
        if k @ k_ref < 0:                     # cos(kx+phi) = cos(-kx-phi): pick the same direction every frame
            r["phase"] = -r["phase"]
            r["popt"][2:] = -r["popt"][2:]
        for key in keys:
            out[key][i] = r[key]
        p0 = r["popt"]
    out["n_failed"] = n_failed
    return out


def phase_to_opd_nm(phase, wavelength_nm):
    """Follow the fringe phase across frames (counting whole fringes) and convert to OPD in nm.
    The mirror itself moves half of this."""
    good = np.isfinite(phase)
    opd = np.full(phase.shape, np.nan)
    opd[good] = np.unwrap(phase[good]) * wavelength_nm / (2 * np.pi)
    return opd


def summarise(values, errors=None):
    """Mean, error of the mean, and frame-to-frame scatter (ignoring failed frames)."""
    v = values[np.isfinite(values)]
    return dict(mean=v.mean(), err=v.std(ddof=1) / np.sqrt(len(v)), scatter=v.std(ddof=1), n=len(v))


# ======================================================================
# 4. Wobble and vibrations
# ======================================================================

def rms_with_error(signal, fs_hz, window_s, n_blocks=10):
    """RMS of the last `window_s` seconds of `signal`, with a block-method error bar (as in Part 2)."""
    n = int(window_s * fs_hz)
    x = np.asarray(signal[-n:], dtype=float)
    x = x[np.isfinite(x)]
    rms = x.std()
    block_rms = np.array([b.std() for b in np.array_split(x, n_blocks)])
    return rms, block_rms.std(ddof=1) / np.sqrt(n_blocks)


def amplitude_spectrum(signal, fs_hz):
    """Which frequencies are shaking the mirror, and by how much (amplitude in the signal's units)."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    w = np.hanning(len(x))
    amp = 2 * np.abs(np.fft.rfft(x * w)) / w.sum()
    return np.fft.rfftfreq(len(x), 1 / fs_hz), amp


def top_peaks(freqs, amp, n=3, fmin=1.0):
    """The n strongest frequencies above fmin (Hz), ignoring their immediate neighbours."""
    a = amp.copy(); a[freqs < fmin] = 0
    peaks = []
    for _ in range(n):
        i = int(np.argmax(a))
        if a[i] == 0:
            break
        peaks.append((freqs[i], amp[i]))
        a[max(i - 3, 0):i + 4] = 0
    return peaks


def resample_uniform(t, x, fs_hz):
    """Put unevenly timed samples (e.g. a loop log) onto an even time grid, for the spectrum."""
    good = np.isfinite(x)
    t_new = np.arange(t[good][0], t[good][-1], 1 / fs_hz)
    return t_new, np.interp(t_new, t[good], x[good])


# ======================================================================
# 5. Calum's loop_test.py logs
# ======================================================================

def load_loop_log(path):
    """Read a loop_test CSV. Lost frames have empty cells, which become NaN."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])
    return {name: col(name) for name in rows[0].keys()}


def loop_summary(log, settle_px=2.0):
    """Loop rate, lost frames, settling time, and pointing stability after settling."""
    t, det = log["t_s"], log["detected"] == 1
    dist = np.hypot(log["ex_px"], log["ey_px"])
    dt = log["loop_dt_s"][1:]
    rate = 1 / np.median(dt)
    outside = np.where(det & (dist > settle_px))[0]       # detected frames still too far from target
    settled_idx = outside[-1] + 1 if len(outside) else 0
    s = dict(duration_s=t[-1] - t[0], n_frames=len(t), lost_fraction=1 - det.mean(),
             loop_rate_hz=rate, dt_scatter_ms=1000 * np.std(dt),
             settle_time_s=t[settled_idx] if settled_idx < len(t) else np.nan,
             final_vx=log["vx"][-1], final_vy=log["vy"][-1])
    after = slice(settled_idx, None)
    for axis in ("cx_px", "cy_px"):
        x = log[axis][after]
        x = x[np.isfinite(x)]
        if len(x) >= 20:
            s[axis + "_rms"], s[axis + "_rms_err"] = rms_with_error(x, rate, len(x) / rate)
    s["settled_index"] = settled_idx
    return s


def load_calibration(path):
    """Calum's saved response matrix: pixels moved per volt, for the x and y piezo axes."""
    with open(path) as f:
        return np.array(json.load(f)["jacobian_px_per_volt"])
