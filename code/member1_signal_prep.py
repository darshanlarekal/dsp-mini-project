"""MFPT bearing-signal helpers used by the Member 1 MFPT section.

The public MFPT test-rig recordings store a ``bearing`` struct with the fields
``gs`` (acceleration signal), ``sr`` (sampling rate), ``rate`` (shaft speed in
Hz) and ``load`` (radial load in lb).  The four characteristic defect
frequencies published with the dataset at the 25 Hz reference shaft speed are
BPFO = 81.125 Hz, BPFI = 118.88 Hz, FTF = 14.838 Hz and BSF = 63.91 Hz; they
scale linearly with shaft speed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
from scipy import signal

# NICE bearing used on the MFPT test rig: 8 balls, 0.2354 in ball diameter,
# 2.815 in pitch diameter, zero contact angle.
MFPT_NICE = SimpleNamespace(
    reference_shaft_rate_hz=25.0,
    BPFO_Hz=81.125,
    BPFI_Hz=118.88,
    FTF_Hz=14.838,
    BSF_Hz=63.91,
    n_balls=8,
    ball_diameter_in=0.2354,
    pitch_diameter_in=2.815,
    contact_angle_deg=0.0,
)


def kinematic_fault_frequencies(shaft_rate_hz, geometry=MFPT_NICE):
    """Return BPFO, BPFI, BSF and FTF in Hz at the given shaft speed."""
    shaft_rate_hz = float(shaft_rate_hz)
    if not np.isfinite(shaft_rate_hz) or shaft_rate_hz <= 0:
        raise ValueError("Shaft rate must be a positive finite value.")

    scale = shaft_rate_hz / geometry.reference_shaft_rate_hz
    return {
        "shaft_rate_Hz": shaft_rate_hz,
        "shaft_rpm": 60.0 * shaft_rate_hz,
        "BPFO_Hz": geometry.BPFO_Hz * scale,
        "BPFI_Hz": geometry.BPFI_Hz * scale,
        "BSF_Hz": geometry.BSF_Hz * scale,
        "FTF_Hz": geometry.FTF_Hz * scale,
    }


def load_mfpt_mat(path):
    """Load one MFPT recording and attach its fault class and operating point."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    if "bearing" not in data:
        raise KeyError(f"No 'bearing' struct in {os.path.basename(path)}")

    bearing = data["bearing"]
    x = np.asarray(bearing.gs, dtype=np.float64).squeeze()
    if x.ndim != 1 or x.size < 16 or not np.all(np.isfinite(x)):
        raise ValueError(f"Invalid signal array in {os.path.basename(path)}: shape={x.shape}")

    sampling_rate = float(bearing.sr)
    shaft_rate_hz = float(bearing.rate)
    load_lbs = float(bearing.load)

    name = os.path.basename(path).lower()
    if name.startswith("baseline"):
        fault = "Normal"
    elif name.startswith("innerracefault"):
        fault = "Inner race"
    elif name.startswith("outerracefault"):
        fault = "Outer race"
    else:
        raise ValueError(f"Unrecognized MFPT filename: {name}")

    return SimpleNamespace(
        signal=x,
        sampling_rate=sampling_rate,
        duration_seconds=x.size / sampling_rate,
        metadata={
            "fault": fault,
            "load_lbs": load_lbs,
            "shaft_rate_hz": shaft_rate_hz,
            "shaft_rpm": 60.0 * shaft_rate_hz,
        },
    )


def preprocess_signal(x, normalization="none"):
    """Remove the DC mean and optionally normalize by RMS or peak amplitude."""
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 1 or x.size < 16 or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite one-dimensional signal of at least 16 samples.")

    dc_offset = float(np.mean(x))
    demeaned = x - dc_offset
    rms = float(np.sqrt(np.mean(demeaned**2)))
    peak = float(np.max(np.abs(demeaned)))

    if normalization == "none":
        normalized = demeaned.copy()
    elif normalization == "rms":
        normalized = demeaned / rms if rms > np.finfo(float).eps else demeaned.copy()
    elif normalization == "peak":
        normalized = demeaned / peak if peak > np.finfo(float).eps else demeaned.copy()
    else:
        raise ValueError(f"normalization must be 'none', 'rms' or 'peak', got {normalization!r}")

    stats = {
        "dc_offset": dc_offset,
        "rms_before_normalization": rms,
        "peak_before_normalization": peak,
    }
    return demeaned, normalized, stats


def single_sided_amplitude_spectrum(x, fs, window="hann"):
    """Coherent-gain-corrected one-sided amplitude spectrum."""
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 1 or x.size < 16:
        raise ValueError("Expected a one-dimensional signal with at least 16 samples.")

    x = x - np.mean(x)
    w = signal.get_window(window, x.size, fftbins=True)
    scale = np.sum(w)
    if scale <= np.finfo(float).eps:
        raise ValueError("Window coherent gain is zero.")

    spectrum = np.abs(np.fft.rfft(x * w)) / scale
    if spectrum.size > 2:
        spectrum[1:-1] *= 2.0

    frequencies = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    return frequencies, spectrum
