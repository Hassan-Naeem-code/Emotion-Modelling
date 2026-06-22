"""rPPG (remote photoplethysmography) — modality two, scaffolded.

Extracts a pulse signal from a sequence of face-ROI frames using the classic
POS algorithm (Wang et al., 2017) — no training required — and returns an
estimated heart rate plus a signal-quality score in [0,1].

HONESTY CONTRACT (read this): rPPG is highly sensitive to motion, lighting, and
skin tone. It is NOT a reliable affect signal on its own. The quality score MUST
gate any downstream use: low quality -> the rPPG feature is down-weighted toward
zero and INCREASES predictive uncertainty. It must never produce a confident
guess from a bad signal. Fusion (see fusion.py) enforces this gating.

This module is intentionally framework-light (numpy) so it can also inform the
browser demo's logic. Full video-window integration is future work.
"""
from __future__ import annotations

import numpy as np


def _bandpass(sig: np.ndarray, fps: float, lo=0.7, hi=4.0) -> np.ndarray:
    """Crude FFT bandpass to the plausible HR band (42–240 bpm)."""
    n = len(sig)
    if n < 4:
        return sig
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec = np.fft.rfft(sig - sig.mean())
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n=n)


def pos_pulse(rgb_means: np.ndarray, fps: float) -> np.ndarray:
    """POS algorithm. rgb_means: (T,3) mean RGB of the face ROI per frame."""
    if rgb_means.shape[0] < 8:
        return np.zeros(rgb_means.shape[0], dtype=np.float32)
    eps = 1e-8
    norm = rgb_means / (rgb_means.mean(axis=0, keepdims=True) + eps)
    # Projection plane orthogonal to skin tone.
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float32)
    s = norm @ proj.T                       # (T,2)
    alpha = (s[:, 0].std() + eps) / (s[:, 1].std() + eps)
    pulse = s[:, 0] + alpha * s[:, 1]
    return _bandpass(pulse.astype(np.float32), fps)


def estimate(rgb_means: np.ndarray, fps: float) -> dict:
    """Return {'hr_bpm', 'quality', 'arousal_feature'}.

    quality is the spectral peak prominence: how dominant the HR peak is over the
    rest of the band. arousal_feature is a quality-gated, normalized HR proxy.
    """
    pulse = pos_pulse(np.asarray(rgb_means, dtype=np.float32), fps)
    if len(pulse) < 8 or not np.any(pulse):
        return {"hr_bpm": 0.0, "quality": 0.0, "arousal_feature": 0.0}
    freqs = np.fft.rfftfreq(len(pulse), d=1.0 / fps)
    power = np.abs(np.fft.rfft(pulse)) ** 2
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band) or power[band].sum() <= 0:
        return {"hr_bpm": 0.0, "quality": 0.0, "arousal_feature": 0.0}
    peak_idx = np.argmax(power * band)
    hr_bpm = float(freqs[peak_idx] * 60.0)
    quality = float(power[peak_idx] / (power[band].sum() + 1e-8))   # in [0,1]
    # Normalize HR ~[40,180] bpm to a [-1,1] arousal proxy, gated by quality so a
    # weak signal contributes ~0 (and the fusion layer raises uncertainty).
    hr_norm = np.clip((hr_bpm - 70.0) / 60.0, -1.0, 1.0)
    arousal_feature = float(hr_norm * quality)
    return {"hr_bpm": hr_bpm, "quality": quality, "arousal_feature": arousal_feature}
