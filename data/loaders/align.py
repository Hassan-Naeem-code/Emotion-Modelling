"""Landmark-based face alignment for the image loaders.

The browser demo aligns faces via MediaPipe before inference; Track A must match
that preprocessing or train/serve skew will hurt the real numbers. This module
provides an eye-centered similarity-transform aligner.

It uses MediaPipe FaceMesh if the optional `mediapipe` package is installed; if
not, it falls back to a center square crop and logs once, so the pipeline never
hard-fails on a missing optional dependency. Install with `pip install mediapipe`
for real runs (kept out of the pinned core requirements to keep them light).
"""
from __future__ import annotations

import warnings

import numpy as np
from PIL import Image

# Canonical positions (fraction of output size) for the two eye centers after
# alignment. Standard face-recognition style template.
_LEFT_EYE_OUT = (0.35, 0.38)
_RIGHT_EYE_OUT = (0.65, 0.38)

# MediaPipe FaceMesh iris/eye landmark indices.
_LEFT_EYE_IDX = [33, 133]    # outer/inner corner of subject's left eye
_RIGHT_EYE_IDX = [362, 263]

_mp_mesh = None
_warned = False


def _get_mesh():
    global _mp_mesh
    if _mp_mesh is None:
        import mediapipe as mp  # raises ImportError if not installed
        _mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
    return _mp_mesh


def _eye_centers(img: np.ndarray):
    mesh = _get_mesh()
    res = mesh.process(img)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    def center(idx):
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
        return pts.mean(axis=0)
    return center(_LEFT_EYE_IDX), center(_RIGHT_EYE_IDX)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)


def align_face(img: Image.Image, size: int) -> Image.Image:
    """Return a `size`x`size` eye-aligned crop, or a center crop on fallback."""
    global _warned
    try:
        arr = np.asarray(img.convert("RGB"))
        eyes = _eye_centers(arr)
    except ImportError:
        if not _warned:
            warnings.warn("mediapipe not installed; using center-crop fallback. "
                          "pip install mediapipe for landmark alignment.")
            _warned = True
        return _center_crop(img, size)
    if eyes is None:
        return _center_crop(img, size)

    left_eye, right_eye = eyes
    # Similarity transform mapping detected eyes to the canonical template.
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = np.hypot(dx, dy)
    target_dist = (_RIGHT_EYE_OUT[0] - _LEFT_EYE_OUT[0]) * size
    scale = target_dist / (eye_dist + 1e-6)

    eyes_center = tuple(((left_eye + right_eye) / 2).tolist())
    rot = img.rotate(angle, center=eyes_center, resample=Image.BILINEAR)
    # Crop a box around the eye center sized so eyes land on the template row.
    cx, cy = eyes_center
    half = size / (2 * scale)
    # vertical offset so eyes sit at _LEFT_EYE_OUT[1] of the output.
    top_off = (_LEFT_EYE_OUT[1]) * size / scale
    box = (cx - half, cy - top_off, cx + half, cy - top_off + size / scale)
    return rot.crop(box).resize((size, size), Image.BILINEAR)
