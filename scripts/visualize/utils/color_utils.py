"""Color utilities: rainbow/HSV colormap matching the reference gradient."""

import numpy as np
from typing import Union


def rainbow_colormap(t: Union[float, np.ndarray]) -> np.ndarray:
    """Map a scalar t ∈ [0, 1] to an RGB color using HSV rainbow.

    Matches the reference color.png gradient:
    cyan → green → yellow → red → magenta → blue (full hue cycle).

    Args:
        t: scalar or (N,) array of values in [0, 1].

    Returns:
        (3,) or (N, 3) RGB values in [0, 1].
    """
    t = np.asarray(t, dtype=np.float64)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    # HSV: hue cycles through the rainbow
    # Reference color.png: cyan → green → yellow → red → magenta → purple
    # Map t=0 → cyan (hue=0.5), t=1 → magenta/purple (hue≈0.83)
    # Use ~0.8 of the hue wheel to avoid wrapping back to cyan
    hue = (0.5 + t * 0.8) % 1.0
    sat = np.ones_like(t)
    val = np.ones_like(t)

    rgb = hsv_to_rgb(hue, sat, val)

    if scalar:
        return rgb[0]
    return rgb


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV to RGB conversion.

    Args:
        h, s, v: (N,) arrays with values in [0, 1].

    Returns:
        (N, 3) RGB array in [0, 1].
    """
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)

    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros((*h.shape, 3))

    mask = i == 0
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)
    mask = i == 1
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)
    mask = i == 2
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)
    mask = i == 3
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)
    mask = i == 4
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)
    mask = i == 5
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

    return rgb


def get_frustum_colors(n: int) -> np.ndarray:
    """Get rainbow colors for n frustums.

    Returns:
        (n, 3) RGB array in [0, 1].
    """
    t = np.linspace(0, 1, n, endpoint=False)
    return rainbow_colormap(t)


def rgb_float_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB [0,1] to uint8 [0,255]."""
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert float RGB [0,1] to hex string like '#ff00aa'."""
    r, g, b = rgb_float_to_uint8(rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"
