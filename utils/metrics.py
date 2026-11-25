"""Backend-independent audio separation metrics (NumPy only)."""

import numpy as np


def score_tracks(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Score one aligned reference/estimate pair of shape (channels, samples).

    Inputs must be non-empty, real-valued waveforms on the same sample grid
    and amplitude scale. No resampling, cropping or normalization is applied.
    Returns {'sdr': float} in dB (higher is better).

    SDR follows utils/metrics.py at commit 1523206: sum the signal and
    error energies across both axes, add 1e-8 to each, then take 10*log10
    of their ratio. Float32/float64 inputs retain their precision; integers
    and low-precision floats are promoted to avoid overflow or epsilon underflow.
    """
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("reference and estimate must have shape (channels, samples)")
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("reference and estimate must have the same non-empty shape")

    def as_waveform(track: np.ndarray) -> np.ndarray:
        if np.issubdtype(track.dtype, np.integer):
            return track.astype(np.float64)
        if not np.issubdtype(track.dtype, np.floating):
            raise TypeError("waveforms must contain real numeric samples")
        if track.dtype.itemsize < 4:
            return track.astype(np.float32)
        return track

    # Add a singleton source axis to reuse the original NumPy SDR reduction.
    references = as_waveform(reference)[None, ...]
    estimates = as_waveform(estimate)[None, ...]
    eps = 1e-8
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += eps
    den += eps
    return {"sdr": float((10 * np.log10(num / den))[0])}
