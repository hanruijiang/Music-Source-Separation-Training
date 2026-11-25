"""Audio separation metrics for aligned waveforms."""

from collections.abc import Sequence

import numpy as np


SUPPORTED_METRICS = ("sdr", "si_sdr", "bleedless", "fullness")


def _as_waveform(track: np.ndarray) -> np.ndarray:
    track = np.asarray(track)
    if np.issubdtype(track.dtype, np.integer):
        track = track.astype(np.float64)
    elif not np.issubdtype(track.dtype, np.floating):
        raise TypeError("waveforms must contain real numeric samples")
    elif track.dtype.itemsize < 4:
        track = track.astype(np.float32)
    if not np.isfinite(track).all():
        raise ValueError("waveforms must contain only finite samples")
    return track


def score_tracks(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    metrics: Sequence[str] = ("sdr",),
    sample_rate: int = 44100,
    device: str = "cpu",
) -> dict[str, float]:
    """Score aligned, non-empty, finite waveforms of shape (channels, samples).

    Both tracks must use the same sample rate and amplitude scale. SDR and
    SI-SDR are in dB; bleedless and fullness are spectral scores from 0 to 100.
    Higher is better for all metrics. The device applies to spectral scoring.
    """
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string")
    requested = tuple(dict.fromkeys(metrics))
    unknown = set(requested) - set(SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"unsupported metrics: {', '.join(sorted(unknown))}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    reference = _as_waveform(reference)
    estimate = _as_waveform(estimate)
    if reference.ndim != 2 or reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("reference and estimate must have the same non-empty (channels, samples) shape")

    scores: dict[str, float] = {}
    if "sdr" in requested:
        scores["sdr"] = _sdr(reference, estimate)
    if "si_sdr" in requested:
        scores["si_sdr"] = si_sdr(reference, estimate)
    if "bleedless" in requested or "fullness" in requested:
        scores["bleedless"], scores["fullness"] = bleed_full(
            reference, estimate, sr=sample_rate, device=device
        )
    return {name: scores[name] for name in requested}


def _sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = 1e-8
    references, estimates = reference[None, ...], estimate[None, ...]
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += eps
    den += eps
    return float((10 * np.log10(num / den))[0])


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR over all channels and samples, without demeaning."""
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(
        reference ** 2 + eps, axis=(0, 1)
    )
    target = reference * np.expand_dims(scale, axis=(0, 1))
    num = np.sum(target ** 2, axis=(0, 1))
    den = np.sum((target - estimate) ** 2, axis=(0, 1)) + eps
    return float(10 * np.log10(num / den + eps))


def bleed_full(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 512,
    device: str = "cpu",
) -> tuple[float, float]:
    """Compute bleedless and fullness from Mel magnitude differences in dB."""
    import librosa
    import torch
    from torchaudio.transforms import AmplitudeToDB

    window = torch.hann_window(n_fft).to(device)
    mel = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).to(device)
    to_db = AmplitudeToDB(stype="magnitude", top_db=80)

    def mel_db(track: np.ndarray):
        waveform = torch.from_numpy(track).float().to(device)
        stft = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, window=window,
            return_complex=True, pad_mode="constant",
        )
        return to_db(torch.matmul(mel, stft.abs()))

    diff = mel_db(estimate) - mel_db(reference)
    if not torch.isfinite(diff).all():
        raise ValueError("Mel spectrogram differences must be finite")
    positive = diff[diff > 0]
    negative = diff[diff < 0]
    mean_positive = positive.mean() if positive.numel() else diff.new_tensor(0.0)
    mean_negative = negative.mean() if negative.numel() else diff.new_tensor(0.0)
    return float(100 / (1 + mean_positive)), float(100 / (1 - mean_negative))
