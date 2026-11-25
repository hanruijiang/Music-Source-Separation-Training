"""Backend-independent audio separation metrics."""

from collections.abc import Sequence

import numpy as np


SUPPORTED_METRICS = ("sdr", "si_sdr", "bleedless", "fullness")


def score_tracks(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    metrics: Sequence[str] = ("sdr",),
    sample_rate: int = 44100,
    device: str = "cpu",
) -> dict[str, float]:
    """Score one aligned reference/estimate pair of shape (channels, samples).

    Inputs must be non-empty, real-valued waveforms on the same sample grid
    and amplitude scale. No resampling, cropping or normalization is applied.
    Returns only requested metrics as Python floats (higher is better).
    SDR and SI-SDR are in dB and use NumPy only. Bleedless and fullness
    are 0–100 spectral scores requiring optional Torch/Librosa/Torchaudio.
    sample_rate is the actual input rate, not a resampling target.
    The default remains SDR-only for backward compatibility.

    SDR follows utils/metrics.py at commit 1523206: sum the signal and
    error energies across both axes, add 1e-8 to each, then take 10*log10
    of their ratio. Float32/float64 inputs retain their precision; integers
    and low-precision floats are promoted to avoid overflow or epsilon underflow.
    """
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string")
    requested = tuple(dict.fromkeys(metrics))
    unknown = set(requested) - set(SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"unsupported metrics: {', '.join(sorted(unknown))}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

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

    reference = as_waveform(reference)
    estimate = as_waveform(estimate)
    scores: dict[str, float] = {}

    if "sdr" in requested:
        # Original NumPy SDR reduction from commit 1523206.
        references = reference[None, ...]
        estimates = estimate[None, ...]
        eps = 1e-8
        num = np.sum(np.square(references), axis=(1, 2))
        den = np.sum(np.square(references - estimates), axis=(1, 2))
        num += eps
        den += eps
        scores["sdr"] = float((10 * np.log10(num / den))[0])

    if "si_sdr" in requested:
        scores["si_sdr"] = si_sdr(reference, estimate)

    if "bleedless" in requested or "fullness" in requested:
        # Compute both from a single pair of spectrograms.
        bleedless, fullness = bleed_full(reference, estimate, sr=sample_rate, device=device)
        scores["bleedless"] = bleedless
        scores["fullness"] = fullness

    return {name: scores[name] for name in requested}


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Original main-branch SI-SDR for a single (channels, samples) pair.

    Preserve the original epsilon placement and lack of mean subtraction.
    """
    eps = 1e-8  # To avoid numerical errors
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference ** 2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))  # Broadcast scalar over channels and samples

    reference = reference * scale
    si_sdr = np.mean(10 * np.log10(
        np.sum(reference ** 2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))

    return float(si_sdr)


def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
) -> tuple[float, float]:
    """Original main-branch Mel-spectrogram bleedless/fullness scores.

    Import optional scoring dependencies only when spectral scoring is used.
    """

    try:
        import torch
        import librosa
        from torchaudio.transforms import AmplitudeToDB
    except (ImportError, OSError) as exc:
        raise ImportError(
            "bleedless/fullness require librosa, torch and matching torchaudio; "
            "install the metrics extra or compatible CPU/CUDA packages"
        ) from exc

    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    window = torch.hann_window(n_fft).to(device)

    # Compute STFTs with the Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    diff = S2_db - S1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return float(bleedless.cpu().numpy()), float(fullness.cpu().numpy())
