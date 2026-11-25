"""CPU-only tests for the shared separation scoring API."""

import numpy as np
import pytest

from utils.metrics import score_tracks


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 2])
def test_sdr_formula(dtype, channels):
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(channels, 257)).astype(dtype)
    estimate = (reference + rng.normal(scale=0.2, size=reference.shape)).astype(dtype)
    refs, ests = reference[None, ...], estimate[None, ...]
    num = np.sum(np.square(refs), axis=(1, 2))
    den = np.sum(np.square(refs - ests), axis=(1, 2))
    num += 1e-8
    den += 1e-8
    expected = float((10 * np.log10(num / den))[0])
    actual = score_tracks(reference, estimate)
    assert actual == {"sdr": expected}
    assert isinstance(actual["sdr"], float)


def test_silence_and_perfect_estimate():
    silence = np.zeros((2, 8), dtype=np.float32)
    reference = np.ones_like(silence)
    assert score_tracks(silence, silence) == {"sdr": 0.0}
    assert score_tracks(reference, silence)["sdr"] == pytest.approx(0.0)
    assert score_tracks(reference, reference)["sdr"] == pytest.approx(
        10 * np.log10((16 + 1e-8) / 1e-8)
    )
    assert score_tracks(silence, reference)["sdr"] < 0


@pytest.mark.parametrize("shapes", [((4,), (4,)), ((1, 4), (2, 4)),
                                     ((2, 4), (2, 3)), ((2, 0), (2, 0)),
                                     ((0, 4), (0, 4)), ((1, 2, 3), (1, 2, 3))])
def test_rejects_invalid_shapes(shapes):
    with pytest.raises(ValueError):
        score_tracks(*(np.zeros(shape) for shape in shapes))


@pytest.mark.parametrize("dtype", [np.int16, np.float16])
def test_low_precision_waveforms(dtype):
    reference = np.full((2, 8), 1000, dtype=dtype)
    estimate = np.full_like(reference, 500)
    assert score_tracks(reference, estimate)["sdr"] == pytest.approx(
        10 * np.log10(4)
    )


def test_rejects_complex_samples():
    with pytest.raises(TypeError):
        score_tracks(np.ones((1, 4), dtype=complex), np.ones((1, 4)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("track_index", [0, 1])
@pytest.mark.parametrize("metric", ["sdr", "si_sdr", "bleedless", "fullness"])
def test_rejects_nonfinite_samples(value, track_index, metric):
    tracks = [np.ones((2, 16), dtype=np.float32) for _ in range(2)]
    tracks[track_index][0, 0] = value
    with pytest.raises(ValueError, match="finite samples"):
        score_tracks(*tracks, metrics=(metric,))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 2])
def test_si_sdr_formula(dtype, channels):
    rng = np.random.default_rng(123)
    ref = rng.normal(size=(channels, 257)).astype(dtype)
    est = (0.5 * ref + rng.normal(scale=0.01, size=ref.shape)).astype(dtype)
    eps = 1e-8
    scale = np.sum(est * ref + eps, axis=(0, 1)) / np.sum(ref ** 2 + eps, axis=(0, 1))
    target = ref * np.expand_dims(scale, axis=(0, 1))
    expected = float(np.mean(10 * np.log10(
        np.sum(target ** 2, axis=(0, 1)) /
        (np.sum((target - est) ** 2, axis=(0, 1)) + eps) + eps)))
    assert score_tracks(ref, est, metrics=("si_sdr",)) == {"si_sdr": expected}


def test_si_sdr_scale_and_silence():
    ref = np.ones((2, 100), dtype=np.float32)
    scores = score_tracks(ref, 0.5 * ref, metrics=("sdr", "si_sdr"))
    assert scores["sdr"] == pytest.approx(6.0206)
    assert scores["si_sdr"] > 90
    silence = np.zeros_like(ref)
    assert score_tracks(silence, silence, metrics=("si_sdr",))["si_sdr"] == pytest.approx(-80)


def test_metric_selection():
    ref = np.ones((1, 16))
    assert list(score_tracks(ref, ref, metrics=("si_sdr", "sdr", "si_sdr"))) == ["si_sdr", "sdr"]
    assert score_tracks(ref, ref, metrics=()) == {}
    with pytest.raises(ValueError, match="unsupported metrics"):
        score_tracks(ref, ref, metrics=("unknown",))
    with pytest.raises(TypeError):
        score_tracks(ref, ref, metrics="sdr")
    with pytest.raises(ValueError, match="sample_rate"):
        score_tracks(ref, ref, sample_rate=0)


def test_numpy_metrics_without_spectral_dependencies(monkeypatch):
    import sys
    for name in ("torch", "librosa", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, None)
    ref = np.ones((1, 16))
    assert set(score_tracks(ref, ref, metrics=("sdr", "si_sdr"))) == {"sdr", "si_sdr"}
    with pytest.raises(ImportError):
        score_tracks(ref, ref, metrics=("fullness",))


def test_spectral_metrics_computed_once(monkeypatch):
    from utils import metrics as module
    calls = []

    def fake_bleed_full(reference, estimate, sr, device):
        calls.append((sr, device))
        return 12.0, 34.0

    monkeypatch.setattr(module, "bleed_full", fake_bleed_full)
    ref = np.ones((1, 16))
    assert score_tracks(ref, ref, metrics=("fullness", "bleedless"), sample_rate=48000) == {
        "fullness": 34.0, "bleedless": 12.0
    }
    assert calls == [(48000, "cpu")]
    assert score_tracks(ref, ref, metrics=("bleedless",)) == {"bleedless": 12.0}


@pytest.mark.parametrize("channels", [1, 2])
def test_spectral_metrics(channels):
    torch = pytest.importorskip("torch")
    librosa = pytest.importorskip("librosa")
    pytest.importorskip("torchaudio")
    from torchaudio.transforms import AmplitudeToDB

    sr = 44100
    rng = np.random.default_rng(23)
    ref = rng.normal(scale=0.1, size=(channels, 8192)).astype(np.float32)
    est = 0.8 * ref + rng.normal(scale=0.02, size=ref.shape).astype(np.float32)
    window = torch.hann_window(4096)
    mel = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=4096, n_mels=512))

    def spectrogram(x):
        stft = torch.stft(torch.from_numpy(x), 4096, 1024, window=window,
                          return_complex=True, pad_mode="constant")
        return AmplitudeToDB(stype="magnitude", top_db=80)(mel @ stft.abs())

    diff = spectrogram(est) - spectrogram(ref)
    pos, neg = diff[diff > 0], diff[diff < 0]
    expected = {
        "bleedless": float(100 / (1 + (pos.mean() if pos.numel() else 0))),
        "fullness": float(100 / (1 - (neg.mean() if neg.numel() else 0))),
    }
    names = ("bleedless", "fullness")
    assert score_tracks(ref, est, metrics=names) == pytest.approx(expected)
    assert score_tracks(ref, ref, metrics=names) == {"bleedless": 100.0, "fullness": 100.0}
    assert score_tracks(ref, ref * 2, metrics=names)["bleedless"] < 100
    assert score_tracks(ref, ref * 0.5, metrics=names)["fullness"] < 100


def test_rejects_nonfinite_spectral_difference():
    pytest.importorskip("torch")
    pytest.importorskip("librosa")
    pytest.importorskip("torchaudio")
    from utils.metrics import bleed_full

    ref = np.ones((1, 4096), dtype=np.float32)
    est = ref.copy()
    est[0, 0] = np.nan
    with pytest.raises(ValueError, match="spectrogram differences must be finite"):
        bleed_full(ref, est)
